"""wgpu (Vulkan) whole-CNS LIF simulator. See shaders/lif.wgsl for the model."""
from __future__ import annotations

import struct
import time
from pathlib import Path

import numpy as np
import wgpu

from . import model as M

SHADER = Path(__file__).parent / "shaders" / "lif.wgsl"
META_WORDS = 8
# Never let one command buffer run long: amdgpu resets the GPU when a single job exceeds its
# ring timeout (~10 s; on 2026-09-21 a 1111-batch submit with the graded gather did exactly
# that on the RX 580 and took the desktop down). Submits are chunked and each chunk is waited
# on; the chunk size adapts so a submit stays around SUBMIT_TARGET_S.
MAX_BATCHES_PER_SUBMIT = 50
SUBMIT_TARGET_S = 0.25



class GpuBrain:
    """Full-graph LIF on the GPU.

    ``run(batches)`` advances ``batches * 18`` steps. Spike counts accumulate in a
    per-neuron buffer until ``read_counts(clear=True)``; individual spikes go to an
    observation ring read with ``read_spikes()`` (which also rewinds it). The
    ring is for observation only: overflowing it drops observations, never
    deliveries.
    """

    def __init__(self, graph, ring_cap: int = 1 << 21, adapter_index: int | None = None, verbose: bool = True,
                 graded: np.ndarray | None = None, graded_fmax_hz: float = 200.0, graded_every: int = 5):
        """``graded``: boolean mask of cells simulated as graded (non-spiking) units; None = all spiking.
        ``graded_every``: run the (expensive, 11 M-edge) graded gather every this many batches; the
        transmitted charge is scaled by the same factor. 5 batches = 9 ms, well under tau_m = 20 ms."""
        self.graded_every = max(1, int(graded_every))
        self.G = graph
        n, E = graph.n, graph.E
        self.n, self.E = n, E
        self.ring_cap = ring_cap
        self.graded = np.zeros(n, bool) if graded is None else np.asarray(graded, bool)
        self.graded_fmax = graded_fmax_hz
        adapters = wgpu.gpu.enumerate_adapters_sync()
        if adapter_index is not None:
            adapter = adapters[adapter_index]
        else:
            gpu_adapters = [a for a in adapters if a.info.get("adapter_type") in ("DiscreteGPU", "IntegratedGPU")]
            adapter = (gpu_adapters or adapters)[0]
        self.adapter = adapter
        lim = adapter.limits
        big = max(E * 4, M.DELAY_STEPS * n * 4, ring_cap * 8)
        need = {
            "max-storage-buffers-per-shader-stage": 18,
            "max-storage-buffer-binding-size": max(lim["max-storage-buffer-binding-size"], (big + 255) // 256 * 256),
            "max-buffer-size": max(lim["max-buffer-size"], big),
        }
        self.device = adapter.request_device_sync(required_limits=need)
        self.info = adapter.info
        if verbose:
            print(f"[gpu] {self.info.get('device')} ({self.info.get('backend_type')}, {self.info.get('adapter_type')})")
        d = self.device
        B = wgpu.BufferUsage
        ro = B.STORAGE | B.COPY_DST
        rw = B.STORAGE | B.COPY_DST | B.COPY_SRC
        self.b_ptr = d.create_buffer_with_data(data=np.ascontiguousarray(graph.ptr, dtype=np.uint32), usage=ro)
        self.b_post = d.create_buffer_with_data(data=np.ascontiguousarray(graph.post, dtype=np.uint32), usage=ro)
        wq = np.rint(graph.weight.astype(np.float64) * M.G_SCALE).astype(np.int32)
        self.b_wq = d.create_buffer_with_data(data=wq, usage=ro)
        self.b_v = d.create_buffer(size=n * 4, usage=rw)
        self.b_g = d.create_buffer(size=n * 4, usage=rw)
        self.b_refr = d.create_buffer(size=n * 4, usage=rw)
        self.b_drive = d.create_buffer(size=n * 4, usage=rw)
        self.b_gin = d.create_buffer(size=M.DELAY_STEPS * n * 4, usage=rw)
        self.b_deliver = d.create_buffer(size=n * 4, usage=rw)
        self.b_meta = d.create_buffer(size=META_WORDS * 4, usage=rw)
        self.b_indirect = d.create_buffer(size=16, usage=rw | B.INDIRECT)
        self.b_counts = d.create_buffer(size=n * 4, usage=rw)
        self.b_ring = d.create_buffer(size=ring_cap * 8, usage=rw)
        # graded pathway: CSC restricted to graded presynaptic cells (weights already carry sign)
        gm = self.graded
        if gm.any() and hasattr(graph, "cptr"):
            keep = gm[graph.cpre]
            gdeg = np.bincount(np.repeat(np.arange(n), np.diff(graph.cptr))[keep], minlength=n)
            gptr = np.r_[0, np.cumsum(gdeg)].astype(np.uint32)
            gpre = graph.cpre[keep].astype(np.uint32)
            gwq = np.rint(graph.cweight[keep].astype(np.float64) * M.G_SCALE).astype(np.int32)
        else:
            gptr = np.zeros(n + 1, np.uint32); gpre = np.zeros(1, np.uint32); gwq = np.zeros(1, np.int32)
        self.n_graded_edges = int(len(gpre) if gm.any() else 0)
        self.b_kind = d.create_buffer_with_data(data=gm.astype(np.uint32), usage=ro)
        self.b_rate = d.create_buffer(size=n * 4, usage=rw)
        self.b_gptr = d.create_buffer_with_data(data=gptr, usage=ro)
        self.b_gpre = d.create_buffer_with_data(data=gpre, usage=ro)
        self.b_gwq = d.create_buffer_with_data(data=gwq, usage=ro)
        # continuous transmission at rate 1 == f_max spikes/s, lumped once per `graded_every` batches
        graded_gain = float(self.graded_fmax * M.DELAY_STEPS * M.DT_MS / 1000.0 * self.graded_every)
        params = struct.pack("<4I8f", n, M.DELAY_STEPS, ring_cap, 0,
                             M.AV, M.AG, M.COUPLING, M.V_REST, M.V_THRESH, float(M.REFRACTORY_STEPS),
                             1.0 / M.G_SCALE, graded_gain)
        self.b_params = d.create_buffer_with_data(data=params, usage=B.UNIFORM | B.COPY_DST)

        S = wgpu.ShaderStage.COMPUTE
        BT = wgpu.BufferBindingType
        def entry(i, typ):
            return {"binding": i, "visibility": S, "buffer": {"type": typ}}
        layout = d.create_bind_group_layout(entries=[
            entry(0, BT.uniform), entry(1, BT.read_only_storage), entry(2, BT.read_only_storage),
            entry(3, BT.read_only_storage), entry(4, BT.storage), entry(5, BT.storage), entry(6, BT.storage),
            entry(7, BT.read_only_storage), entry(8, BT.storage), entry(9, BT.storage), entry(10, BT.storage),
            entry(11, BT.storage), entry(12, BT.storage), entry(13, BT.read_only_storage), entry(14, BT.storage),
            entry(15, BT.read_only_storage), entry(16, BT.read_only_storage), entry(17, BT.read_only_storage)])
        bufs = [self.b_params, self.b_ptr, self.b_post, self.b_wq, self.b_v, self.b_g, self.b_refr, self.b_drive,
                self.b_gin, self.b_deliver, self.b_meta, self.b_counts, self.b_ring,
                self.b_kind, self.b_rate, self.b_gptr, self.b_gpre, self.b_gwq]
        self.bind_group = d.create_bind_group(layout=layout, entries=[
            {"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}} for i, b in enumerate(bufs)])
        layout1 = d.create_bind_group_layout(entries=[entry(0, BT.storage)])
        self.bind_group1 = d.create_bind_group(layout=layout1, entries=[
            {"binding": 0, "resource": {"buffer": self.b_indirect, "offset": 0, "size": 16}}])
        pl0 = d.create_pipeline_layout(bind_group_layouts=[layout])
        pl1 = d.create_pipeline_layout(bind_group_layouts=[layout, layout1])
        module = d.create_shader_module(code=SHADER.read_text())
        mk = lambda pl, ep: d.create_compute_pipeline(layout=pl, compute={"module": module, "entry_point": ep})
        self.p_integrate, self.p_scatter, self.p_finish, self.p_graded = [mk(pl0, ep) for ep in ("integrate", "scatter", "finish", "graded_gather")]
        self.p_prep = mk(pl1, "prep")
        self.wg_integrate = (n + 63) // 64
        self.sim_steps = 0
        self.total_spikes = 0
        self._wall = 0.0
        self._ring_base = 0  # ring entries already consumed (host side)
        self.chunk = 8       # batches per submit, adapted from measured submit time
        self._batch_no = 0
        self.reset_state()

    # ---- state -----------------------------------------------------------
    def reset_state(self):
        q = self.device.queue
        n = self.n
        q.write_buffer(self.b_v, 0, np.full(n, M.V_REST, np.float32))
        q.write_buffer(self.b_g, 0, np.zeros(n, np.float32))
        q.write_buffer(self.b_refr, 0, np.zeros(n, np.int32))
        q.write_buffer(self.b_gin, 0, np.zeros(M.DELAY_STEPS * n, np.int32))
        q.write_buffer(self.b_meta, 0, np.zeros(META_WORDS, np.uint32))
        q.write_buffer(self.b_counts, 0, np.zeros(n, np.uint32))
        q.write_buffer(self.b_rate, 0, np.zeros(n, np.float32))
        self.sim_steps = 0
        self.total_spikes = 0
        self._wall = 0.0
        self._ring_base = 0

    def set_drive(self, drive: np.ndarray):
        drive = np.ascontiguousarray(drive, dtype=np.float32)
        if drive.shape != (self.n,):
            raise ValueError("drive shape")
        self.device.queue.write_buffer(self.b_drive, 0, drive)

    # ---- stepping --------------------------------------------------------
    def _submit(self, batches: int):
        enc = self.device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_bind_group(0, self.bind_group)
        for _ in range(batches):
            cp.set_pipeline(self.p_integrate)
            cp.dispatch_workgroups(self.wg_integrate, 1, 1)
            cp.set_pipeline(self.p_prep)
            cp.set_bind_group(1, self.bind_group1)
            cp.dispatch_workgroups(1, 1, 1)
            cp.set_pipeline(self.p_scatter)
            cp.dispatch_workgroups_indirect(self.b_indirect, 0)
            if self.n_graded_edges and self._batch_no % self.graded_every == 0:
                cp.set_pipeline(self.p_graded)
                cp.dispatch_workgroups(self.wg_integrate, 1, 1)
            cp.set_pipeline(self.p_finish)
            cp.dispatch_workgroups(1, 1, 1)
            self._batch_no += 1
        cp.end()
        self.device.queue.submit([enc.finish()])
        self.device.queue.read_buffer(self.b_meta, 0, 4)  # blocks until this submit has completed

    def run(self, batches: int, sync: bool = True) -> float:
        """Advance ``batches`` (each 18 steps) in bounded submits. Returns wall seconds.

        ``sync`` is accepted for API compatibility; every chunk is always waited on."""
        t = time.perf_counter()
        done = 0
        while done < batches:
            k = min(self.chunk, batches - done)
            t0 = time.perf_counter()
            self._submit(k)
            dt = time.perf_counter() - t0
            done += k
            # adapt: aim for SUBMIT_TARGET_S per submit, never above the hard cap
            if dt > 0:
                est = int(k * SUBMIT_TARGET_S / dt)
                self.chunk = int(np.clip(est, 1, MAX_BATCHES_PER_SUBMIT))
        el = time.perf_counter() - t
        self.sim_steps += batches * M.DELAY_STEPS
        self._wall += el
        return el

    def run_steps(self, steps: int, sync: bool = True) -> float:
        if steps % M.DELAY_STEPS:
            raise ValueError(f"steps must be a multiple of {M.DELAY_STEPS}")
        return self.run(steps // M.DELAY_STEPS, sync=sync)

    # ---- readback --------------------------------------------------------
    def read_meta(self) -> np.ndarray:
        return np.frombuffer(self.device.queue.read_buffer(self.b_meta), dtype=np.uint32).copy()

    def read_counts(self, clear: bool = True) -> np.ndarray:
        c = np.frombuffer(self.device.queue.read_buffer(self.b_counts), dtype=np.uint32).copy()
        if clear:
            self.device.queue.write_buffer(self.b_counts, 0, np.zeros(self.n, np.uint32))
        self.total_spikes += int(c.sum())
        return c

    def read_spikes(self):
        """Returns (step, neuron) int64 arrays for spikes logged since the last call, and an overflow flag.

        Absolute step = batch*18 + substep. The ring is rewound after reading."""
        meta = self.read_meta()
        head = int(meta[4]); over = bool(meta[5])
        m = min(head, self.ring_cap)
        if m:
            raw = np.frombuffer(self.device.queue.read_buffer(self.b_ring, 0, m * 8), dtype=np.uint32).reshape(-1, 2)
            step = raw[:, 0].astype(np.int64) * M.DELAY_STEPS + (raw[:, 1] & 31)
            neuron = (raw[:, 1] >> 5).astype(np.int64)
        else:
            step = neuron = np.zeros(0, np.int64)
        self.device.queue.write_buffer(self.b_meta, 16, np.zeros(2, np.uint32))  # head, overflow
        return step, neuron, over

    def read_rate(self) -> np.ndarray:
        return np.frombuffer(self.device.queue.read_buffer(self.b_rate), dtype=np.float32).copy()

    def read_state(self):
        q = self.device.queue
        v = np.frombuffer(q.read_buffer(self.b_v), dtype=np.float32).copy()
        g = np.frombuffer(q.read_buffer(self.b_g), dtype=np.float32).copy()
        r = np.frombuffer(q.read_buffer(self.b_refr), dtype=np.int32).copy()
        return v, g, r

    @property
    def sim_ms(self) -> float:
        return self.sim_steps * M.DT_MS

    @property
    def wall_s(self) -> float:
        return self._wall
