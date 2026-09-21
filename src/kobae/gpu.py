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



class GpuBrain:
    """Full-graph LIF on the GPU.

    ``run(batches)`` advances ``batches * 18`` steps. Spike counts accumulate in a
    per-neuron buffer until ``read_counts(clear=True)``; individual spikes go to an
    observation ring read with ``read_spikes()`` (which also rewinds it). The
    ring is for observation only: overflowing it drops observations, never
    deliveries.
    """

    def __init__(self, graph, ring_cap: int = 1 << 21, adapter_index: int | None = None, verbose: bool = True):
        self.G = graph
        n, E = graph.n, graph.E
        self.n, self.E = n, E
        self.ring_cap = ring_cap
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
            "max-storage-buffers-per-shader-stage": 13,
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
        params = struct.pack("<4I8f", n, M.DELAY_STEPS, ring_cap, 0,
                             M.AV, M.AG, M.COUPLING, M.V_REST, M.V_THRESH, float(M.REFRACTORY_STEPS),
                             1.0 / M.G_SCALE, 0.0)
        self.b_params = d.create_buffer_with_data(data=params, usage=B.UNIFORM | B.COPY_DST)

        S = wgpu.ShaderStage.COMPUTE
        BT = wgpu.BufferBindingType
        def entry(i, typ):
            return {"binding": i, "visibility": S, "buffer": {"type": typ}}
        layout = d.create_bind_group_layout(entries=[
            entry(0, BT.uniform), entry(1, BT.read_only_storage), entry(2, BT.read_only_storage),
            entry(3, BT.read_only_storage), entry(4, BT.storage), entry(5, BT.storage), entry(6, BT.storage),
            entry(7, BT.read_only_storage), entry(8, BT.storage), entry(9, BT.storage), entry(10, BT.storage),
            entry(11, BT.storage), entry(12, BT.storage)])
        bufs = [self.b_params, self.b_ptr, self.b_post, self.b_wq, self.b_v, self.b_g, self.b_refr, self.b_drive,
                self.b_gin, self.b_deliver, self.b_meta, self.b_counts, self.b_ring]
        self.bind_group = d.create_bind_group(layout=layout, entries=[
            {"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}} for i, b in enumerate(bufs)])
        layout1 = d.create_bind_group_layout(entries=[entry(0, BT.storage)])
        self.bind_group1 = d.create_bind_group(layout=layout1, entries=[
            {"binding": 0, "resource": {"buffer": self.b_indirect, "offset": 0, "size": 16}}])
        pl0 = d.create_pipeline_layout(bind_group_layouts=[layout])
        pl1 = d.create_pipeline_layout(bind_group_layouts=[layout, layout1])
        module = d.create_shader_module(code=SHADER.read_text())
        mk = lambda pl, ep: d.create_compute_pipeline(layout=pl, compute={"module": module, "entry_point": ep})
        self.p_integrate, self.p_scatter, self.p_finish = [mk(pl0, ep) for ep in ("integrate", "scatter", "finish")]
        self.p_prep = mk(pl1, "prep")
        self.wg_integrate = (n + 63) // 64
        self.sim_steps = 0
        self.total_spikes = 0
        self._wall = 0.0
        self._ring_base = 0  # ring entries already consumed (host side)
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
    def run(self, batches: int, sync: bool = True) -> float:
        """Encode+submit ``batches`` (each 18 steps). Returns wall seconds (exact only with sync)."""
        t = time.perf_counter()
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
            cp.set_pipeline(self.p_finish)
            cp.dispatch_workgroups(1, 1, 1)
        cp.end()
        self.device.queue.submit([enc.finish()])
        if sync:
            self.device.queue.read_buffer(self.b_meta, 0, 4)  # blocks until the queue drains
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
