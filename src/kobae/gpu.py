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
INDIRECT_OFFSET = 16  # meta[4..6]


class GpuBrain:
    """Full-graph LIF on the GPU.

    ``run(batches)`` advances ``batches * 18`` steps. Spike counts accumulate in a
    per-neuron buffer until ``read_counts(clear=True)``; individual spikes go to a
    log read with ``read_spikes()`` (which also resets the log).
    """

    def __init__(self, graph, log_cap: int = 1 << 22, power_preference: str = "high-performance",
                 adapter_index: int | None = None, verbose: bool = True):
        self.G = graph
        n, E = graph.n, graph.E
        self.n, self.E = n, E
        self.log_cap = log_cap
        adapters = wgpu.gpu.enumerate_adapters_sync()
        if adapter_index is not None:
            adapter = adapters[adapter_index]
        else:
            # prefer a real Vulkan GPU over llvmpipe
            gpu_adapters = [a for a in adapters if a.info.get("adapter_type") in ("DiscreteGPU", "IntegratedGPU")]
            adapter = (gpu_adapters or adapters)[0]
        self.adapter = adapter
        lim = adapter.limits
        need = {
            "max-storage-buffers-per-shader-stage": 12,
            "max-storage-buffer-binding-size": max(lim["max-storage-buffer-binding-size"], (E * 4 + 255) // 256 * 256),
            "max-buffer-size": max(lim["max-buffer-size"], E * 4),
        }
        self.device = adapter.request_device_sync(required_limits=need)
        self.info = adapter.info
        if verbose:
            print(f"[gpu] {self.info.get('device')} ({self.info.get('backend_type')}, {self.info.get('adapter_type')}) "
                  f"driver={self.info.get('description', '')}")
        d = self.device
        B = wgpu.BufferUsage
        ro = B.STORAGE | B.COPY_DST
        rw = B.STORAGE | B.COPY_DST | B.COPY_SRC
        self.b_ptr = d.create_buffer_with_data(data=graph.ptr.astype(np.uint32), usage=ro)
        self.b_post = d.create_buffer_with_data(data=graph.post.astype(np.uint32), usage=ro)
        wq = np.rint(graph.weight.astype(np.float64) * M.G_SCALE).astype(np.int32)
        self.b_wq = d.create_buffer_with_data(data=wq, usage=ro)
        self.b_v = d.create_buffer_with_data(data=np.full(n, M.V_REST, np.float32), usage=rw)
        self.b_g = d.create_buffer(size=n * 4, usage=rw)
        self.b_refr = d.create_buffer(size=n * 4, usage=rw)
        self.b_drive = d.create_buffer(size=n * 4, usage=rw)
        self.b_gin = d.create_buffer(size=M.DELAY_STEPS * n * 4, usage=rw)
        self.b_log = d.create_buffer(size=log_cap * 4, usage=rw)
        self.b_meta = d.create_buffer(size=META_WORDS * 4, usage=rw | B.INDIRECT)
        self.b_counts = d.create_buffer(size=n * 4, usage=rw)
        params = struct.pack("<4I8f", n, M.DELAY_STEPS, log_cap, 0,
                             M.AV, M.AG, M.COUPLING, M.V_REST, M.V_THRESH, float(M.REFRACTORY_STEPS),
                             1.0 / M.G_SCALE, 0.0)
        self.b_params = d.create_buffer_with_data(data=params, usage=B.UNIFORM | B.COPY_DST)

        S = wgpu.ShaderStage.COMPUTE
        def entry(i, typ):
            return {"binding": i, "visibility": S, "buffer": {"type": typ}}
        BT = wgpu.BufferBindingType
        layout = d.create_bind_group_layout(entries=[
            entry(0, BT.uniform), entry(1, BT.read_only_storage), entry(2, BT.read_only_storage),
            entry(3, BT.read_only_storage), entry(4, BT.storage), entry(5, BT.storage), entry(6, BT.storage),
            entry(7, BT.read_only_storage), entry(8, BT.storage), entry(9, BT.storage), entry(10, BT.storage),
            entry(11, BT.storage)])
        bufs = [self.b_params, self.b_ptr, self.b_post, self.b_wq, self.b_v, self.b_g, self.b_refr, self.b_drive,
                self.b_gin, self.b_log, self.b_meta, self.b_counts]
        self.bind_group = d.create_bind_group(layout=layout, entries=[
            {"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}} for i, b in enumerate(bufs)])
        pl = d.create_pipeline_layout(bind_group_layouts=[layout])
        module = d.create_shader_module(code=SHADER.read_text())
        self.p_integrate, self.p_prep, self.p_scatter = [
            d.create_compute_pipeline(layout=pl, compute={"module": module, "entry_point": ep})
            for ep in ("integrate", "prep", "scatter")]
        self.wg_integrate = (n + 63) // 64
        self.sim_steps = 0
        self.total_spikes = 0
        self._wall = 0.0
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

    def set_drive(self, drive: np.ndarray):
        drive = np.ascontiguousarray(drive, dtype=np.float32)
        if drive.shape != (self.n,):
            raise ValueError("drive shape")
        self.device.queue.write_buffer(self.b_drive, 0, drive)

    # ---- stepping --------------------------------------------------------
    def run(self, batches: int, sync: bool = False) -> float:
        """Encode+submit ``batches`` (each 18 steps). Returns wall seconds (only exact with sync=True)."""
        t = time.perf_counter()
        enc = self.device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_bind_group(0, self.bind_group)
        for _ in range(batches):
            cp.set_pipeline(self.p_integrate)
            cp.dispatch_workgroups(self.wg_integrate, 1, 1)
            cp.set_pipeline(self.p_prep)
            cp.dispatch_workgroups(1, 1, 1)
            cp.set_pipeline(self.p_scatter)
            cp.dispatch_workgroups_indirect(self.b_meta, INDIRECT_OFFSET)
        cp.end()
        self.device.queue.submit([enc.finish()])
        if sync:
            self.device.queue.read_buffer(self.b_meta)  # blocks until the queue drains
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

    def read_spikes(self, reset: bool = True) -> tuple[np.ndarray, np.ndarray, bool]:
        """Returns (neuron, substep) for logged spikes since the last reset, plus an overflow flag.

        ``substep`` is relative to the batch the spike was in; batches are in log
        order, so absolute step = batch_index*18 + substep where batch_index is
        recovered by the monotonic substep resets (see ``spike_steps``)."""
        meta = self.read_meta()
        head = int(meta[0])
        over = head > self.log_cap
        m = min(head, self.log_cap)
        packed = np.frombuffer(self.device.queue.read_buffer(self.b_log, 0, m * 4), dtype=np.uint32).copy() if m else np.zeros(0, np.uint32)
        if reset:
            # restart the log; batch start must match head so prep() counts only new spikes
            self.device.queue.write_buffer(self.b_meta, 0, np.zeros(2, np.uint32))
        return packed >> 5, packed & 31, over

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


def spike_steps(neuron: np.ndarray, substep: np.ndarray, first_batch: int = 0) -> np.ndarray:
    """Recover absolute step indices from a batch-ordered spike log.

    Within one batch spikes are appended in substep order per thread but threads
    interleave, so substeps are not monotonic inside a batch; batches themselves
    are sequential. We cannot separate batches from the log alone, hence the
    server reads the log every batch group and passes the batch index."""
    raise NotImplementedError
