"""aiohttp server: runs the brain in a thread and streams activity to the viewer.

Endpoints
  GET  /                      viewer (built with `pnpm build` in viewer/)
  GET  /api/meta              JSON: n, superclasses, readouts, buttons, orn glomeruli, stats
  GET  /api/positions         binary float32 [n,3] (µm)
  GET  /api/superclass        binary uint8 [n] superclass index
  GET  /api/cell/<i>          JSON: info about one cell
  GET  /api/search?q=glob     JSON: cell types matching a glob (count per type)
  WS   /ws                    server -> client binary frames; client -> server JSON commands

Frame (binary, little endian):
  u32 magic 0x4B4F4241 ('KOBA'), f32 sim_ms, f32 realtime_x, u32 total_spikes_in_frame,
  u32 n_ids, u32 n_sc, u32 n_ro, u32 n_ret,
  u32[n_ids] spiking neuron ids (unique within the frame),
  f32[n_sc]  per-superclass rate (Hz per cell), f32[n_ro] readout rates (Hz), f32[n_ret] retina luminance
"""
from __future__ import annotations

import asyncio
import json
import struct
import threading
import time
from pathlib import Path

import numpy as np
from aiohttp import web

from . import model as M
from .graph import Graph
from .positions import compute_positions
from .stimulus import BUTTONS, Stimulator, custom_indices

MAGIC = 0x4B4F4241
FRAME_STEPS = M.DELAY_STEPS * 2  # 3.6 ms of simulation per frame


class Engine(threading.Thread):
    def __init__(self, G: Graph, backend: str = "cpu", fps_cap: float = 60.0):
        super().__init__(daemon=True)
        self.G = G
        self.backend = backend
        if backend == "gpu":
            from .gpu import GpuBrain
            self.brain = GpuBrain(G)
        else:
            from .cpu import CpuBrain
            self.brain = CpuBrain(G)
        self.stim = Stimulator(G)
        self.paused = False
        self.lock = threading.Lock()
        self.frames: asyncio.Queue | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.sim_ms = 0.0
        self.rt = 0.0
        self.rate_tau = 0.1  # s, exponential window for rate estimates
        sc = G.superclass
        self.sc_names = sorted(set(sc.tolist()))
        self.sc_index = np.array([self.sc_names.index(s) for s in sc], dtype=np.uint8)
        self.sc_count = np.bincount(self.sc_index, minlength=len(self.sc_names)).astype(np.float32)
        self.sc_rate = np.zeros(len(self.sc_names), np.float32)
        self.readouts = G.meta.get("readouts", [])
        self.ro_idx = np.array([r["index"] for r in self.readouts], dtype=np.int64)
        self.ro_rate = np.zeros(len(self.ro_idx), np.float32)
        self.cell_rate = np.zeros(G.n, np.float32)
        self.fps_cap = fps_cap
        self._dirty = True
        self._last_stim_t = -1.0

    def apply(self, patch: dict):
        with self.lock:
            self.stim.set(patch)
            self._dirty = True

    def run(self):
        b = self.brain
        dt_frame = FRAME_STEPS * M.DT_MS / 1000.0
        t_wall_prev = time.perf_counter()
        while True:
            if self.paused:
                time.sleep(0.05); t_wall_prev = time.perf_counter(); continue
            t0 = time.perf_counter()
            with self.lock:
                vis_dyn = self.stim.state.get("visual") in ("flash", "bar", "grating", "loom")
                if self._dirty or (vis_dyn and self.sim_ms - self._last_stim_t >= 18.0):
                    b.set_drive(self.stim.drive(self.sim_ms / 1000.0))
                    self._dirty = False; self._last_stim_t = self.sim_ms
                lum = self.stim.retina_lum.copy()
            if self.backend == "gpu":
                b.run_steps(FRAME_STEPS, sync=True)
                counts = b.read_counts(clear=True)
            else:
                counts, _, _ = b.run(FRAME_STEPS)
            self.sim_ms += FRAME_STEPS * M.DT_MS
            el = time.perf_counter() - t0
            self.rt = 0.9 * self.rt + 0.1 * (dt_frame / max(el, 1e-6)) if self.rt else dt_frame / max(el, 1e-6)
            # rates: exponential moving average of spikes/s per cell
            a = dt_frame / self.rate_tau
            self.cell_rate *= (1 - a)
            self.cell_rate += counts.astype(np.float32) / dt_frame * a
            np.add.at(self.sc_rate, np.arange(len(self.sc_names)), 0)  # keep dtype
            self.sc_rate = np.bincount(self.sc_index, weights=self.cell_rate, minlength=len(self.sc_names)).astype(np.float32) / self.sc_count
            self.ro_rate = self.cell_rate[self.ro_idx].astype(np.float32)
            ids = np.flatnonzero(counts).astype(np.uint32)
            frame = struct.pack("<IffIIIII", MAGIC, self.sim_ms, self.rt, int(counts.sum()),
                                len(ids), len(self.sc_rate), len(self.ro_rate), len(lum)) \
                + ids.tobytes() + self.sc_rate.tobytes() + self.ro_rate.tobytes() + lum.astype(np.float32).tobytes()
            if self.loop is not None and self.frames is not None:
                self.loop.call_soon_threadsafe(self._offer, frame)
            # do not out-run the display; cap frames/s
            budget = 1.0 / self.fps_cap
            if el < budget:
                time.sleep(budget - el)

    def _offer(self, frame):
        q = self.frames
        if q.full():
            try: q.get_nowait()
            except asyncio.QueueEmpty: pass
        q.put_nowait(frame)


def make_app(G: Graph, engine: Engine, static_dir: Path) -> web.Application:
    pos, measured = compute_positions(G)
    sc_u8 = engine.sc_index
    clients: set[web.WebSocketResponse] = set()

    async def index(request):
        f = static_dir / "index.html"
        if not f.exists():
            return web.Response(text="viewer not built: run `pnpm install && pnpm build` in viewer/", status=503)
        return web.FileResponse(f)

    async def meta(request):
        return web.json_response({
            "n": G.n, "edges": G.E, "superclasses": engine.sc_names,
            "superclass_counts": engine.sc_count.astype(int).tolist(),
            "readouts": engine.readouts, "buttons": {k: v[0] for k, v in BUTTONS.items()},
            "orn": {k: int(len(v)) for k, v in engine.stim.orn.items()},
            "measured_positions": int(measured.sum()), "backend": engine.backend,
            "retina": int(len(G.retina)), "uv": G.uv.astype(np.float32).ravel().tolist(),
            "state": engine.stim.state, "dt_ms": M.DT_MS, "frame_ms": FRAME_STEPS * M.DT_MS,
        })

    async def positions(request):
        return web.Response(body=pos.tobytes(), content_type="application/octet-stream")

    async def superclass(request):
        return web.Response(body=sc_u8.tobytes(), content_type="application/octet-stream")

    async def cell(request):
        i = int(request.match_info["i"])
        if not 0 <= i < G.n:
            raise web.HTTPNotFound()
        return web.json_response({
            "index": i, "bodyId": int(G.ids[i]), "type": str(G.cell_type[i]), "superclass": str(G.superclass[i]),
            "side": str(G.soma_side[i]), "nt": str(G.neurotransmitter[i]), "sign": int(G.sign[i]),
            "out_degree": int(G.ptr[i + 1] - G.ptr[i]), "in_degree": int(np.count_nonzero(G.post == i)) if G.E < 5e7 else -1,
            "rate_hz": float(engine.cell_rate[i]), "measured_soma": bool(measured[i]),
        })

    async def search(request):
        q = request.query.get("q", "")
        idx = custom_indices(G, q) if q else np.zeros(0, np.int64)
        types, counts = np.unique(G.cell_type[idx], return_counts=True) if len(idx) else ([], [])
        return web.json_response({"glob": q, "n": int(len(idx)),
                                  "types": [{"type": str(t), "n": int(c)} for t, c in zip(types, counts)][:200],
                                  "ids": idx[:5000].astype(int).tolist()})

    async def ws(request):
        sock = web.WebSocketResponse(max_msg_size=64 << 20)
        await sock.prepare(request)
        clients.add(sock)
        await sock.send_str(json.dumps({"op": "state", "state": engine.stim.state, "paused": engine.paused}))
        try:
            async for msg in sock:
                if msg.type == web.WSMsgType.TEXT:
                    cmd = json.loads(msg.data)
                    if cmd.get("op") == "stim":
                        engine.apply(cmd.get("patch", {}))
                    elif cmd.get("op") == "pause":
                        engine.paused = bool(cmd.get("value"))
                    elif cmd.get("op") == "reset":
                        engine.paused = True; time.sleep(0.1)
                        if engine.backend == "gpu":
                            engine.brain.reset_state()
                        else:
                            from .cpu import CpuBrain
                            engine.brain = CpuBrain(G)
                        engine.cell_rate[:] = 0; engine.sim_ms = 0.0; engine._dirty = True
                        engine.paused = False
                    msg_state = json.dumps({"op": "state", "state": engine.stim.state, "paused": engine.paused})
                    for c in list(clients):
                        try: await c.send_str(msg_state)
                        except Exception: pass
        finally:
            clients.discard(sock)
        return sock

    async def broadcaster(app):
        engine.loop = asyncio.get_running_loop()
        engine.frames = asyncio.Queue(maxsize=4)
        while True:
            frame = await engine.frames.get()
            dead = []
            for c in list(clients):
                try:
                    await c.send_bytes(frame)
                except Exception:
                    dead.append(c)
            for c in dead:
                clients.discard(c)

    async def on_startup(app):
        app["bcast"] = asyncio.create_task(broadcaster(app))

    app = web.Application()
    app.on_startup.append(on_startup)
    app.router.add_get("/", index)
    app.router.add_get("/api/meta", meta)
    app.router.add_get("/api/positions", positions)
    app.router.add_get("/api/superclass", superclass)
    app.router.add_get("/api/cell/{i}", cell)
    app.router.add_get("/api/search", search)
    app.router.add_get("/ws", ws)
    if static_dir.exists():
        app.router.add_static("/assets", static_dir / "assets")
    return app


def serve(graph_path: Path, host: str, port: int, backend: str):
    G = Graph(graph_path)
    engine = Engine(G, backend=backend)
    engine.start()
    static_dir = Path(__file__).parent / "viewer" / "dist"
    app = make_app(G, engine, static_dir)
    print(f"[kobae] serving on http://{host}:{port}  backend={backend}  n={G.n} E={G.E}", flush=True)
    web.run_app(app, host=host, port=port, print=None)
