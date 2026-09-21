"""kobae command line: build | serve | bench | validate."""
from __future__ import annotations

import argparse
from pathlib import Path

DATA = Path("data/malecns_v1")
OUT = Path("outputs/malecns_v1/graph.npz")


def main(argv=None):
    p = argparse.ArgumentParser(prog="kobae", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="feather -> CSR graph.npz")
    b.add_argument("--data", type=Path, default=DATA)
    b.add_argument("--out", type=Path, default=OUT)
    b.add_argument("--no-verify", action="store_true")
    s = sub.add_parser("serve", help="run the brain and the viewer")
    s.add_argument("--graph", type=Path, default=OUT)
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    s.add_argument("--graded", action="store_true", help="stage 2: optic lobe as graded units (gpu only)")
    b2 = sub.add_parser("bench", help="realtime-factor benchmark")
    b2.add_argument("--graph", type=Path, default=OUT)
    b2.add_argument("--backend", choices=["gpu", "cpu", "both"], default="gpu")
    b2.add_argument("--sim-s", type=float, default=2.0)
    b2.add_argument("--out", type=Path, default=None)
    b2.add_argument("--graded", action="store_true")
    v = sub.add_parser("validate", help="GPU vs CPU reference")
    v.add_argument("--graph", type=Path, default=OUT)
    v.add_argument("--protocol", default="sugar")
    v.add_argument("--total-ms", type=float, default=2520)
    v.add_argument("--steady-ms", type=float, default=900)
    args = p.parse_args(argv)
    if args.cmd == "build":
        from .graph import build
        build(args.data, args.out, verify=not args.no_verify)
    elif args.cmd == "serve":
        from .server import serve
        serve(args.graph, args.host, args.port, args.backend, graded=args.graded)
    elif args.cmd == "bench":
        import json, platform, socket
        from .graph import Graph
        from .bench import bench_cpu, bench_gpu
        G = Graph(args.graph)
        res = []
        if args.backend in ("gpu", "both"):
            res += bench_gpu(G, sim_s=args.sim_s, graded=args.graded)
        if args.backend in ("cpu", "both"):
            res += bench_cpu(G, sim_s=min(args.sim_s, 1.0))
        if args.out:
            args.out.write_text(json.dumps({"host": socket.gethostname(), "results": res}, indent=1) + "\n")
    elif args.cmd == "validate":
        import json
        from .graph import Graph
        from .validate import compare
        compare(Graph(args.graph), args.protocol, total_ms=args.total_ms, steady_ms=args.steady_ms)
