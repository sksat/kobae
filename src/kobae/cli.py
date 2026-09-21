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
    args = p.parse_args(argv)
    if args.cmd == "build":
        from .graph import build
        build(args.data, args.out, verify=not args.no_verify)
    elif args.cmd == "serve":
        from .server import serve
        serve(args.graph, args.host, args.port, args.backend)
