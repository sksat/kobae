#!/bin/bash
# Viewer on BC-250 (GPU backend). Holds the GPU lock while running; stop with: pkill -f "[k]obae serve"
export PATH=~/.local/bin:$PATH
cd ~/prog/kobae
exec ./gpu-run.sh serve uv run --no-sync kobae serve --backend gpu --port 8765 "$@"
