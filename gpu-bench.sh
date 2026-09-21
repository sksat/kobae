#!/bin/bash
# Wait until BC-250 is free (parent batch, llama tools, lock), then take the lock and bench kobae.
export PATH=~/.local/bin:$PATH
cd ~/prog/kobae
until [ -x .venv/bin/python ] && .venv/bin/python -c "import wgpu, numba" 2>/dev/null; do sleep 20; done   # uv sync may still be running
echo "$(date -Is) venv ready"
busy() {
  pgrep -f "[g]pu-batch" >/dev/null && return 0
  for p in llama-bench llama-server sd-cli transcribe-cli llama-diffusion-cli llama-embedding llama-mtmd-cli llama-perplexity llama-cli; do
    pgrep -f "[/]$p( |$)" >/dev/null && return 0
  done
  [ -e /tmp/bc250-gpu.lock ] && return 0
  return 1
}
until ! busy; do sleep 30; done
echo "kobae bench $(date -Is)" > /tmp/bc250-gpu.lock
echo "$(date -Is) lock taken"
trap "rm -f /tmp/bc250-gpu.lock" EXIT
uv run --no-sync kobae bench --backend both --sim-s 2 --out outputs/bench-gpu.json 2>&1 | tee outputs/bench-gpu.log
uv run --no-sync kobae validate --protocol visual --total-ms 1800 --steady-ms 900 2>&1 | tee outputs/validate-gpu-visual.log
rm -f /tmp/bc250-gpu.lock
echo "=== KOBAE GPU DONE ==="
