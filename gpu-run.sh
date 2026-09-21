#!/bin/bash
# Run a kobae command on BC-250 under the GPU lock protocol. Usage: gpu-run.sh <label> <cmd...>
export PATH=~/.local/bin:$PATH
cd ~/prog/kobae
label=$1; shift
busy() {
  pgrep -f "[g]pu-batch" >/dev/null && return 0
  for p in llama-bench llama-server sd-cli transcribe-cli llama-diffusion-cli llama-embedding llama-mtmd-cli llama-perplexity llama-cli; do
    pgrep -f "[/]$p( |$)" >/dev/null && return 0
  done
  [ -e /tmp/bc250-gpu.lock ] && return 0
  return 1
}
until ! busy; do sleep 30; done
echo "kobae $label $(date -Is)" > /tmp/bc250-gpu.lock
trap "rm -f /tmp/bc250-gpu.lock" EXIT
echo "$(date -Is) lock taken: $label"
"$@"
rc=$?
rm -f /tmp/bc250-gpu.lock
echo "=== KOBAE $label DONE rc=$rc ==="
