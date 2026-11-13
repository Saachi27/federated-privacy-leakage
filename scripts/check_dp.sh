#!/usr/bin/env bash
# Why does DP accuracy barely change between ε=21 and ε=1?
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
PY=${PY:-.venv/bin/python}
JOBS=${JOBS:-3}
SEEDS=${SEEDS:-0}
OUT=results/checks
mkdir -p "$OUT"

arm() {  # arm <name> <flags...>
  local name=$1; shift
  for s in $SEEDS; do
    out=$OUT/dp_${name}_s$s.json
    echo "$out $PY scripts/train_fedavg.py --config configs/mnist.yaml --seed $s --device cpu --no-plot --dp --out $out $*"
  done
}

{ arm noclip_nonoise --dp-noise 0 --dp-clip 1000000
  arm clip1_nonoise --dp-noise 0 --dp-clip 1
  arm sigma1_lr0.03 --dp-noise 1 --dp-clip 1 --lr 0.03
  arm sigma1_lr0.1 --dp-noise 1 --dp-clip 1 --lr 0.1
  arm sigma1_clip2 --dp-noise 1 --dp-clip 2
  arm sigma1_clip5 --dp-noise 1 --dp-clip 5
} | xargs -P "$JOBS" -L 1 bash -c '
  out=$0
  [ -f "$out" ] && { echo "skip $out"; exit 0; }
  echo "start $out"
  "$@" > "$out.log" 2>&1 && echo "done $out" || echo "FAILED $out (see $out.log)"
'
