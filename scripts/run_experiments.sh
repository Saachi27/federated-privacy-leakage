#!/usr/bin/env bash
# Reproduce every number and figure in the report.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
PY=${PY:-.venv/bin/python}
JOBS=${JOBS:-4}  # one per performance core; more runs slower overall on an M1
SEEDS="0 1 2"
STUDY_SEEDS="0 1 2 3 4"
# The cosine attacker's TV weight
COSINE_TV=${COSINE_TV:-1e-3}
RUNS=results/runs
mkdir -p "$RUNS"

# Reads one "output-file command..." job per line and runs them in parallel
parallel() {
  xargs -P "$JOBS" -L 1 bash -c '
    out=$0
    [ -f "$out" ] && { echo "skip $out"; exit 0; }
    echo "start $out"
    "$@" > "$out.log" 2>&1 && echo "done $out" || echo "FAILED $out (see $out.log)"
  '
}

train() {  # train <name> <extra flags...>
  local name=$1; shift
  for s in $SEEDS; do
    out=$RUNS/train_${name}_s$s.json
    echo "$out $PY scripts/train_fedavg.py --seed $s --device cpu --no-plot --out $out $*"
  done
}

attack() {  # attack <model>
  for s in $SEEDS; do
    out=$RUNS/attack_$1_s$s.json
    echo "$out $PY scripts/attack_vs_training.py --model $1 --seed $s --cosine-tv $COSINE_TV --out $out"
  done
}

evaluate() {  # evaluate <done-marker> <flags...>
  local marker=$1; shift
  [ -f "$marker" ] && { echo "skip $marker"; return; }
  echo "start $marker"
  "$PY" scripts/evaluate.py --seeds $SEEDS --workers "$JOBS" --cosine-tv "$COSINE_TV" "$@" > "$marker.log" 2>&1 \
    && echo "done $marker" || echo "FAILED $marker (see $marker.log)"
}

step() {  # step <log> <command...>: run one study step; the scripts skip finished seeds
  local log=$1; shift
  echo "start $log"
  "$@" > "$log" 2>&1 && echo "done $log" || echo "FAILED $log"
}

study() {
  # 1.
  for s in 0 1 2; do
    out=results/checks/cosine_heldout_s$s.json
    [ -f "$out" ] && { echo "skip $out"; continue; }
    step "$out.log" "$PY" scripts/validate_cosine.py --first-image 25 --num-images 25 --seed "$s" \
      --arms "current (lr 0.1, 3000 iters, TV 1e-4)" "no TV" "TV 1e-3" "TV 3e-3" "TV 1e-2" \
      --workers "$JOBS" --out "$out"
  done
  # 2.
  step $RUNS/sum.out "$PY" scripts/attack_sum.py --ks 1 2 4 8 16 --num-images 32 \
    --seeds $STUDY_SEEDS --cosine-tv "$COSINE_TV" --workers "$JOBS" --out $RUNS/sum
  step $RUNS/sum_long.out "$PY" scripts/attack_sum.py --ks 8 16 --num-images 32 --iters 1200 \
    --cosine-iters 12000 --seeds $STUDY_SEEDS --cosine-tv "$COSINE_TV" --workers "$JOBS" --out $RUNS/sum_long
  # 3.
  step $RUNS/noise.out "$PY" scripts/attack_noise_sum.py --ks 1 2 4 8 --num-images 16 \
    --seeds $STUDY_SEEDS --cosine-tv "$COSINE_TV" --workers "$JOBS" --out $RUNS/noise
  step $RUNS/matched.out "$PY" scripts/attack_matched_snr.py --ks 2 4 8 --sigmas 1e-4 3e-4 1e-3 3e-3 \
    --num-images 16 --seeds $STUDY_SEEDS --cosine-tv "$COSINE_TV" --workers "$JOBS" --out $RUNS/matched
  # 4.
  step results/baselines.log "$PY" scripts/baselines.py --first-image 50 --num-images 32 --out results/baselines
  step results/baselines_noise.log "$PY" scripts/baselines.py --first-image 50 --num-images 16 \
    --ks 1 2 4 8 --out results/baselines_noise
  step results/baselines_cifar10.log "$PY" scripts/baselines_cifar10.py
  # 5.
  JOBS=$JOBS SEEDS="$SEEDS" PY=$PY bash scripts/check_dp.sh
  # 6.
  step results/rq4.log "$PY" scripts/analyze_rq4.py
  step results/plot_final.log "$PY" scripts/plot_final.py
  step results/plot_dp.log "$PY" scripts/plot_dp.py
  step results/examples.log "$PY" scripts/make_examples_final.py --workers "$JOBS"
}

core() {
  { train mnist_baseline --config configs/mnist.yaml
    train mnist_mask --config configs/mnist.yaml --mask
    train mnist_dp1 --config configs/mnist.yaml --dp --dp-noise 1.0
    attack lenet
    attack smallcnn
  } | parallel
  evaluate results/evaluation.md --num-images 50 --out results/evaluation
  evaluate results/evaluation_sweep.md --num-images 25 --attackers adaptive cosine \
    --dp-sweep 0 0.0001 0.001 0.003 0.01 0.1 1.0 --out results/evaluation
  "$PY" scripts/make_gallery.py --dataset mnist --out results/reconstruction_gallery.png
  "$PY" scripts/run_privacy_eval.py --dataset mnist --index 1 --out results/privacy_eval
}

extras() {
  { train mnist_dp0.5 --config configs/mnist.yaml --dp --dp-noise 0.5
    train mnist_dp2 --config configs/mnist.yaml --dp --dp-noise 2.0
    train mnist_alpha0.5 --config configs/mnist.yaml --partition dirichlet --alpha 0.5
    train mnist_alpha0.1 --config configs/mnist.yaml --partition dirichlet --alpha 0.1
    train cifar10_baseline --config configs/cifar10.yaml
  } | parallel
  evaluate results/evaluation_cifar10.md --dataset cifar10 --num-images 25 \
    --out results/evaluation_cifar10
  "$PY" scripts/make_gallery.py --dataset cifar10 --out results/reconstruction_gallery_cifar10.png
}

case "${1:-all}" in
  study) study ;;
  core) core ;;
  extras) extras ;;
  all) study; core; extras ;;
  *) echo "usage: $0 [study|core|extras|all]"; exit 2 ;;
esac
"$PY" scripts/summarize_results.py > /dev/null && echo "summary -> results/summary.md"
