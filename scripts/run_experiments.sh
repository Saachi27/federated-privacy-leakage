#!/usr/bin/env bash
# Reproduce every number and figure in the report.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
PY=${PY:-.venv/bin/python}
JOBS=${JOBS:-4}  # one per performance core; more runs slower overall on an M1
SEEDS="0 1 2"
# The final study: four settings (src/settings.py), 32 target images, 3 seeds.
STUDY=results/study
STUDY_SETTINGS=${STUDY_SETTINGS:-"mnist-fresh mnist-trained cifar10-fresh mnist-fedavg"}
STUDY_SEEDS="0 1 2"
STUDY_SIGMAS="0 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1"
# The cosine TV weight for the supporting tables (core, extras)
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

tuned() {  # tuned <setting> <key>: a value chosen by scripts/tune_attack.py
  "$PY" -c "import json; print(json.load(open('$STUDY/$1/tuning.json'))['$2'])"
}

study() {
  # 0.
  for s in $STUDY_SEEDS; do
    out=results/models/mnist_lenet_r20_s$s.pt
    [ -f "$out" ] && { echo "skip $out"; continue; }
    step "${out%.pt}.log" "$PY" scripts/train_attack_target.py --seed "$s" --out "$out"
  done
  # 1.
  for st in $STUDY_SETTINGS; do
    mkdir -p "$STUDY/$st"
    [ -f "$STUDY/$st/tuning.json" ] && { echo "skip $STUDY/$st/tuning.json"; continue; }
    step "$STUDY/$st/tuning.log" "$PY" scripts/tune_attack.py --setting "$st" --workers "$JOBS"
  done
  # 2.
  for d in mnist cifar10; do
    [ -f "$STUDY/baselines_$d.json" ] && { echo "skip $STUDY/baselines_$d.json"; continue; }
    step "$STUDY/baselines_$d.log" "$PY" scripts/baselines.py --dataset "$d" --first-image 50 \
      --num-images 32 --ks 1 2 4 8 16 --out "$STUDY/baselines_$d"
  done
  # 3.
  for st in $STUDY_SETTINGS; do
    sigmas=$STUDY_SIGMAS; [ "$st" = mnist-fedavg ] && sigmas=0
    step "$STUDY/$st/noise.out" "$PY" scripts/attack_noise_sum.py --setting "$st" --ks 1 2 4 8 \
      --sigmas $sigmas --first-image 50 --num-images 32 --seeds $STUDY_SEEDS \
      --clip "$(tuned "$st" clip)" --cosine-tv "$(tuned "$st" cosine_tv)" --workers "$JOBS" \
      --out "$STUDY/$st/noise"
  done
  # 4.
  for st in $STUDY_SETTINGS; do
    step "$STUDY/$st/sum16.out" "$PY" scripts/attack_sum.py --setting "$st" --ks 16 --first-image 50 \
      --num-images 32 --seeds $STUDY_SEEDS --cosine-tv "$(tuned "$st" cosine_tv)" --workers "$JOBS" \
      --out "$STUDY/$st/sum16"
  done
  step "$STUDY/mnist-fresh/sum_long.out" "$PY" scripts/attack_sum.py --setting mnist-fresh --ks 8 16 \
    --iters 1200 --cosine-iters 12000 --first-image 50 --num-images 32 --seeds $STUDY_SEEDS \
    --cosine-tv "$(tuned mnist-fresh cosine_tv)" --workers "$JOBS" --out "$STUDY/mnist-fresh/sum_long"
  # 5.
  JOBS=$JOBS SEEDS="$SEEDS" PY=$PY bash scripts/check_dp.sh
  # 6.
  for st in $STUDY_SETTINGS; do
    step "$STUDY/$st/rq4.log" "$PY" scripts/analyze_rq4.py --setting "$st"
    step "$STUDY/$st/plot.log" "$PY" scripts/plot_final.py --setting "$st"
  done
  step "$STUDY/compare.log" "$PY" scripts/compare_settings.py
  step results/plot_final.log "$PY" scripts/plot_final.py
  cp "$STUDY"/mnist-fresh/fig_rq4_*.png results/  # the report's copies are the mnist-fresh ones
  step results/plot_dp.log "$PY" scripts/plot_dp.py
  # Fig.
  step results/examples.log "$PY" scripts/make_examples_final.py --workers "$JOBS" \
    --clip "$(tuned mnist-fresh clip)" --no-title --out results/fig_examples_final_v2.png
}

multiclient() {
  # Client-split study.
  local out=results/multiclient
  mkdir -p "$out"
  step "$out/lr_grid.log" "$PY" scripts/attack_multiclient.py --lr-grid --out "$out"
  step "$out/grid_$(date +%Y%m%d_%H%M%S).log" "$PY" scripts/attack_multiclient.py --cut "${MC_CUT:-0}" \
    --workers "$JOBS" --out "$out"
  "$PY" scripts/attack_multiclient.py --summary --out "$out" > "$out/summary.md" && echo "summary -> $out/summary.md"
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
  multiclient) multiclient; exit 0 ;;
  *) echo "usage: $0 [study|core|extras|all|multiclient]"; exit 2 ;;
esac
"$PY" scripts/summarize_results.py > /dev/null && echo "summary -> results/summary.md"
