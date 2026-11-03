#!/usr/bin/env bash
# Regenerate everything in results/ from scratch: train 3 seeds, evaluate, figures.
# Each seed takes about 8 minutes on one CPU core. The three runs execute in parallel.
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=(0 1 2)
for s in "${SEEDS[@]}"; do
  python -m voltvar_rl train --set seed="$s" --out "runs/seed$s" &
done
wait

MODELS=()
for s in "${SEEDS[@]}"; do
  mkdir -p "results/seed$s"
  cp "runs/seed$s/ppo_best.pt" "runs/seed$s/training_log.csv" "runs/seed$s/train_summary.json" "results/seed$s/"
  MODELS+=(--model "results/seed$s/ppo_best.pt")
done

python -m voltvar_rl plot-training --run results/seed0 --run results/seed1 --run results/seed2 \
       --out results/training_curve.png
python -m voltvar_rl evaluate "${MODELS[@]}" --out results
# Single-scenario figures use the seed-0 policy.
python -m voltvar_rl profile      --model results/seed0/ppo_best.pt --day 2001 --hour 12 --out results
python -m voltvar_rl simulate-day --model results/seed0/ppo_best.pt --day 2001 --out results
