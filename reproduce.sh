#!/usr/bin/env bash
# End-to-end Task 2: every number and figure in results/ and the deck, from data/creditcard.csv.
#   data + common split -> all models x 5 seeds (protocol, cost variants, ranking loss, amplitude damping)
#   -> shot test -> dense stress test -> noise types -> dense protocol sweep -> tables + figures.
# PARALLEL jobs run at once (each uses 2 torch threads). Expect many CPU hours: the dense baseline protocol
# sweep alone is ~4 h per seed on a 12-core laptop. All jobs are independent; figures are built at the end.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
PARALLEL=${PARALLEL:-6}
export PYTHONIOENCODING=utf-8
mkdir -p results/logs
{
  for s in 0 1 2 3 4; do
    # official protocol (ncrc_4q_L4 explicitly with plain parameter shift, as reported in the main tables)
    for m in baseline logreg ncrc_1q_L2 ncrc_1q_L4 ncrc_1q_L6 ncrc_4q_L2 ncrc_4q_L4_cnot ncrc_1q_L2_pruned ncrc_1q_L4_pruned; do
      echo "src/run.py --model $m --seed $s"
    done
    echo "src/run.py --model ncrc_4q_L4 --seed $s --gradient-backend standard"
    # cost variants: cheap schedule, packed gradients (identical results, 15 instead of 81 executions/sample)
    echo "src/run.py --model ncrc_1q_L2 --seed $s --val-fraction 0.2 --epochs 15 --tag ncrc_1q_L2_cheap"
    echo "src/run.py --model ncrc_1q_L2_pruned --seed $s --val-fraction 0.2 --epochs 15 --tag ncrc_1q_L2_pruned_cheap"
    echo "src/run.py --model ncrc_4q_L4 --seed $s --val-fraction 0.2 --epochs 15 --gradient-backend standard --tag ncrc_4q_L4_cheap"
    echo "src/run.py --model ncrc_4q_L4 --seed $s --val-fraction 0.2 --epochs 15 --gradient-backend combined --tag ncrc_4q_L4_cheap_packed"
    echo "src/run.py --model ncrc_4q_L4 --seed $s --tag ncrc_4q_L4_packed"
    # optional ranking loss (pairwise term, weight 0.5)
    echo "src/run.py --model ncrc_4q_L4_pairwise05 --seed $s"
    # amplitude damping instead of depolarizing noise
    for m in baseline ncrc_1q_L2_pruned ncrc_4q_L4 ncrc_4q_L4_cnot; do
      echo "src/run.py --model $m --seed $s --channel amplitude_damping --gradient-backend standard --tag ${m}_ad"
    done
    # 1,024-shot readout (independent NCRC wires sampled separately, entangled NCRC as joint bitstrings)
    for m in baseline ncrc_1q_L2_pruned ncrc_4q_L4 ncrc_4q_L4_pairwise05 ncrc_4q_L4_cnot; do echo "src/shots.py --model $m --seed $s"; done
    # noise types / uneven noise and the dense stress test, fraud-rich 500-row sets
    for m in baseline logreg ncrc_4q_L4_cnot ncrc_4q_L4 ncrc_1q_L2_pruned; do
      echo "src/noise_types.py --model $m --seed $s"
      echo "src/sweep.py --mode stress --model $m --seed $s"
    done
    # dense official-protocol sweep (21 levels, full test set)
    for m in baseline logreg ncrc_4q_L4_cnot ncrc_4q_L4; do echo "src/sweep.py --mode protocol --model $m --seed $s"; done
  done
} > results/logs/all_jobs.txt
xargs -P "$PARALLEL" -L 1 "$PY" < results/logs/all_jobs.txt
cd src
for script in report.py plot_deck.py plot_noise_types.py plot_sweep.py plot_slides.py demo_seed0.py plot_canva.py; do "$PY" "$script"; done
