#!/usr/bin/env bash
# Lane 1 of the experiment runs (10 workers). Every step resumes from its JSONL
# and rewrites its summary CSV and plots when it finishes.
set -x
cd "$(dirname "$0")/.."
PY=.venv/bin/python
W=${W:-10}
R=experiments/results
$PY experiments/vqpm_grid.py --n $(seq 4 20) --instances 100 --workers $W --out $R/vqpm_grid/raw/grid.jsonl
while [ ! -f experiments/instances/qubo_n22_seed2026.npz ]; do sleep 60; done
$PY experiments/vqpm_grid.py --n 21 22 --instances 100 --workers $W --out $R/vqpm_grid/raw/grid.jsonl
# expected energies for the records made before they were recorded (deterministic re-run)
$PY experiments/vqpm_grid.py --instances 100 --workers $W --out $R/vqpm_grid/raw/grid.jsonl --backfill-expected-energy
$PY experiments/wrong_locks.py --grid $R/vqpm_grid/raw/grid.jsonl
$PY experiments/phase_map_ablation.py --n 8 10 12 14 16 --instances 100 --workers $W --out $R/phase_map/raw/phase_map.jsonl
$PY experiments/finite_shots.py --n 8 10 12 14 16 --instances 100 --workers $W --out $R/finite_shots/raw/finite_shots.jsonl
$PY experiments/qaoa_vs_vqpm.py --n 4 6 8 10 12 --workers $W --out $R/qaoa_vs_vqpm/raw/qaoa_vs_vqpm.jsonl
$PY experiments/qaoa_vs_vqpm.py --n 4 6 8 10 12 14 16 --workers $W --out $R/qaoa_vs_vqpm/raw/qaoa_vs_vqpm.jsonl
echo LANE1 DONE
