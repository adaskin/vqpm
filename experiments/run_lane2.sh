#!/usr/bin/env bash
# Lane 2 of the experiment runs (4 workers): noisy Qiskit Aer runs (density matrix).
# The n = 12 noisy runs were dropped (about 13 min per iteration).
set -x
cd "$(dirname "$0")/.."
PY=.venv/bin/python
W=${W:-4}
R=experiments/results/noisy
TORINO=depol:2.7e-4,4.1e-3,2.3e-2   # FakeTorino median sx, CZ, readout errors, all-to-all
# wait for a sweep process started by an earlier version of this script
if [ -n "$WAIT_PID" ]; then while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done; fi
# (c) noise-strength sweep at n = 8 (ideal sampling + three depolarising levels)
$PY experiments/noisy_runs.py --n 8 --instances 5 --sim-seeds 0 1 2 --shots 100000 --workers $W \
    --noise none depol:1e-5,1e-4,1e-3 depol:1e-4,1e-3,1e-2 $TORINO --out $R/raw/noise_sweep_n8.jsonl
# (b) full FakeTorino model (routing + per-qubit calibration), n = 8
$PY experiments/noisy_runs.py --n 8 --instances 5 --sim-seeds 0 1 2 --shots 100000 --workers $W \
    --noise fake:FakeTorino --out $R/raw/faketorino_n8.jsonl
# (a) Torino-calibrated depolarising + readout, n = 8, 10 (5 instances x 3 seeds)
$PY experiments/noisy_runs.py --n 8 10 --instances 5 --sim-seeds 0 1 2 --shots 100000 --workers $W \
    --noise $TORINO --out $R/raw/depol_torino.jsonl
echo LANE2 DONE
