# Pilot: QAOA optimizer and learning-rate choice (n = 8, tuning instances)

_Report generated 2026-11-27T07:51:15 by `experiments/reports.py`; status: finished._

## Purpose

Choose learning rates on separate instances; check L-BFGS-B vs COBYLA and adjoint vs finite-difference gradients.

## Method

20 instances (seed 777, not used elsewhere), n = 8, p = 4 and 8, 5 restarts, random and INTERP starts, cap 500. Optimizers: L-BFGS-B (adjoint and finite-difference gradients), COBYLA, gradient descent (lr 0.03 .. 10), Adam (lr 0.01 .. 0.3).

Metrics (per run, energies in the units of the original, unscaled Q): found optimum = the returned bitstring is the exact ground state (found by enumerating all 2^n energies); bit difference = Hamming distance to the ground state; energy gap = E_found - E_min; normalised energy gap = (E_found - E_min)/(E_max - E_min) in [0, 1]; normalised expected-energy gap = (<E> - E_min)/(E_max - E_min) with <E> the expected energy of the final output distribution (VQPM, QAOA). Intervals: Wilson 95% for rates, mean +- 1.96 s/sqrt(N) for means.

## Provenance

- Invocation 1 of `experiments/results/qaoa_pilot/raw/pilot_lr_seed777.jsonl`: started 2026-11-25T17:19:32, git commit `1da710c18b1e1d8f5d4374caa9bcd7482fcf070a`, python 3.12.12, numpy 2.5.5, scipy 1.18.1
  - command: `.venv/bin/python experiments/qaoa_vs_vqpm.py --n 8 --instances 20 --instances-small 20 --seed 777 --p 4 8 --restarts 5 --optimizers L-BFGS-B L-BFGS-B#fd COBYLA gd@0.03 gd@0.1 gd@0.3 gd@1 adam@0.01 adam@0.03 adam@0.1 adam@0.3 --workers 12 --out experiments/results/qaoa_pilot/raw/pilot_lr_seed777.jsonl`
  - parameters: `{"inits": ["random", "interp"], "inst_dir": "experiments/instances", "instances": 20, "instances_small": 20, "maxiter": 500, "n": [8], "optimizers": ["L-BFGS-B", "L-BFGS-B#fd", "COBYLA", "gd@0.03", "gd@0.1", "gd@0.3", "gd@1", "adam@0.01", "adam@0.03", "adam@0.1", "adam@0.3"], "out": "experiments/results/qaoa_pilot/raw/pilot_lr_seed777.jsonl", "p": [4, 8], "plot_only": false, "restarts": 5, "sa_sweeps": [10, 100, 1000], "seed": 777, "skip": [], "small_n": 10, "vqpm_budgets": [30, 100], "workers": 12}`
- Library versions (pinned in requirements.txt): matplotlib 3.11.2, numpy 2.5.5, pytest 9.2.0, qiskit 2.6.2, qiskit-aer 0.18.4, qiskit-ibm-runtime 0.52.1, scipy 1.18.1, python 3.12.12
- Machine: AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 16 logical CPUs, 90 GiB RAM, Linux 6.12.113+deb13-amd64
- End time (last write of the raw data): `pilot_lr_seed777.jsonl` 2026-11-25T17:41:55
- Wall time `pilot_lr_seed777.jsonl` (first start to last write, includes waits between invocations): 0.37 h
- Instances: seed 777, files `experiments/instances/qubo_n<n>_seed777.npz` (instance i is `default_rng([777, n, i])`, Q = A + A^T, A_ij ~ N(0, 1); each file stores Q, optimum, E_min, E_max, gap); sha256 listed below

## Where the data is

Paths are relative to the repository root.

| file (repo-relative) | records / size | sha256 |
|---|---|---|
| `experiments/results/qaoa_pilot/raw/pilot_lr_seed777.jsonl` | 1180 records, 0.7 MiB | `2599e8efbb40d08c9ec457ba33c77386cb97d032f85566926422ae9f9324edb3` |
| `experiments/instances/qubo_n8_seed777.npz` | 8 KiB | `6320be7b8343549b44df6dd80813e3068e186d5bcd039628bf6b3c4ee6d26664` |

Summary tables and plots (committed to git unless noted):

- `experiments/results/qaoa_pilot/pilot_lr_seed777.pdf`
- `experiments/results/qaoa_pilot/pilot_lr_seed777.png`
- `experiments/results/qaoa_pilot/pilot_lr_seed777_metrics_COBYLA.pdf`
- `experiments/results/qaoa_pilot/pilot_lr_seed777_metrics_COBYLA.png`
- `experiments/results/qaoa_pilot/pilot_lr_seed777_metrics_L-BFGS-B#fd.pdf`
- `experiments/results/qaoa_pilot/pilot_lr_seed777_metrics_L-BFGS-B#fd.png`
- `experiments/results/qaoa_pilot/pilot_lr_seed777_metrics_L-BFGS-B.pdf`
- `experiments/results/qaoa_pilot/pilot_lr_seed777_metrics_L-BFGS-B.png`
- `experiments/results/qaoa_pilot/pilot_lr_seed777_metrics_adam@0.01.pdf`
- `experiments/results/qaoa_pilot/pilot_lr_seed777_metrics_adam@0.01.png`
- `experiments/results/qaoa_pilot/pilot_lr_seed777_metrics_adam@0.03.pdf`
- `experiments/results/qaoa_pilot/pilot_lr_seed777_metrics_adam@0.03.png`
- `experiments/results/qaoa_pilot/pilot_lr_seed777_metrics_adam@0.1.pdf`
- `experiments/results/qaoa_pilot/pilot_lr_seed777_metrics_adam@0.1.png`
- `experiments/results/qaoa_pilot/pilot_lr_seed777_metrics_adam@0.3.pdf`
- `experiments/results/qaoa_pilot/pilot_lr_seed777_metrics_adam@0.3.png`
- `experiments/results/qaoa_pilot/pilot_lr_seed777_metrics_all_optimizers_p4.pdf`
- `experiments/results/qaoa_pilot/pilot_lr_seed777_metrics_all_optimizers_p4.png`
- `experiments/results/qaoa_pilot/pilot_lr_seed777_metrics_all_optimizers_p8.pdf`
- `experiments/results/qaoa_pilot/pilot_lr_seed777_metrics_all_optimizers_p8.png`
- `experiments/results/qaoa_pilot/pilot_lr_seed777_metrics_gd@0.03.pdf`
- `experiments/results/qaoa_pilot/pilot_lr_seed777_metrics_gd@0.03.png`
- `experiments/results/qaoa_pilot/pilot_lr_seed777_metrics_gd@0.1.pdf`
- `experiments/results/qaoa_pilot/pilot_lr_seed777_metrics_gd@0.1.png`
- `experiments/results/qaoa_pilot/pilot_lr_seed777_metrics_gd@0.3.pdf`
- `experiments/results/qaoa_pilot/pilot_lr_seed777_metrics_gd@0.3.png`
- `experiments/results/qaoa_pilot/pilot_lr_seed777_metrics_gd@1.pdf`
- `experiments/results/qaoa_pilot/pilot_lr_seed777_metrics_gd@1.png`
- `experiments/results/qaoa_pilot/pilot_lr_seed777_metrics_gd@10.pdf`
- `experiments/results/qaoa_pilot/pilot_lr_seed777_metrics_gd@10.png`
- `experiments/results/qaoa_pilot/pilot_lr_seed777_metrics_gd@3.pdf`
- `experiments/results/qaoa_pilot/pilot_lr_seed777_metrics_gd@3.png`
- `experiments/results/qaoa_pilot/pilot_lr_seed777_runs.csv`
- `experiments/results/qaoa_pilot/pilot_lr_seed777_summary.csv`

Raw JSONL files are not in the repository because of their size. The scripts write them to `experiments/results/<experiment>/raw/`, and the summary tables above are computed from them.

## Results

| method | n | runs | found-opt [CI] | P(opt) | bit diff | norm. energy gap [CI] | norm. expected-E gap | evals |
|---|---|---|---|---|---|---|---|---|
| SA (10 sweeps) | 8 | 20 | 0.95 [0.76, 0.99] | 0.95 | 0.10 | 0.002 [-0.002, 0.006] | - | 80 |
| SA (100 sweeps) | 8 | 20 | 1.00 [0.84, 1.00] | 1.00 | 0.00 | 0.000 [-0.000, 0.000] | - | 800 |
| SA (1000 sweeps) | 8 | 20 | 1.00 [0.84, 1.00] | 1.00 | 0.00 | 0.000 [-0.000, 0.000] | - | 8000 |
| QAOA gd@1 interp p=4 | 8 | 20 | 0.65 [0.43, 0.82] | 0.21 | 0.60 | 0.005 [0.001, 0.009] | - | 60000 |
| QAOA gd@1 interp p=8 | 8 | 20 | 0.80 [0.58, 0.92] | 0.38 | 0.25 | 0.001 [0.000, 0.002] | - | 200000 |
| QAOA gd@1 random p=4 | 8 | 20 | 0.55 [0.34, 0.74] | 0.09 | 0.65 | 0.010 [0.002, 0.019] | - | 22500 |
| QAOA gd@1 random p=8 | 8 | 20 | 0.55 [0.34, 0.74] | 0.13 | 0.65 | 0.010 [0.002, 0.019] | - | 42500 |
| QAOA gd@3 interp p=4 | 8 | 20 | 0.60 [0.39, 0.78] | 0.12 | 0.55 | 0.015 [-0.001, 0.032] | - | 60000 |
| QAOA gd@3 interp p=8 | 8 | 20 | 0.60 [0.39, 0.78] | 0.16 | 0.60 | 0.009 [0.001, 0.017] | - | 200000 |
| QAOA gd@3 random p=4 | 8 | 20 | 0.55 [0.34, 0.74] | 0.08 | 0.65 | 0.008 [0.000, 0.016] | - | 22500 |
| QAOA gd@3 random p=8 | 8 | 20 | 0.50 [0.30, 0.70] | 0.08 | 0.70 | 0.031 [0.008, 0.055] | - | 42500 |
| QAOA gd@10 interp p=4 | 8 | 20 | 0.20 [0.08, 0.42] | 0.04 | 2.20 | 0.198 [0.086, 0.311] | - | 60000 |
| QAOA gd@10 interp p=8 | 8 | 20 | 0.20 [0.08, 0.42] | 0.03 | 1.90 | 0.189 [0.093, 0.284] | - | 200000 |
| QAOA gd@10 random p=4 | 8 | 20 | 0.25 [0.11, 0.47] | 0.03 | 1.80 | 0.184 [0.086, 0.282] | - | 22500 |
| QAOA gd@10 random p=8 | 8 | 20 | 0.30 [0.15, 0.52] | 0.02 | 1.55 | 0.154 [0.069, 0.239] | - | 42500 |
| QAOA COBYLA interp p=4 | 8 | 20 | 0.75 [0.53, 0.89] | 0.32 | 0.35 | 0.002 [0.000, 0.004] | - | 10000 |
| QAOA COBYLA interp p=8 | 8 | 20 | 0.90 [0.70, 0.97] | 0.54 | 0.10 | 0.000 [-0.000, 0.001] | - | 20000 |
| QAOA COBYLA random p=4 | 8 | 20 | 0.55 [0.34, 0.74] | 0.09 | 0.65 | 0.010 [0.002, 0.019] | - | 2500 |
| QAOA COBYLA random p=8 | 8 | 20 | 0.55 [0.34, 0.74] | 0.13 | 0.65 | 0.010 [0.002, 0.019] | - | 2500 |
| QAOA L-BFGS-B interp p=4 | 8 | 20 | 0.80 [0.58, 0.92] | 0.39 | 0.25 | 0.001 [0.000, 0.002] | - | 5320 |
| QAOA L-BFGS-B interp p=8 | 8 | 20 | 0.75 [0.53, 0.89] | 0.60 | 0.25 | 0.002 [0.000, 0.005] | - | 44785 |
| QAOA L-BFGS-B random p=4 | 8 | 20 | 0.80 [0.58, 0.92] | 0.41 | 0.25 | 0.001 [0.000, 0.002] | - | 4882 |
| QAOA L-BFGS-B random p=8 | 8 | 20 | 0.75 [0.53, 0.89] | 0.61 | 0.30 | 0.003 [0.000, 0.005] | - | 44630 |
| QAOA L-BFGS-B#fd interp p=4 | 8 | 20 | 0.80 [0.58, 0.92] | 0.39 | 0.25 | 0.001 [0.000, 0.002] | - | 5326 |
| QAOA L-BFGS-B#fd interp p=8 | 8 | 20 | 0.75 [0.53, 0.89] | 0.60 | 0.25 | 0.002 [0.000, 0.005] | - | 45446 |
| QAOA L-BFGS-B#fd random p=4 | 8 | 20 | 0.80 [0.58, 0.92] | 0.41 | 0.25 | 0.001 [0.000, 0.002] | - | 4888 |
| QAOA L-BFGS-B#fd random p=8 | 8 | 20 | 0.80 [0.58, 0.92] | 0.60 | 0.20 | 0.001 [-0.000, 0.003] | - | 45092 |
| QAOA adam@0.01 interp p=4 | 8 | 20 | 0.70 [0.48, 0.85] | 0.34 | 0.40 | 0.002 [0.000, 0.003] | - | 60000 |
| QAOA adam@0.01 interp p=8 | 8 | 20 | 0.75 [0.53, 0.89] | 0.59 | 0.25 | 0.002 [0.000, 0.005] | - | 197989 |
| QAOA adam@0.01 random p=4 | 8 | 20 | 0.55 [0.34, 0.74] | 0.13 | 0.65 | 0.010 [0.002, 0.019] | - | 22500 |
| QAOA adam@0.01 random p=8 | 8 | 20 | 0.65 [0.43, 0.82] | 0.24 | 0.55 | 0.004 [0.001, 0.007] | - | 42500 |
| QAOA adam@0.03 interp p=4 | 8 | 20 | 0.80 [0.58, 0.92] | 0.39 | 0.25 | 0.001 [0.000, 0.002] | - | 53806 |
| QAOA adam@0.03 interp p=8 | 8 | 20 | 0.75 [0.53, 0.89] | 0.60 | 0.25 | 0.002 [0.000, 0.005] | - | 147747 |
| QAOA adam@0.03 random p=4 | 8 | 20 | 0.70 [0.48, 0.85] | 0.24 | 0.50 | 0.003 [0.000, 0.006] | - | 22500 |
| QAOA adam@0.03 random p=8 | 8 | 20 | 0.75 [0.53, 0.89] | 0.38 | 0.45 | 0.003 [-0.000, 0.005] | - | 42500 |
| QAOA adam@0.1 interp p=4 | 8 | 20 | 0.80 [0.58, 0.92] | 0.39 | 0.25 | 0.001 [0.000, 0.002] | - | 32069 |
| QAOA adam@0.1 interp p=8 | 8 | 20 | 0.75 [0.53, 0.89] | 0.60 | 0.25 | 0.002 [0.000, 0.005] | - | 116398 |
| QAOA adam@0.1 random p=4 | 8 | 20 | 0.70 [0.48, 0.85] | 0.36 | 0.45 | 0.003 [-0.000, 0.005] | - | 22500 |
| QAOA adam@0.1 random p=8 | 8 | 20 | 0.85 [0.64, 0.95] | 0.53 | 0.15 | 0.001 [-0.000, 0.002] | - | 42500 |
| QAOA adam@0.3 interp p=4 | 8 | 20 | 0.80 [0.58, 0.92] | 0.39 | 0.25 | 0.001 [0.000, 0.002] | - | 26826 |
| QAOA adam@0.3 interp p=8 | 8 | 20 | 0.75 [0.53, 0.89] | 0.60 | 0.25 | 0.002 [0.000, 0.005] | - | 125002 |
| QAOA adam@0.3 random p=4 | 8 | 20 | 0.70 [0.48, 0.85] | 0.39 | 0.40 | 0.006 [-0.001, 0.013] | - | 22175 |
| QAOA adam@0.3 random p=8 | 8 | 20 | 0.85 [0.64, 0.95] | 0.60 | 0.15 | 0.001 [-0.000, 0.002] | - | 42500 |
| QAOA gd@0.03 interp p=4 | 8 | 20 | 0.50 [0.30, 0.70] | 0.05 | 0.70 | 0.012 [0.003, 0.021] | - | 60000 |
| QAOA gd@0.03 interp p=8 | 8 | 20 | 0.60 [0.39, 0.78] | 0.13 | 0.60 | 0.008 [0.000, 0.016] | - | 200000 |
| QAOA gd@0.03 random p=4 | 8 | 20 | 0.55 [0.34, 0.74] | 0.02 | 0.85 | 0.028 [0.002, 0.054] | - | 22500 |
| QAOA gd@0.03 random p=8 | 8 | 20 | 0.60 [0.39, 0.78] | 0.04 | 0.60 | 0.012 [0.003, 0.021] | - | 42500 |
| QAOA gd@0.1 interp p=4 | 8 | 20 | 0.55 [0.34, 0.74] | 0.07 | 0.65 | 0.010 [0.002, 0.019] | - | 60000 |
| QAOA gd@0.1 interp p=8 | 8 | 20 | 0.65 [0.43, 0.82] | 0.19 | 0.55 | 0.004 [0.001, 0.007] | - | 200000 |
| QAOA gd@0.1 random p=4 | 8 | 20 | 0.45 [0.26, 0.66] | 0.03 | 0.75 | 0.017 [0.005, 0.028] | - | 22500 |
| QAOA gd@0.1 random p=8 | 8 | 20 | 0.55 [0.34, 0.74] | 0.06 | 0.65 | 0.016 [0.005, 0.028] | - | 42500 |
| QAOA gd@0.3 interp p=4 | 8 | 20 | 0.55 [0.34, 0.74] | 0.13 | 0.65 | 0.010 [0.002, 0.019] | - | 60000 |
| QAOA gd@0.3 interp p=8 | 8 | 20 | 0.65 [0.43, 0.82] | 0.28 | 0.55 | 0.003 [0.000, 0.006] | - | 200000 |
| QAOA gd@0.3 random p=4 | 8 | 20 | 0.50 [0.30, 0.70] | 0.05 | 0.70 | 0.012 [0.003, 0.021] | - | 22500 |
| QAOA gd@0.3 random p=8 | 8 | 20 | 0.50 [0.30, 0.70] | 0.09 | 0.70 | 0.012 [0.003, 0.021] | - | 42500 |
| VQPM (max_prob, T=30) | 8 | 20 | 0.65 [0.43, 0.82] | 0.44 | 0.45 | 0.006 [0.001, 0.011] | - | 8 |
| VQPM (max_prob, T=100) | 8 | 20 | 0.65 [0.43, 0.82] | 0.44 | 0.45 | 0.006 [0.001, 0.011] | - | 11 |
| VQPM (oracle, T=30) | 8 | 20 | 0.70 [0.48, 0.85] | 0.44 | 0.35 | 0.003 [0.000, 0.005] | - | 8 |
| VQPM (oracle, T=100) | 8 | 20 | 0.70 [0.48, 0.85] | 0.44 | 0.35 | 0.003 [0.000, 0.005] | - | 16 |

## Findings and caveats

Written 2026-11-25 (finished pilot; 20 tuning instances, seed 777, n = 8, never used in the main runs).

- L-BFGS-B with exact adjoint gradients gives the same results as with finite-difference gradients
  (p = 8, random starts: found-optimum 0.75 vs 0.80, P(opt) 0.61 vs 0.60) at about 5x less wall time,
  so the main runs use adjoint gradients and report finite-difference-equivalent evaluation counts.
- L-BFGS-B beats COBYLA with random starts (P(opt) at p = 8: 0.61 vs 0.13; found-optimum 0.75 vs 0.55),
  supporting the paper's statement. With INTERP, COBYLA catches up in found-optimum rate (0.90) but
  not in P(opt) (0.54).
- Adam: best learning rate 0.3 (tried 0.01-0.3): random starts p = 8 found-optimum 0.85, P(opt) 0.60.
- Plain gradient descent (what SGD reduces to on a statevector): best learning rate 1 (tried 0.03-10),
  but clearly worse (random p = 8: 0.55, P(opt) 0.13); lr 10 diverges.
- VQPM on the same instances: found-optimum 0.70 (oracle) / 0.65 (max_prob), P(opt) 0.44, using at
  most 30 iterations. Simulated annealing: 0.95 (10 sweeps), 1.00 (100 sweeps).
- Evaluation cost per instance (5 restarts, p = 8): ~4.5e4 finite-difference-equivalent evaluations
  for L-BFGS-B, 2.5e3 for COBYLA (random), ~4e4 (random) to 2e5 (INTERP chain) for gd / Adam.

## Regenerate tables and plots from the saved data (no re-run)

```bash
.venv/bin/python experiments/qaoa_vs_vqpm.py --plot-only --seed 777 --out experiments/results/qaoa_pilot/raw/pilot_lr_seed777.jsonl
.venv/bin/python experiments/reports.py
```
