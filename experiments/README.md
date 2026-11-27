# Experiments

Scripts for the numerical experiments of Secs. 2, 4 and 5 of the paper: the iteration budget and rounding, wrong
locks, phase maps, finite shots, the comparison with QAOA and simulated annealing, and the noisy circuit runs.
Each experiment has a report in `reports/` with its purpose, method, command, parameters, raw files, wall time
and results; `reports/INDEX.md` lists them.

All scripts run from the repository root and take command-line arguments (`--help`). They write raw results
incrementally as JSON lines, one line per finished task, so an interrupted run resumes where it stopped. CSV
summaries and plots are made from the saved data; `--plot-only` re-plots without simulating.

## Instances

`common.get_instances(n, count, seed)` generates the random QUBOs once per `(n, seed)` and stores them in
`instances/qubo_n{n}_seed{seed}.npz` (Q matrices, index of the optimum, minimum and maximum energy, spectral gap).
Instance `i` comes from its own stream `default_rng([seed, n, i])` and has the distribution of
`vqpm.random_qubo`: $Q=G+G^T$ with $G_{ij}\sim\mathcal N(0,1)$. Every experiment uses seed 2026 by default, so
all methods see the same instances. Seed 777 gives the separate tuning instances of the QAOA pilot.

## Scripts

| script | experiment | report |
|---|---|---|
| `vqpm_grid.py` | VQPM over n = 4 to 22, precision {3, 6, exact}, stop rule {oracle, max_prob} and iteration budget {30, 100, 300, 10n}: found-optimum rate, P(optimum), energy gap, iterations, locks, wrong locks, stall flag | `vqpm_grid.md` |
| `wrong_locks.py` | wrong-lock statistics from the lock logs of the grid (no new simulation) | `wrong_locks.md` |
| `phase_map_ablation.py` | four phase maps (`paper`, `tight`, `tight_wide`, `shifted`) | `phase_map.md` |
| `finite_shots.py` | marginals from $10^3$ to $10^6$ shots per iteration; fixed threshold against the union-bound Hoeffding threshold | `finite_shots.md` |
| `qaoa_vs_vqpm.py` | QAOA (L-BFGS-B, COBYLA, gradient descent, Adam; random restarts and INTERP; energy evaluations counted) against VQPM and simulated annealing | `qaoa_vs_vqpm.md`, `qaoa_optimizer_pilot.md` |
| `noisy_runs.py` | Qiskit Aer runs (density matrices) with all-qubit measurement and a noise model; a representative trajectory and aggregate plots | `noise_sweep_n8.md`, `depol_torino.md`, `faketorino_n8.md` |
| `../vqpm_qiskit_noisy/vqpm_qiskit.py --gate-counts 8 12 16` | two-qubit gate count and depth of the transpiled controlled-U | `gate_counts.md` |
| `reproduce_published.py` | checks that the default code path reproduces the figures of the original code (mean bit difference 1.16 of the run figure without locking; 0.46 / 0.40 / 0.18 at n = 12 / 14 / 16 in the original scaling figure) | `reproduction.md` |
| `reports.py` | writes the reports in `reports/` from the results | |
| `status.py` | progress of running jobs | |
| `run_lane1.sh`, `run_lane2.sh` | the job sequences used to run everything on one workstation | |

The iteration-budget sweep is part of `vqpm_grid.py` (`--budgets 30 100 300 10n`). One trajectory per
(instance, precision) is simulated with the largest budget, and the outcome under every budget and stop rule is
read off it (`common.derive_result`, tested to equal direct runs).

The noise runs with the locking threshold $p_{diff}=0.01$ of the rest of the paper are in
`theory_checks/noise_rerun.py`, which calls `noisy_runs.run_task`.

## Output layout

- `results/<experiment>/raw/*.jsonl`: raw results (not committed). `reports.py --archive` archives them as dated
  tarballs with checksums in the directory given by the environment variable `VQPM_BACKUP_DIR`.
- `results/<experiment>/*_summary.csv` and `*.png`, `*.pdf`: summaries and plots (committed).
- `logs/`: logs of the runs (not committed).

## Running the long jobs

Use one BLAS thread per worker (set by `common.py`) and leave one or two cores free, for example

```bash
nohup .venv/bin/python experiments/vqpm_grid.py --n $(seq 4 22) --instances 100 --workers 14 \
    > experiments/logs/vqpm_grid.log 2>&1 &
```

The exact command of every run is in its report.
