# Exp. 1: VQPM baseline grid (precision x iteration budget x stop rule)

_Report generated 2026-11-27T07:51:12 by `experiments/reports.py`; status: finished._

## Purpose

Iteration budget vs rounding precision (the decline of the success rate with n under the 30-iteration cap); baseline numbers for the wrong-lock statistics and the oracle-free stop rule.

## Method

For every n, instance (100 per n, seed 2026) and rounding precision of the marginals (3, 6 decimals, or exact = no rounding), VQPM (exact marginals, pdiff = 0.01, paper phase map) is run once with stop_rule='none' for the largest budget max(300, 10n), recording the trajectory (per iteration: target probability, largest basis-state probability and its index, number of locked qubits, whether the prepared state changed, and, for runs made after commit ed0b4c4 or backfilled, the expected energy) and the lock log. Outcomes for the stop rules 'oracle' (stop when P(target) >= 0.5, abort when it is 0) and 'max_prob' (stop when the largest probability >= 0.5 or all qubits are locked) and budgets 30, 100, 300, 10n are read off that trajectory (common.derive_result; tested to equal direct runs). Stall flag: the budget ran out and nothing changed in the last >= 10 iterations. Shots-equivalent (for the planned figure): resolving a marginal to d decimals needs about 10^(2d) shots per iteration (standard error 1/sqrt(M) ~ 10^-d), so shots_equiv_total = iterations x 10^(2d); undefined for exact marginals.

Metrics (per run, energies in the units of the original, unscaled Q): found optimum = the returned bitstring is the exact ground state (found by enumerating all 2^n energies); bit difference = Hamming distance to the ground state; energy gap = E_found - E_min; normalised energy gap = (E_found - E_min)/(E_max - E_min) in [0, 1]; normalised expected-energy gap = (<E> - E_min)/(E_max - E_min) with <E> the expected energy of the final output distribution (VQPM, QAOA). Intervals: Wilson 95% for rates, mean +- 1.96 s/sqrt(N) for means.

## Provenance

- Invocation 1 of `experiments/results/vqpm_grid/raw/grid.jsonl`: started 2026-11-25T17:45:14, git commit `fbc2114ddcbb368ea55e458fa0dd6cb0485c37d7`, python 3.12.12, numpy 2.5.5, scipy 1.18.1
  - command: `.venv/bin/python experiments/vqpm_grid.py --n 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 --instances 100 --workers 10 --out experiments/results/vqpm_grid/raw/grid.jsonl`
  - parameters: `{"budgets": ["30", "100", "300", "10n"], "inst_dir": "experiments/instances", "instances": 100, "n": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], "out": "experiments/results/vqpm_grid/raw/grid.jsonl", "pdiff": 0.01, "phase_maps": ["paper"], "plot_only": false, "precision": ["3", "6", "exact"], "seed": 2026, "summary_dir": null, "workers": 10}`
- Invocation 1 of `experiments/results/vqpm_grid/raw/grid_expected_energy.jsonl`: started 2026-11-25T19:53:05, git commit `61a1e1fe0662bf8c454b52a2427ba08b6a976cf9` (dirty tree), python 3.12.12, numpy 2.5.5, scipy 1.18.1
  - command: `.venv/bin/python experiments/vqpm_grid.py --instances 100 --workers 10 --out experiments/results/vqpm_grid/raw/grid.jsonl --backfill-expected-energy`
  - parameters: `{"backfill_expected_energy": true, "budgets": ["30", "100", "300", "10n"], "inst_dir": "experiments/instances", "instances": 100, "n": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], "out": "experiments/results/vqpm_grid/raw/grid.jsonl", "pdiff": 0.01, "phase_maps": ["paper"], "plot_only": true, "precision": ["3", "6", "exact"], "seed": 2026, "summary_dir": null, "workers": 10}`
- The n = 21, 22 records were added by a second invocation of the same command with `--n 21 22` (it resumes the same file); expected energies of records made before commit ed0b4c4 come from `--backfill-expected-energy` (sidecar file).
- Library versions (pinned in requirements.txt): matplotlib 3.11.2, numpy 2.5.5, pytest 9.2.0, qiskit 2.6.2, qiskit-aer 0.18.4, qiskit-ibm-runtime 0.52.1, scipy 1.18.1, python 3.12.12
- Machine: AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 16 logical CPUs, 90 GiB RAM, Linux 6.12.113+deb13-amd64
- End time (last write of the raw data): `grid.jsonl` 2026-11-25T19:52:56, `grid_expected_energy.jsonl` 2026-11-25T20:17:21
- Wall time `grid.jsonl` (first start to last write, includes waits between invocations): 2.13 h
- Wall time `grid_expected_energy.jsonl` (first start to last write, includes waits between invocations): 0.40 h
- Instances: seed 2026, files `experiments/instances/qubo_n<n>_seed2026.npz` (instance i is `default_rng([2026, n, i])`, Q = A + A^T, A_ij ~ N(0, 1); each file stores Q, optimum, E_min, E_max, gap); sha256 listed below

## Where the data is

Paths are relative to the repository root.

| file (repo-relative) | records / size | sha256 |
|---|---|---|
| `experiments/results/vqpm_grid/raw/grid.jsonl` | 5700 records, 42.3 MiB | `04f126275ecae6a79ac87004419781b94cecade848291b2c0204faf2f2e77200` |
| `experiments/results/vqpm_grid/raw/grid_expected_energy.jsonl` | 5100 records, 1.7 MiB | `276242ec217597484c90c8ec7ed355383b58cf27d9f1ee323e68ebddbe09d1ba` |
| `experiments/instances/qubo_n10_seed2026.npz` | 53 KiB | `8d49d8ae73eca6b6354b380c0db6b34cbe9468d47a7361c1cd356f1a3667b22b` |
| `experiments/instances/qubo_n11_seed2026.npz` | 63 KiB | `a153cac75d69305feeae2f7205afeb9c4826a4b1103e48c8204390c372bbee24` |
| `experiments/instances/qubo_n12_seed2026.npz` | 75 KiB | `aa4cb387c8c0e96855df94242ea38d17d2031f554930229091385368ed74d2dd` |
| `experiments/instances/qubo_n13_seed2026.npz` | 87 KiB | `b29a9187f8e98beeb49b6e47dc34b559b1f185c3c25f73dcfc7bed937c25e86a` |
| `experiments/instances/qubo_n14_seed2026.npz` | 100 KiB | `f97bceaccb9c1b4f0a38d27b6fb5007ff1fe0676baab4b17174df2023d6a2613` |
| `experiments/instances/qubo_n15_seed2026.npz` | 114 KiB | `bdbb649eabe36ec8d7b905db93bbd88b3d1303cf358945676ba48d59cb3e4c91` |
| `experiments/instances/qubo_n16_seed2026.npz` | 129 KiB | `0bbd4f0f221e12268e83b01c1be2e55de062ab9f0261e7bcaf866117601f3f0a` |
| `experiments/instances/qubo_n17_seed2026.npz` | 145 KiB | `562e98c2e42da3d4eeb7e654d1ce06bee2b68cb599059d84ab080efa9a29c114` |
| `experiments/instances/qubo_n18_seed2026.npz` | 163 KiB | `f92df484597bb695d55a37d2be17344acb6c038cd4290b54213a4f9472c6f64e` |
| `experiments/instances/qubo_n19_seed2026.npz` | 181 KiB | `0e60aceca6db42384e7b639df00714c64d19f6b2465a3db91a14137bd1f74af8` |
| `experiments/instances/qubo_n20_seed2026.npz` | 200 KiB | `58dfa22a9d4f866fbb5b4c322c33448f58fd6bd3fe974a9813880d9318196fe7` |
| `experiments/instances/qubo_n21_seed2026.npz` | 220 KiB | `af9fb3ffb8caf7b25619650ed17a3d968ebaac0e6a0a8a91908be467e4e107f7` |
| `experiments/instances/qubo_n22_seed2026.npz` | 241 KiB | `0fcb0115307cddf05aa976940e0e84a69c2148c527537d830c5230f0dc4f4cf9` |
| `experiments/instances/qubo_n4_seed2026.npz` | 12 KiB | `6115029ba7f0b64c856982af17a7ddc9ac9ae858b7ef11e29ef07884bc302c43` |
| `experiments/instances/qubo_n5_seed2026.npz` | 17 KiB | `abceeff504295892420d678f7b26e647dfcbab872b40f13c321c2446d6a13448` |
| `experiments/instances/qubo_n6_seed2026.npz` | 22 KiB | `1827e0082e2deb8b6c4e32178a12a086606c138822eb42a12409899cfe921a38` |
| `experiments/instances/qubo_n7_seed2026.npz` | 28 KiB | `e5efd13b01073e93e77a61a22fc12a21ffb3ed8b2879c103b36e92d884b6e957` |
| `experiments/instances/qubo_n8_seed2026.npz` | 36 KiB | `16c3915c4af3d527804890655d38c383c03e6eedbc8b6b3e5614607ec9f90222` |
| `experiments/instances/qubo_n9_seed2026.npz` | 44 KiB | `07a93729e9e993161cbcbd868ca6981976fa7f80997c31acd5c5a66eb3c5bb1e` |

Summary tables and plots (committed to git unless noted):

- `experiments/results/vqpm_grid/vqpm_grid_failures.pdf`
- `experiments/results/vqpm_grid/vqpm_grid_failures.png`
- `experiments/results/vqpm_grid/vqpm_grid_locks.csv` (not committed: large; regenerate)
- `experiments/results/vqpm_grid/vqpm_grid_max_prob.pdf`
- `experiments/results/vqpm_grid/vqpm_grid_max_prob.png`
- `experiments/results/vqpm_grid/vqpm_grid_metrics_max_prob_prec3.pdf`
- `experiments/results/vqpm_grid/vqpm_grid_metrics_max_prob_prec3.png`
- `experiments/results/vqpm_grid/vqpm_grid_metrics_max_prob_prec6.pdf`
- `experiments/results/vqpm_grid/vqpm_grid_metrics_max_prob_prec6.png`
- `experiments/results/vqpm_grid/vqpm_grid_metrics_max_prob_precexact.pdf`
- `experiments/results/vqpm_grid/vqpm_grid_metrics_max_prob_precexact.png`
- `experiments/results/vqpm_grid/vqpm_grid_metrics_oracle_prec3.pdf`
- `experiments/results/vqpm_grid/vqpm_grid_metrics_oracle_prec3.png`
- `experiments/results/vqpm_grid/vqpm_grid_metrics_oracle_prec6.pdf`
- `experiments/results/vqpm_grid/vqpm_grid_metrics_oracle_prec6.png`
- `experiments/results/vqpm_grid/vqpm_grid_metrics_oracle_precexact.pdf`
- `experiments/results/vqpm_grid/vqpm_grid_metrics_oracle_precexact.png`
- `experiments/results/vqpm_grid/vqpm_grid_oracle.pdf`
- `experiments/results/vqpm_grid/vqpm_grid_oracle.png`
- `experiments/results/vqpm_grid/vqpm_grid_runs.csv` (not committed: large; regenerate)
- `experiments/results/vqpm_grid/vqpm_grid_summary.csv`
- `experiments/results/vqpm_grid/vqpm_grid_timing.csv`
- `experiments/results/vqpm_grid/vqpm_grid_wrong_locks.pdf`
- `experiments/results/vqpm_grid/vqpm_grid_wrong_locks.png`

Raw JSONL files are not in the repository because of their size. The scripts write them to `experiments/results/<experiment>/raw/`, and the summary tables above are computed from them.

## Results

**Found-optimum rate, stop rule `oracle`** (columns: precision pP, budget T; 100 instances per n)

| n | p3 T30 | p3 T100 | p3 T300 | p3 T10n | p6 T30 | p6 T100 | p6 T300 | p6 T10n | pexact T30 | pexact T100 | pexact T300 | pexact T10n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.80 | 0.80 | 0.80 | 0.80 | 0.78 | 0.79 | 0.79 | 0.78 | 0.78 | 0.79 | 0.79 | 0.78 |
| 5 | 0.74 | 0.74 | 0.74 | 0.74 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| 6 | 0.74 | 0.74 | 0.74 | 0.74 | 0.75 | 0.77 | 0.77 | 0.76 | 0.75 | 0.77 | 0.77 | 0.76 |
| 7 | 0.77 | 0.77 | 0.77 | 0.77 | 0.77 | 0.77 | 0.77 | 0.77 | 0.77 | 0.77 | 0.77 | 0.77 |
| 8 | 0.69 | 0.69 | 0.69 | 0.69 | 0.72 | 0.73 | 0.74 | 0.73 | 0.72 | 0.73 | 0.74 | 0.73 |
| 9 | 0.60 | 0.60 | 0.60 | 0.60 | 0.72 | 0.72 | 0.72 | 0.72 | 0.72 | 0.72 | 0.72 | 0.72 |
| 10 | 0.69 | 0.69 | 0.69 | 0.69 | 0.75 | 0.77 | 0.78 | 0.77 | 0.75 | 0.77 | 0.78 | 0.77 |
| 11 | 0.57 | 0.57 | 0.57 | 0.57 | 0.65 | 0.68 | 0.68 | 0.68 | 0.65 | 0.68 | 0.68 | 0.68 |
| 12 | 0.59 | 0.59 | 0.59 | 0.59 | 0.62 | 0.65 | 0.66 | 0.65 | 0.62 | 0.65 | 0.66 | 0.65 |
| 13 | 0.45 | 0.45 | 0.45 | 0.45 | 0.55 | 0.64 | 0.64 | 0.64 | 0.55 | 0.64 | 0.64 | 0.64 |
| 14 | 0.48 | 0.48 | 0.48 | 0.48 | 0.56 | 0.65 | 0.66 | 0.66 | 0.56 | 0.65 | 0.66 | 0.66 |
| 15 | 0.35 | 0.35 | 0.35 | 0.35 | 0.44 | 0.52 | 0.53 | 0.53 | 0.44 | 0.52 | 0.53 | 0.53 |
| 16 | 0.37 | 0.38 | 0.38 | 0.38 | 0.52 | 0.57 | 0.57 | 0.57 | 0.52 | 0.57 | 0.57 | 0.57 |
| 17 | 0.39 | 0.40 | 0.40 | 0.40 | 0.48 | 0.62 | 0.66 | 0.66 | 0.48 | 0.62 | 0.66 | 0.66 |
| 18 | 0.38 | 0.38 | 0.38 | 0.38 | 0.46 | 0.56 | 0.58 | 0.58 | 0.46 | 0.56 | 0.58 | 0.58 |
| 19 | 0.30 | 0.30 | 0.30 | 0.30 | 0.47 | 0.56 | 0.57 | 0.57 | 0.47 | 0.56 | 0.57 | 0.57 |
| 20 | 0.28 | 0.28 | 0.28 | 0.28 | 0.41 | 0.56 | 0.56 | 0.56 | 0.41 | 0.56 | 0.56 | 0.56 |
| 21 | 0.20 | 0.20 | 0.20 | 0.20 | 0.35 | 0.50 | 0.51 | 0.51 | 0.35 | 0.50 | 0.51 | 0.51 |
| 22 | 0.33 | 0.33 | 0.33 | 0.33 | 0.42 | 0.55 | 0.57 | 0.56 | 0.42 | 0.55 | 0.57 | 0.56 |

**Found-optimum rate, stop rule `max_prob`** (columns: precision pP, budget T; 100 instances per n)

| n | p3 T30 | p3 T100 | p3 T300 | p3 T10n | p6 T30 | p6 T100 | p6 T300 | p6 T10n | pexact T30 | pexact T100 | pexact T300 | pexact T10n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.77 | 0.77 | 0.77 | 0.77 | 0.73 | 0.73 | 0.73 | 0.73 | 0.73 | 0.73 | 0.73 | 0.73 |
| 5 | 0.73 | 0.73 | 0.73 | 0.73 | 0.72 | 0.72 | 0.72 | 0.72 | 0.72 | 0.72 | 0.72 | 0.72 |
| 6 | 0.61 | 0.61 | 0.61 | 0.61 | 0.59 | 0.59 | 0.59 | 0.59 | 0.59 | 0.59 | 0.59 | 0.59 |
| 7 | 0.66 | 0.66 | 0.66 | 0.66 | 0.62 | 0.62 | 0.62 | 0.62 | 0.62 | 0.62 | 0.62 | 0.62 |
| 8 | 0.59 | 0.59 | 0.59 | 0.59 | 0.59 | 0.59 | 0.59 | 0.59 | 0.59 | 0.59 | 0.59 | 0.59 |
| 9 | 0.51 | 0.51 | 0.51 | 0.51 | 0.59 | 0.59 | 0.59 | 0.59 | 0.59 | 0.59 | 0.59 | 0.59 |
| 10 | 0.64 | 0.64 | 0.64 | 0.64 | 0.65 | 0.65 | 0.65 | 0.65 | 0.65 | 0.65 | 0.65 | 0.65 |
| 11 | 0.50 | 0.50 | 0.50 | 0.50 | 0.48 | 0.49 | 0.49 | 0.49 | 0.48 | 0.49 | 0.49 | 0.49 |
| 12 | 0.50 | 0.50 | 0.50 | 0.50 | 0.53 | 0.54 | 0.54 | 0.54 | 0.53 | 0.54 | 0.54 | 0.54 |
| 13 | 0.34 | 0.34 | 0.34 | 0.34 | 0.41 | 0.42 | 0.42 | 0.42 | 0.41 | 0.42 | 0.42 | 0.42 |
| 14 | 0.46 | 0.46 | 0.46 | 0.46 | 0.45 | 0.48 | 0.48 | 0.48 | 0.45 | 0.48 | 0.48 | 0.48 |
| 15 | 0.35 | 0.35 | 0.35 | 0.35 | 0.39 | 0.39 | 0.39 | 0.39 | 0.39 | 0.39 | 0.39 | 0.39 |
| 16 | 0.34 | 0.35 | 0.35 | 0.35 | 0.43 | 0.44 | 0.44 | 0.44 | 0.43 | 0.44 | 0.44 | 0.44 |
| 17 | 0.38 | 0.38 | 0.38 | 0.38 | 0.40 | 0.43 | 0.43 | 0.43 | 0.40 | 0.43 | 0.43 | 0.43 |
| 18 | 0.37 | 0.37 | 0.37 | 0.37 | 0.40 | 0.39 | 0.39 | 0.39 | 0.40 | 0.39 | 0.39 | 0.39 |
| 19 | 0.30 | 0.30 | 0.30 | 0.30 | 0.44 | 0.46 | 0.46 | 0.46 | 0.44 | 0.46 | 0.46 | 0.46 |
| 20 | 0.27 | 0.27 | 0.27 | 0.27 | 0.37 | 0.42 | 0.42 | 0.42 | 0.37 | 0.42 | 0.42 | 0.42 |
| 21 | 0.20 | 0.20 | 0.20 | 0.20 | 0.32 | 0.36 | 0.36 | 0.36 | 0.32 | 0.36 | 0.36 | 0.36 |
| 22 | 0.33 | 0.33 | 0.33 | 0.33 | 0.40 | 0.48 | 0.48 | 0.48 | 0.40 | 0.48 | 0.48 | 0.48 |

**Details at T = 300, stop rule `oracle`**

| n | prec | found-opt [95% CI] | bit diff [CI] | norm. energy gap [CI] | norm. expected-E gap [CI] | iters used | hit budget | stalled | runs w/ wrong lock |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 3 | 0.80 [0.71, 0.87] | 0.22 [0.13, 0.31] | 0.013 [0.004, 0.022] | 0.060 [0.048, 0.072] | 14.2 | 0.04 | 0.04 | 0.16 |
| 4 | 6 | 0.79 [0.70, 0.86] | 0.23 [0.14, 0.32] | 0.019 [0.008, 0.030] | 0.061 [0.048, 0.074] | 3.3 | 0.00 | 0.00 | 0.21 |
| 4 | exact | 0.79 [0.70, 0.86] | 0.23 [0.14, 0.32] | 0.019 [0.008, 0.030] | 0.061 [0.048, 0.074] | 3.3 | 0.00 | 0.00 | 0.21 |
| 5 | 3 | 0.74 [0.65, 0.82] | 0.34 [0.21, 0.47] | 0.020 [0.010, 0.030] | 0.058 [0.046, 0.070] | 6.1 | 0.01 | 0.01 | 0.25 |
| 5 | 6 | 0.75 [0.66, 0.82] | 0.37 [0.22, 0.52] | 0.019 [0.009, 0.029] | 0.060 [0.048, 0.072] | 3.1 | 0.00 | 0.00 | 0.25 |
| 5 | exact | 0.75 [0.66, 0.82] | 0.37 [0.22, 0.52] | 0.019 [0.009, 0.029] | 0.060 [0.048, 0.072] | 3.1 | 0.00 | 0.00 | 0.25 |
| 6 | 3 | 0.74 [0.65, 0.82] | 0.33 [0.21, 0.45] | 0.015 [0.008, 0.022] | 0.052 [0.039, 0.065] | 16.3 | 0.04 | 0.04 | 0.23 |
| 6 | 6 | 0.77 [0.68, 0.84] | 0.29 [0.17, 0.41] | 0.014 [0.008, 0.020] | 0.052 [0.040, 0.063] | 5.8 | 0.00 | 0.00 | 0.23 |
| 6 | exact | 0.77 [0.68, 0.84] | 0.29 [0.17, 0.41] | 0.014 [0.008, 0.020] | 0.052 [0.040, 0.063] | 5.8 | 0.00 | 0.00 | 0.23 |
| 7 | 3 | 0.77 [0.68, 0.84] | 0.31 [0.19, 0.43] | 0.012 [0.006, 0.018] | 0.048 [0.035, 0.060] | 19.8 | 0.05 | 0.05 | 0.18 |
| 7 | 6 | 0.77 [0.68, 0.84] | 0.33 [0.20, 0.46] | 0.014 [0.008, 0.021] | 0.054 [0.042, 0.067] | 5.2 | 0.00 | 0.00 | 0.23 |
| 7 | exact | 0.77 [0.68, 0.84] | 0.33 [0.20, 0.46] | 0.014 [0.008, 0.021] | 0.054 [0.042, 0.067] | 5.2 | 0.00 | 0.00 | 0.23 |
| 8 | 3 | 0.69 [0.59, 0.77] | 0.49 [0.32, 0.66] | 0.016 [0.009, 0.023] | 0.048 [0.035, 0.061] | 35.1 | 0.10 | 0.10 | 0.22 |
| 8 | 6 | 0.74 [0.65, 0.82] | 0.44 [0.27, 0.61] | 0.017 [0.010, 0.024] | 0.053 [0.038, 0.067] | 7.9 | 0.00 | 0.00 | 0.26 |
| 8 | exact | 0.74 [0.65, 0.82] | 0.44 [0.27, 0.61] | 0.017 [0.010, 0.024] | 0.053 [0.038, 0.067] | 7.8 | 0.00 | 0.00 | 0.26 |
| 9 | 3 | 0.60 [0.50, 0.69] | 0.66 [0.47, 0.85] | 0.017 [0.009, 0.024] | 0.045 [0.033, 0.057] | 56.9 | 0.17 | 0.17 | 0.26 |
| 9 | 6 | 0.72 [0.63, 0.80] | 0.49 [0.31, 0.67] | 0.011 [0.006, 0.016] | 0.044 [0.033, 0.055] | 11.4 | 0.01 | 0.00 | 0.27 |
| 9 | exact | 0.72 [0.63, 0.80] | 0.49 [0.31, 0.67] | 0.011 [0.006, 0.016] | 0.044 [0.033, 0.055] | 11.4 | 0.01 | 0.00 | 0.27 |
| 10 | 3 | 0.69 [0.59, 0.77] | 0.58 [0.37, 0.79] | 0.014 [0.008, 0.021] | 0.046 [0.033, 0.058] | 45.5 | 0.13 | 0.13 | 0.21 |
| 10 | 6 | 0.78 [0.69, 0.85] | 0.42 [0.25, 0.59] | 0.012 [0.006, 0.018] | 0.044 [0.031, 0.057] | 10.2 | 0.00 | 0.00 | 0.22 |
| 10 | exact | 0.78 [0.69, 0.85] | 0.42 [0.25, 0.59] | 0.012 [0.006, 0.018] | 0.043 [0.031, 0.056] | 10.2 | 0.00 | 0.00 | 0.22 |
| 11 | 3 | 0.57 [0.47, 0.66] | 0.83 [0.58, 1.08] | 0.017 [0.012, 0.023] | 0.063 [0.046, 0.080] | 48.7 | 0.14 | 0.14 | 0.33 |
| 11 | 6 | 0.68 [0.58, 0.76] | 0.68 [0.44, 0.92] | 0.016 [0.010, 0.021] | 0.059 [0.043, 0.074] | 10.4 | 0.00 | 0.00 | 0.32 |
| 11 | exact | 0.68 [0.58, 0.76] | 0.68 [0.44, 0.92] | 0.016 [0.010, 0.021] | 0.059 [0.043, 0.074] | 10.3 | 0.00 | 0.00 | 0.32 |
| 12 | 3 | 0.59 [0.49, 0.68] | 0.86 [0.60, 1.12] | 0.020 [0.012, 0.028] | 0.058 [0.041, 0.075] | 58.2 | 0.17 | 0.17 | 0.29 |
| 12 | 6 | 0.66 [0.56, 0.75] | 0.76 [0.50, 1.02] | 0.022 [0.013, 0.032] | 0.060 [0.043, 0.077] | 15.9 | 0.01 | 0.00 | 0.33 |
| 12 | exact | 0.66 [0.56, 0.75] | 0.76 [0.50, 1.02] | 0.022 [0.013, 0.032] | 0.060 [0.043, 0.077] | 15.8 | 0.01 | 0.00 | 0.33 |
| 13 | 3 | 0.45 [0.36, 0.55] | 0.98 [0.76, 1.20] | 0.021 [0.015, 0.027] | 0.049 [0.035, 0.063] | 112.0 | 0.35 | 0.35 | 0.28 |
| 13 | 6 | 0.64 [0.54, 0.73] | 0.72 [0.50, 0.94] | 0.018 [0.012, 0.024] | 0.054 [0.039, 0.069] | 17.8 | 0.01 | 0.00 | 0.35 |
| 13 | exact | 0.64 [0.54, 0.73] | 0.72 [0.50, 0.94] | 0.018 [0.012, 0.024] | 0.054 [0.039, 0.069] | 17.8 | 0.01 | 0.00 | 0.35 |
| 14 | 3 | 0.48 [0.38, 0.58] | 1.02 [0.75, 1.29] | 0.018 [0.012, 0.025] | 0.057 [0.041, 0.074] | 80.4 | 0.24 | 0.24 | 0.34 |
| 14 | 6 | 0.66 [0.56, 0.75] | 0.69 [0.47, 0.91] | 0.017 [0.011, 0.023] | 0.056 [0.039, 0.073] | 17.0 | 0.00 | 0.00 | 0.34 |
| 14 | exact | 0.66 [0.56, 0.75] | 0.69 [0.47, 0.91] | 0.017 [0.011, 0.023] | 0.056 [0.039, 0.073] | 17.0 | 0.00 | 0.00 | 0.34 |
| 15 | 3 | 0.35 [0.26, 0.45] | 1.49 [1.19, 1.79] | 0.024 [0.018, 0.030] | 0.072 [0.056, 0.089] | 91.2 | 0.28 | 0.28 | 0.41 |
| 15 | 6 | 0.53 [0.43, 0.62] | 1.12 [0.83, 1.41] | 0.020 [0.014, 0.025] | 0.073 [0.056, 0.090] | 16.5 | 0.00 | 0.00 | 0.47 |
| 15 | exact | 0.53 [0.43, 0.62] | 1.12 [0.83, 1.41] | 0.020 [0.014, 0.025] | 0.073 [0.056, 0.090] | 16.5 | 0.00 | 0.00 | 0.47 |
| 16 | 3 | 0.38 [0.29, 0.48] | 1.49 [1.18, 1.80] | 0.025 [0.018, 0.031] | 0.070 [0.052, 0.088] | 100.7 | 0.31 | 0.31 | 0.34 |
| 16 | 6 | 0.57 [0.47, 0.66] | 1.19 [0.87, 1.51] | 0.019 [0.013, 0.026] | 0.068 [0.050, 0.086] | 18.7 | 0.01 | 0.00 | 0.42 |
| 16 | exact | 0.57 [0.47, 0.66] | 1.19 [0.87, 1.51] | 0.019 [0.013, 0.026] | 0.068 [0.050, 0.086] | 18.7 | 0.01 | 0.00 | 0.42 |
| 17 | 3 | 0.40 [0.31, 0.50] | 1.48 [1.13, 1.83] | 0.029 [0.021, 0.038] | 0.074 [0.055, 0.093] | 135.6 | 0.43 | 0.43 | 0.33 |
| 17 | 6 | 0.66 [0.56, 0.75] | 1.11 [0.75, 1.47] | 0.024 [0.015, 0.033] | 0.064 [0.044, 0.083] | 24.1 | 0.00 | 0.00 | 0.34 |
| 17 | exact | 0.66 [0.56, 0.75] | 1.11 [0.75, 1.47] | 0.024 [0.015, 0.033] | 0.064 [0.044, 0.083] | 24.1 | 0.00 | 0.00 | 0.34 |
| 18 | 3 | 0.38 [0.29, 0.48] | 1.47 [1.15, 1.79] | 0.021 [0.016, 0.027] | 0.064 [0.047, 0.080] | 130.5 | 0.41 | 0.41 | 0.33 |
| 18 | 6 | 0.58 [0.48, 0.67] | 1.25 [0.90, 1.60] | 0.020 [0.014, 0.027] | 0.072 [0.053, 0.091] | 25.6 | 0.01 | 0.00 | 0.41 |
| 18 | exact | 0.58 [0.48, 0.67] | 1.25 [0.90, 1.60] | 0.020 [0.014, 0.027] | 0.072 [0.053, 0.091] | 25.6 | 0.01 | 0.00 | 0.41 |
| 19 | 3 | 0.30 [0.22, 0.40] | 1.71 [1.35, 2.07] | 0.029 [0.022, 0.037] | 0.079 [0.060, 0.097] | 169.8 | 0.55 | 0.55 | 0.32 |
| 19 | 6 | 0.57 [0.47, 0.66] | 1.28 [0.91, 1.65] | 0.024 [0.016, 0.033] | 0.069 [0.050, 0.088] | 21.6 | 0.00 | 0.00 | 0.43 |
| 19 | exact | 0.57 [0.47, 0.66] | 1.28 [0.91, 1.65] | 0.024 [0.016, 0.033] | 0.069 [0.050, 0.088] | 21.6 | 0.00 | 0.00 | 0.43 |
| 20 | 3 | 0.28 [0.20, 0.37] | 1.98 [1.60, 2.36] | 0.030 [0.023, 0.037] | 0.102 [0.081, 0.122] | 142.9 | 0.46 | 0.46 | 0.45 |
| 20 | 6 | 0.56 [0.46, 0.65] | 1.56 [1.15, 1.97] | 0.029 [0.020, 0.037] | 0.089 [0.066, 0.112] | 20.8 | 0.00 | 0.00 | 0.44 |
| 20 | exact | 0.56 [0.46, 0.65] | 1.56 [1.15, 1.97] | 0.029 [0.020, 0.037] | 0.089 [0.066, 0.111] | 20.8 | 0.00 | 0.00 | 0.44 |
| 21 | 3 | 0.20 [0.13, 0.29] | 2.24 [1.88, 2.60] | 0.033 [0.026, 0.040] | 0.101 [0.084, 0.119] | 131.7 | 0.42 | 0.42 | 0.50 |
| 21 | 6 | 0.51 [0.41, 0.61] | 1.47 [1.11, 1.83] | 0.030 [0.022, 0.038] | 0.092 [0.070, 0.113] | 23.4 | 0.00 | 0.00 | 0.49 |
| 21 | exact | 0.51 [0.41, 0.61] | 1.47 [1.11, 1.83] | 0.030 [0.022, 0.038] | 0.092 [0.070, 0.113] | 23.4 | 0.00 | 0.00 | 0.49 |
| 22 | 3 | 0.33 [0.25, 0.43] | 1.94 [1.53, 2.35] | 0.030 [0.022, 0.037] | 0.099 [0.078, 0.119] | 166.1 | 0.54 | 0.54 | 0.41 |
| 22 | 6 | 0.57 [0.47, 0.66] | 1.58 [1.14, 2.02] | 0.029 [0.019, 0.039] | 0.089 [0.065, 0.113] | 28.3 | 0.01 | 0.00 | 0.42 |
| 22 | exact | 0.57 [0.47, 0.66] | 1.58 [1.14, 2.02] | 0.029 [0.019, 0.039] | 0.089 [0.065, 0.113] | 28.3 | 0.01 | 0.00 | 0.42 |

## Findings and caveats

Written 2026-11-25 from the n = 4..20 data; n = 21, 22 (finished later) follow the same pattern:
precision 3 0.20 / 0.33, precision 6 or exact at T = 300 0.51 / 0.57 (oracle rule).

- **The 30-iteration cap is not what limits precision-3 VQPM; rounding is.** With marginals rounded
  to 3 decimals the found-optimum rate is the same for every budget (oracle rule: n = 12 0.59,
  n = 16 0.37 -> 0.38, n = 20 0.28 for T = 30 and T = 300). At T = 300, every precision-3 run that
  reaches the budget is stalled (no change of marginals or locks for >= 10 iterations): 17% of runs
  at n = 12, 31% at n = 16, 46% at n = 20. The rounded marginals become a fixed point.
- **With finer rounding the budget matters and the decline with n flattens.** Precision 6 and exact
  marginals give identical results (6-decimal rounding is effectively exact). Oracle rule,
  T = 30 -> T = 300: n = 12 0.62 -> 0.66, n = 16 0.52 -> 0.57, n = 17 0.48 -> 0.66, n = 20 0.41 -> 0.56.
  At T = 300 almost no run hits the budget (<= 1%); mean iterations used 16 (n = 12) to 21 (n = 20).
  Between n = 12 and n = 20 the rate stays around 0.55-0.66 instead of falling to 0.28.
- **The remaining failures are wrong locks.** Failure split at T = 300 (oracle): precision 6, n = 20:
  44% of runs fail by a wrong lock, 0% by not converging; precision 3, n = 20: 45% wrong lock, 27%
  not converged (stalled). About 5-7% of all locks are wrong, and the fraction of runs with at least
  one wrong lock grows with n (0.26 at n = 8, 0.44 at n = 20 for precision 6); see wrong_locks.md.
- **Missed optima are near-optimal in energy.** Mean normalised energy gap (E_found - E_min)/(E_max - E_min)
  at T = 300 is 0.016-0.030 for all n and precisions, even where the found-optimum rate is 0.3-0.6;
  mean Hamming distance 0.4-2.0 bits.
- **max_prob (no oracle) is lower than oracle** at small n (it can stop on a non-optimal dominant state or
  when all qubits lock), e.g. n = 12 precision 6: 0.54 vs 0.66; budgets matter less for it.
- **Shots-equivalent (for the planned figure):** resolving marginals to d decimals needs about 10^(2d)
  shots per iteration, i.e. ~1e6 for precision 3 (total ~1e7-1e8 per run here) and ~1e12 for
  precision 6 (total ~1e13). The precision-6 gains are therefore not reachable with realistic shot
  counts; Exp. 4 (finite shots) measures what is reachable.
- Caveats: exact marginals (no shot noise, no circuit noise); fixed pdiff = 0.01; paper phase map;
  the oracle rule uses the known optimum to stop; random Gaussian QUBOs only.
- Data for the found-optimum-rate figure (found-optimum rate vs n for precision {3, 6, exact} x budget
  {30, 100, 300, 10n}, with a theory curve and the shots caveat): `vqpm_grid_runs.csv` (one row per
  run: success, iterations used, stop reason, stall flag, shots_equiv_total, energy metrics) and
  `vqpm_grid_summary.csv` (rates with Wilson intervals); the per-iteration trajectories are in the
  raw JSONL, so other budgets can be read off without re-running.

## Regenerate tables and plots from the saved data (no re-run)

```bash
.venv/bin/python experiments/vqpm_grid.py --plot-only --out experiments/results/vqpm_grid/raw/grid.jsonl
.venv/bin/python experiments/wrong_locks.py --grid experiments/results/vqpm_grid/raw/grid.jsonl
.venv/bin/python experiments/reports.py
```
