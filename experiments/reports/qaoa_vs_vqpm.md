# Exp. 5: QAOA (4 optimizers) vs VQPM vs simulated annealing

_Report generated 2026-11-27T07:51:14 by `experiments/reports.py`; status: finished._

## Purpose

A fair QAOA baseline (restarts, INTERP, several optimizers, evaluation counts).

## Method

Statevector QAOA on the paper's scaled QUBO, p in {2, 4, 6, 8}, optimizers L-BFGS-B (exact adjoint gradients), COBYLA, gradient descent (lr 1) and Adam (lr 0.3), cap 500 iterations, 5 random restarts (best energy kept; same initial angles for every optimizer) and INTERP (chain 1..8, 5 random depth-1 starts). Evaluations: nfev + gradient evaluations x (2p+1) (finite-difference-equivalent). VQPM: precision 3, pdiff 0.01, both stop rules, budgets 30 and 100. SA: 10, 100, 1000 sweeps. 100 instances for n <= 10, 50 for n = 12..16.

Metrics (per run, energies in the units of the original, unscaled Q): found optimum = the returned bitstring is the exact ground state (found by enumerating all 2^n energies); bit difference = Hamming distance to the ground state; energy gap = E_found - E_min; normalised energy gap = (E_found - E_min)/(E_max - E_min) in [0, 1]; normalised expected-energy gap = (<E> - E_min)/(E_max - E_min) with <E> the expected energy of the final output distribution (VQPM, QAOA). Intervals: Wilson 95% for rates, mean +- 1.96 s/sqrt(N) for means.

## Provenance

- Invocation 1 of `experiments/results/qaoa_vs_vqpm/raw/qaoa_vs_vqpm.jsonl`: started 2026-11-25T20:19:28, git commit `61a1e1fe0662bf8c454b52a2427ba08b6a976cf9` (dirty tree), python 3.12.12, numpy 2.5.5, scipy 1.18.1
  - command: `.venv/bin/python experiments/qaoa_vs_vqpm.py --n 4 6 8 10 12 --workers 10 --out experiments/results/qaoa_vs_vqpm/raw/qaoa_vs_vqpm.jsonl`
  - parameters: `{"inits": ["random", "interp"], "inst_dir": "experiments/instances", "instances": 50, "instances_small": 100, "maxiter": 500, "n": [4, 6, 8, 10, 12], "optimizers": ["L-BFGS-B", "COBYLA", "gd", "adam"], "out": "experiments/results/qaoa_vs_vqpm/raw/qaoa_vs_vqpm.jsonl", "p": [2, 4, 6, 8], "plot_only": false, "restarts": 5, "sa_sweeps": [10, 100, 1000], "seed": 2026, "skip": [], "small_n": 10, "vqpm_budgets": [30, 100], "workers": 10}`
- Invocation 2 of `experiments/results/qaoa_vs_vqpm/raw/qaoa_vs_vqpm.jsonl`: started 2026-11-25T22:45:08, git commit `61a1e1fe0662bf8c454b52a2427ba08b6a976cf9` (dirty tree), python 3.12.12, numpy 2.5.5, scipy 1.18.1
  - command: `.venv/bin/python experiments/qaoa_vs_vqpm.py --n 4 6 8 10 12 14 16 --workers 10 --out experiments/results/qaoa_vs_vqpm/raw/qaoa_vs_vqpm.jsonl`
  - parameters: `{"inits": ["random", "interp"], "inst_dir": "experiments/instances", "instances": 50, "instances_small": 100, "maxiter": 500, "n": [4, 6, 8, 10, 12, 14, 16], "optimizers": ["L-BFGS-B", "COBYLA", "gd", "adam"], "out": "experiments/results/qaoa_vs_vqpm/raw/qaoa_vs_vqpm.jsonl", "p": [2, 4, 6, 8], "plot_only": false, "restarts": 5, "sa_sweeps": [10, 100, 1000], "seed": 2026, "skip": [], "small_n": 10, "vqpm_budgets": [30, 100], "workers": 10}`
- Library versions (pinned in requirements.txt): matplotlib 3.11.2, numpy 2.5.5, pytest 9.2.0, qiskit 2.6.2, qiskit-aer 0.18.4, qiskit-ibm-runtime 0.52.1, scipy 1.18.1, python 3.12.12
- Machine: AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 16 logical CPUs, 90 GiB RAM, Linux 6.12.113+deb13-amd64
- End time (last write of the raw data): `qaoa_vs_vqpm.jsonl` 2026-11-27T06:34:25
- Wall time `qaoa_vs_vqpm.jsonl` (first start to last write, includes waits between invocations): 34.25 h
- Instances: seed 2026, files `experiments/instances/qubo_n<n>_seed2026.npz` (instance i is `default_rng([2026, n, i])`, Q = A + A^T, A_ij ~ N(0, 1); each file stores Q, optimum, E_min, E_max, gap); sha256 listed below

## Where the data is

Paths are relative to the repository root.

| file (repo-relative) | records / size | sha256 |
|---|---|---|
| `experiments/results/qaoa_vs_vqpm/raw/qaoa_vs_vqpm.jsonl` | 21450 records, 17.8 MiB | `76d00e7ddd2a672b404e4025780e592d14e32b05883f6815379a842eb726d7ad` |
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

- `experiments/results/qaoa_vs_vqpm/qaoa_vs_vqpm.pdf`
- `experiments/results/qaoa_vs_vqpm/qaoa_vs_vqpm.png`
- `experiments/results/qaoa_vs_vqpm/qaoa_vs_vqpm_metrics_COBYLA.pdf`
- `experiments/results/qaoa_vs_vqpm/qaoa_vs_vqpm_metrics_COBYLA.png`
- `experiments/results/qaoa_vs_vqpm/qaoa_vs_vqpm_metrics_L-BFGS-B.pdf`
- `experiments/results/qaoa_vs_vqpm/qaoa_vs_vqpm_metrics_L-BFGS-B.png`
- `experiments/results/qaoa_vs_vqpm/qaoa_vs_vqpm_metrics_adam.pdf`
- `experiments/results/qaoa_vs_vqpm/qaoa_vs_vqpm_metrics_adam.png`
- `experiments/results/qaoa_vs_vqpm/qaoa_vs_vqpm_metrics_all_optimizers_p2.pdf`
- `experiments/results/qaoa_vs_vqpm/qaoa_vs_vqpm_metrics_all_optimizers_p2.png`
- `experiments/results/qaoa_vs_vqpm/qaoa_vs_vqpm_metrics_all_optimizers_p4.pdf`
- `experiments/results/qaoa_vs_vqpm/qaoa_vs_vqpm_metrics_all_optimizers_p4.png`
- `experiments/results/qaoa_vs_vqpm/qaoa_vs_vqpm_metrics_all_optimizers_p6.pdf`
- `experiments/results/qaoa_vs_vqpm/qaoa_vs_vqpm_metrics_all_optimizers_p6.png`
- `experiments/results/qaoa_vs_vqpm/qaoa_vs_vqpm_metrics_all_optimizers_p8.pdf`
- `experiments/results/qaoa_vs_vqpm/qaoa_vs_vqpm_metrics_all_optimizers_p8.png`
- `experiments/results/qaoa_vs_vqpm/qaoa_vs_vqpm_metrics_gd.pdf`
- `experiments/results/qaoa_vs_vqpm/qaoa_vs_vqpm_metrics_gd.png`
- `experiments/results/qaoa_vs_vqpm/qaoa_vs_vqpm_runs.csv` (not committed: large; regenerate)
- `experiments/results/qaoa_vs_vqpm/qaoa_vs_vqpm_summary.csv`

Raw JSONL files are not in the repository because of their size. The scripts write them to `experiments/results/<experiment>/raw/`, and the summary tables above are computed from them.

## Results

| method | n | runs | found-opt [CI] | P(opt) | bit diff | norm. energy gap [CI] | norm. expected-E gap | evals |
|---|---|---|---|---|---|---|---|---|
| SA (10 sweeps) | 4 | 100 | 0.96 [0.90, 0.98] | 0.96 | 0.08 | 0.006 [-0.001, 0.012] | - | 40 |
| SA (10 sweeps) | 6 | 100 | 0.94 [0.88, 0.97] | 0.94 | 0.16 | 0.003 [0.000, 0.005] | - | 60 |
| SA (10 sweeps) | 8 | 100 | 0.91 [0.84, 0.95] | 0.91 | 0.29 | 0.004 [0.001, 0.008] | - | 80 |
| SA (10 sweeps) | 10 | 100 | 0.92 [0.85, 0.96] | 0.92 | 0.31 | 0.004 [0.000, 0.008] | - | 100 |
| SA (10 sweeps) | 12 | 50 | 0.92 [0.81, 0.97] | 0.92 | 0.26 | 0.002 [-0.000, 0.005] | - | 120 |
| SA (10 sweeps) | 14 | 50 | 0.70 [0.56, 0.81] | 0.70 | 1.16 | 0.010 [0.003, 0.017] | - | 140 |
| SA (10 sweeps) | 16 | 50 | 0.94 [0.84, 0.98] | 0.94 | 0.34 | 0.001 [-0.001, 0.002] | - | 160 |
| SA (100 sweeps) | 4 | 100 | 1.00 [0.96, 1.00] | 1.00 | 0.00 | 0.000 [-0.000, 0.000] | - | 400 |
| SA (100 sweeps) | 6 | 100 | 1.00 [0.96, 1.00] | 1.00 | 0.00 | 0.000 [-0.000, 0.000] | - | 600 |
| SA (100 sweeps) | 8 | 100 | 1.00 [0.96, 1.00] | 1.00 | 0.00 | 0.000 [-0.000, 0.000] | - | 800 |
| SA (100 sweeps) | 10 | 100 | 1.00 [0.96, 1.00] | 1.00 | 0.00 | 0.000 [-0.000, 0.000] | - | 1000 |
| SA (100 sweeps) | 12 | 50 | 0.98 [0.90, 1.00] | 0.98 | 0.04 | 0.000 [-0.000, 0.001] | - | 1200 |
| SA (100 sweeps) | 14 | 50 | 0.98 [0.90, 1.00] | 0.98 | 0.12 | 0.000 [-0.000, 0.000] | - | 1400 |
| SA (100 sweeps) | 16 | 50 | 0.98 [0.90, 1.00] | 0.98 | 0.08 | 0.000 [-0.000, 0.000] | - | 1600 |
| SA (1000 sweeps) | 4 | 100 | 1.00 [0.96, 1.00] | 1.00 | 0.00 | 0.000 [-0.000, 0.000] | - | 4000 |
| SA (1000 sweeps) | 6 | 100 | 1.00 [0.96, 1.00] | 1.00 | 0.00 | 0.000 [-0.000, 0.000] | - | 6000 |
| SA (1000 sweeps) | 8 | 100 | 1.00 [0.96, 1.00] | 1.00 | 0.00 | 0.000 [-0.000, 0.000] | - | 8000 |
| SA (1000 sweeps) | 10 | 100 | 1.00 [0.96, 1.00] | 1.00 | 0.00 | 0.000 [-0.000, 0.000] | - | 10000 |
| SA (1000 sweeps) | 12 | 50 | 1.00 [0.93, 1.00] | 1.00 | 0.00 | 0.000 [-0.000, 0.000] | - | 12000 |
| SA (1000 sweeps) | 14 | 50 | 1.00 [0.93, 1.00] | 1.00 | 0.00 | 0.000 [-0.000, 0.000] | - | 14000 |
| SA (1000 sweeps) | 16 | 50 | 1.00 [0.93, 1.00] | 1.00 | 0.00 | 0.000 [-0.000, 0.000] | - | 16000 |
| QAOA gd interp p=2 | 4 | 100 | 0.83 [0.74, 0.89] | 0.50 | 0.19 | 0.007 [0.003, 0.011] | 0.086 | 19946 |
| QAOA gd interp p=2 | 6 | 100 | 0.65 [0.55, 0.74] | 0.21 | 0.45 | 0.016 [0.010, 0.022] | 0.142 | 20000 |
| QAOA gd interp p=2 | 8 | 100 | 0.58 [0.48, 0.67] | 0.09 | 0.65 | 0.017 [0.011, 0.024] | 0.204 | 20000 |
| QAOA gd interp p=2 | 10 | 100 | 0.59 [0.49, 0.68] | 0.03 | 0.78 | 0.010 [0.006, 0.013] | 0.268 | 20000 |
| QAOA gd interp p=2 | 12 | 50 | 0.66 [0.52, 0.78] | 0.01 | 0.50 | 0.010 [0.004, 0.015] | 0.327 | 20000 |
| QAOA gd interp p=2 | 14 | 50 | 0.78 [0.65, 0.87] | 0.00 | 0.46 | 0.004 [0.000, 0.007] | 0.360 | 20000 |
| QAOA gd interp p=2 | 16 | 50 | 0.74 [0.60, 0.84] | 0.00 | 0.52 | 0.002 [0.001, 0.004] | 0.420 | 20000 |
| QAOA gd interp p=4 | 4 | 100 | 0.89 [0.81, 0.94] | 0.65 | 0.13 | 0.003 [0.000, 0.005] | 0.050 | 59946 |
| QAOA gd interp p=4 | 6 | 100 | 0.72 [0.63, 0.80] | 0.35 | 0.37 | 0.008 [0.005, 0.012] | 0.075 | 60000 |
| QAOA gd interp p=4 | 8 | 100 | 0.62 [0.52, 0.71] | 0.23 | 0.67 | 0.014 [0.009, 0.019] | 0.099 | 60000 |
| QAOA gd interp p=4 | 10 | 100 | 0.63 [0.53, 0.72] | 0.11 | 0.73 | 0.008 [0.005, 0.011] | 0.142 | 60000 |
| QAOA gd interp p=4 | 12 | 50 | 0.56 [0.42, 0.69] | 0.04 | 0.78 | 0.013 [0.007, 0.018] | 0.191 | 60000 |
| QAOA gd interp p=4 | 14 | 50 | 0.50 [0.37, 0.63] | 0.01 | 1.02 | 0.014 [0.008, 0.021] | 0.230 | 60000 |
| QAOA gd interp p=4 | 16 | 50 | 0.52 [0.39, 0.65] | 0.00 | 1.00 | 0.008 [0.005, 0.011] | 0.301 | 60000 |
| QAOA gd interp p=6 | 4 | 100 | 0.93 [0.86, 0.97] | 0.70 | 0.08 | 0.002 [-0.000, 0.004] | 0.049 | 119946 |
| QAOA gd interp p=6 | 6 | 100 | 0.78 [0.69, 0.85] | 0.44 | 0.32 | 0.006 [0.003, 0.009] | 0.051 | 120000 |
| QAOA gd interp p=6 | 8 | 100 | 0.77 [0.68, 0.84] | 0.32 | 0.46 | 0.006 [0.003, 0.009] | 0.070 | 120000 |
| QAOA gd interp p=6 | 10 | 100 | 0.69 [0.59, 0.77] | 0.19 | 0.64 | 0.007 [0.004, 0.010] | 0.098 | 120000 |
| QAOA gd interp p=6 | 12 | 50 | 0.60 [0.46, 0.72] | 0.09 | 0.90 | 0.013 [0.007, 0.019] | 0.129 | 120000 |
| QAOA gd interp p=6 | 14 | 50 | 0.46 [0.33, 0.60] | 0.03 | 1.06 | 0.017 [0.010, 0.023] | 0.162 | 120000 |
| QAOA gd interp p=6 | 16 | 50 | 0.44 [0.31, 0.58] | 0.01 | 1.30 | 0.012 [0.008, 0.016] | 0.221 | 120000 |
| QAOA gd interp p=8 | 4 | 100 | 0.94 [0.88, 0.97] | 0.73 | 0.07 | 0.001 [-0.000, 0.001] | 0.052 | 199946 |
| QAOA gd interp p=8 | 6 | 100 | 0.86 [0.78, 0.91] | 0.51 | 0.19 | 0.002 [0.001, 0.004] | 0.045 | 200000 |
| QAOA gd interp p=8 | 8 | 100 | 0.83 [0.74, 0.89] | 0.36 | 0.37 | 0.005 [0.002, 0.008] | 0.067 | 200000 |
| QAOA gd interp p=8 | 10 | 100 | 0.69 [0.59, 0.77] | 0.23 | 0.57 | 0.007 [0.004, 0.009] | 0.083 | 200000 |
| QAOA gd interp p=8 | 12 | 50 | 0.66 [0.52, 0.78] | 0.13 | 0.78 | 0.008 [0.003, 0.013] | 0.104 | 200000 |
| QAOA gd interp p=8 | 14 | 50 | 0.54 [0.40, 0.67] | 0.05 | 1.00 | 0.010 [0.006, 0.014] | 0.125 | 200000 |
| QAOA gd interp p=8 | 16 | 50 | 0.40 [0.28, 0.54] | 0.02 | 1.32 | 0.014 [0.010, 0.019] | 0.177 | 200000 |
| QAOA gd random p=2 | 4 | 100 | 0.78 [0.69, 0.85] | 0.46 | 0.25 | 0.011 [0.006, 0.017] | 0.098 | 12500 |
| QAOA gd random p=2 | 6 | 100 | 0.59 [0.49, 0.68] | 0.16 | 0.50 | 0.020 [0.014, 0.027] | 0.176 | 12500 |
| QAOA gd random p=2 | 8 | 100 | 0.58 [0.48, 0.67] | 0.05 | 0.68 | 0.018 [0.012, 0.025] | 0.263 | 12500 |
| QAOA gd random p=2 | 10 | 100 | 0.67 [0.57, 0.75] | 0.01 | 0.68 | 0.008 [0.004, 0.012] | 0.331 | 12500 |
| QAOA gd random p=2 | 12 | 50 | 0.66 [0.52, 0.78] | 0.00 | 0.50 | 0.007 [0.002, 0.011] | 0.381 | 12500 |
| QAOA gd random p=2 | 14 | 50 | 0.80 [0.67, 0.89] | 0.00 | 0.38 | 0.002 [-0.000, 0.005] | 0.404 | 12500 |
| QAOA gd random p=2 | 16 | 50 | 0.78 [0.65, 0.87] | 0.00 | 0.46 | 0.001 [0.000, 0.002] | 0.448 | 12500 |
| QAOA gd random p=4 | 4 | 100 | 0.85 [0.77, 0.91] | 0.53 | 0.16 | 0.005 [0.002, 0.008] | 0.072 | 22500 |
| QAOA gd random p=4 | 6 | 100 | 0.63 [0.53, 0.72] | 0.22 | 0.44 | 0.017 [0.010, 0.023] | 0.134 | 22500 |
| QAOA gd random p=4 | 8 | 100 | 0.57 [0.47, 0.66] | 0.10 | 0.67 | 0.018 [0.012, 0.025] | 0.188 | 22500 |
| QAOA gd random p=4 | 10 | 100 | 0.58 [0.48, 0.67] | 0.03 | 0.77 | 0.011 [0.007, 0.014] | 0.256 | 22500 |
| QAOA gd random p=4 | 12 | 50 | 0.66 [0.52, 0.78] | 0.01 | 0.50 | 0.010 [0.004, 0.015] | 0.310 | 22500 |
| QAOA gd random p=4 | 14 | 50 | 0.72 [0.58, 0.83] | 0.00 | 0.66 | 0.005 [0.001, 0.009] | 0.342 | 22500 |
| QAOA gd random p=4 | 16 | 50 | 0.70 [0.56, 0.81] | 0.00 | 0.56 | 0.003 [0.001, 0.005] | 0.407 | 22500 |
| QAOA gd random p=6 | 4 | 100 | 0.86 [0.78, 0.91] | 0.55 | 0.16 | 0.004 [0.001, 0.007] | 0.068 | 32500 |
| QAOA gd random p=6 | 6 | 100 | 0.63 [0.53, 0.72] | 0.22 | 0.47 | 0.017 [0.011, 0.023] | 0.125 | 32500 |
| QAOA gd random p=6 | 8 | 100 | 0.59 [0.49, 0.68] | 0.12 | 0.65 | 0.016 [0.011, 0.022] | 0.163 | 32500 |
| QAOA gd random p=6 | 10 | 100 | 0.60 [0.50, 0.69] | 0.04 | 0.80 | 0.010 [0.006, 0.013] | 0.219 | 32500 |
| QAOA gd random p=6 | 12 | 50 | 0.60 [0.46, 0.72] | 0.01 | 0.64 | 0.011 [0.005, 0.017] | 0.265 | 32500 |
| QAOA gd random p=6 | 14 | 50 | 0.58 [0.44, 0.71] | 0.00 | 0.94 | 0.012 [0.005, 0.019] | 0.301 | 32500 |
| QAOA gd random p=6 | 16 | 50 | 0.62 [0.48, 0.74] | 0.00 | 0.82 | 0.005 [0.003, 0.008] | 0.368 | 32500 |
| QAOA gd random p=8 | 4 | 100 | 0.86 [0.78, 0.91] | 0.56 | 0.16 | 0.004 [0.001, 0.007] | 0.071 | 42500 |
| QAOA gd random p=8 | 6 | 100 | 0.67 [0.57, 0.75] | 0.24 | 0.40 | 0.014 [0.008, 0.020] | 0.120 | 42500 |
| QAOA gd random p=8 | 8 | 100 | 0.61 [0.51, 0.70] | 0.12 | 0.73 | 0.016 [0.010, 0.022] | 0.163 | 42500 |
| QAOA gd random p=8 | 10 | 100 | 0.60 [0.50, 0.69] | 0.05 | 0.77 | 0.009 [0.006, 0.013] | 0.202 | 42500 |
| QAOA gd random p=8 | 12 | 50 | 0.64 [0.50, 0.76] | 0.02 | 0.64 | 0.011 [0.005, 0.017] | 0.240 | 42500 |
| QAOA gd random p=8 | 14 | 50 | 0.52 [0.39, 0.65] | 0.00 | 1.08 | 0.015 [0.008, 0.022] | 0.273 | 42500 |
| QAOA gd random p=8 | 16 | 50 | 0.58 [0.44, 0.71] | 0.00 | 0.66 | 0.006 [0.003, 0.009] | 0.337 | 42500 |
| QAOA adam interp p=2 | 4 | 100 | 0.86 [0.78, 0.91] | 0.54 | 0.16 | 0.005 [0.002, 0.008] | 0.076 | 8684 |
| QAOA adam interp p=2 | 6 | 100 | 0.59 [0.49, 0.68] | 0.27 | 0.53 | 0.019 [0.013, 0.025] | 0.108 | 8473 |
| QAOA adam interp p=2 | 8 | 100 | 0.56 [0.46, 0.65] | 0.19 | 0.70 | 0.019 [0.013, 0.025] | 0.117 | 8796 |
| QAOA adam interp p=2 | 10 | 100 | 0.59 [0.49, 0.68] | 0.13 | 0.77 | 0.013 [0.008, 0.018] | 0.126 | 9547 |
| QAOA adam interp p=2 | 12 | 50 | 0.52 [0.39, 0.65] | 0.08 | 1.08 | 0.017 [0.010, 0.024] | 0.134 | 10560 |
| QAOA adam interp p=2 | 14 | 50 | 0.36 [0.24, 0.50] | 0.05 | 1.40 | 0.023 [0.015, 0.030] | 0.132 | 11903 |
| QAOA adam interp p=2 | 16 | 50 | 0.30 [0.19, 0.44] | 0.03 | 1.74 | 0.021 [0.015, 0.027] | 0.145 | 12646 |
| QAOA adam interp p=4 | 4 | 100 | 0.91 [0.84, 0.95] | 0.75 | 0.10 | 0.001 [0.000, 0.003] | 0.022 | 29238 |
| QAOA adam interp p=4 | 6 | 100 | 0.79 [0.70, 0.86] | 0.47 | 0.33 | 0.006 [0.003, 0.009] | 0.044 | 27650 |
| QAOA adam interp p=4 | 8 | 100 | 0.81 [0.72, 0.87] | 0.41 | 0.40 | 0.005 [0.002, 0.007] | 0.048 | 27415 |
| QAOA adam interp p=4 | 10 | 100 | 0.75 [0.66, 0.82] | 0.33 | 0.46 | 0.004 [0.003, 0.006] | 0.054 | 28083 |
| QAOA adam interp p=4 | 12 | 50 | 0.70 [0.56, 0.81] | 0.25 | 0.70 | 0.008 [0.002, 0.014] | 0.062 | 28867 |
| QAOA adam interp p=4 | 14 | 50 | 0.74 [0.60, 0.84] | 0.15 | 0.60 | 0.003 [0.001, 0.006] | 0.063 | 31997 |
| QAOA adam interp p=4 | 16 | 50 | 0.62 [0.48, 0.74] | 0.12 | 0.98 | 0.006 [0.003, 0.009] | 0.068 | 31312 |
| QAOA adam interp p=6 | 4 | 100 | 0.94 [0.88, 0.97] | 0.84 | 0.08 | 0.000 [0.000, 0.001] | 0.009 | 71207 |
| QAOA adam interp p=6 | 6 | 100 | 0.88 [0.80, 0.93] | 0.60 | 0.16 | 0.002 [0.000, 0.004] | 0.023 | 62081 |
| QAOA adam interp p=6 | 8 | 100 | 0.87 [0.79, 0.92] | 0.55 | 0.28 | 0.002 [0.001, 0.003] | 0.026 | 60438 |
| QAOA adam interp p=6 | 10 | 100 | 0.80 [0.71, 0.87] | 0.47 | 0.36 | 0.003 [0.001, 0.004] | 0.028 | 60332 |
| QAOA adam interp p=6 | 12 | 50 | 0.82 [0.69, 0.90] | 0.38 | 0.44 | 0.003 [0.000, 0.005] | 0.036 | 62827 |
| QAOA adam interp p=6 | 14 | 50 | 0.76 [0.63, 0.86] | 0.26 | 0.54 | 0.001 [0.001, 0.002] | 0.037 | 64981 |
| QAOA adam interp p=6 | 16 | 50 | 0.70 [0.56, 0.81] | 0.23 | 0.58 | 0.004 [0.001, 0.006] | 0.040 | 65919 |
| QAOA adam interp p=8 | 4 | 100 | 0.93 [0.86, 0.97] | 0.86 | 0.09 | 0.000 [0.000, 0.001] | 0.007 | 146685 |
| QAOA adam interp p=8 | 6 | 100 | 0.90 [0.83, 0.94] | 0.67 | 0.13 | 0.002 [0.000, 0.004] | 0.018 | 124935 |
| QAOA adam interp p=8 | 8 | 100 | 0.91 [0.84, 0.95] | 0.64 | 0.21 | 0.001 [0.000, 0.002] | 0.017 | 120346 |
| QAOA adam interp p=8 | 10 | 100 | 0.87 [0.79, 0.92] | 0.56 | 0.20 | 0.001 [0.000, 0.002] | 0.017 | 119582 |
| QAOA adam interp p=8 | 12 | 50 | 0.86 [0.74, 0.93] | 0.49 | 0.30 | 0.002 [-0.000, 0.003] | 0.023 | 121283 |
| QAOA adam interp p=8 | 14 | 50 | 0.80 [0.67, 0.89] | 0.36 | 0.50 | 0.001 [0.000, 0.002] | 0.025 | 130218 |
| QAOA adam interp p=8 | 16 | 50 | 0.76 [0.63, 0.86] | 0.32 | 0.42 | 0.002 [0.001, 0.004] | 0.026 | 127845 |
| QAOA adam random p=2 | 4 | 100 | 0.83 [0.74, 0.89] | 0.54 | 0.19 | 0.006 [0.002, 0.009] | 0.073 | 6349 |
| QAOA adam random p=2 | 6 | 100 | 0.59 [0.49, 0.68] | 0.27 | 0.53 | 0.020 [0.013, 0.026] | 0.107 | 6983 |
| QAOA adam random p=2 | 8 | 100 | 0.57 [0.47, 0.66] | 0.19 | 0.67 | 0.018 [0.012, 0.024] | 0.116 | 8493 |
| QAOA adam random p=2 | 10 | 100 | 0.59 [0.49, 0.68] | 0.13 | 0.80 | 0.013 [0.008, 0.018] | 0.125 | 10085 |
| QAOA adam random p=2 | 12 | 50 | 0.52 [0.39, 0.65] | 0.08 | 1.08 | 0.017 [0.010, 0.024] | 0.133 | 11376 |
| QAOA adam random p=2 | 14 | 50 | 0.36 [0.24, 0.50] | 0.05 | 1.40 | 0.023 [0.015, 0.030] | 0.132 | 12269 |
| QAOA adam random p=2 | 16 | 50 | 0.30 [0.19, 0.44] | 0.03 | 1.72 | 0.021 [0.015, 0.027] | 0.145 | 12457 |
| QAOA adam random p=4 | 4 | 100 | 0.89 [0.81, 0.94] | 0.76 | 0.12 | 0.002 [0.000, 0.003] | 0.021 | 19774 |
| QAOA adam random p=4 | 6 | 100 | 0.76 [0.67, 0.83] | 0.46 | 0.35 | 0.007 [0.004, 0.010] | 0.044 | 19868 |
| QAOA adam random p=4 | 8 | 100 | 0.83 [0.74, 0.89] | 0.41 | 0.35 | 0.004 [0.002, 0.006] | 0.049 | 21464 |
| QAOA adam random p=4 | 10 | 100 | 0.76 [0.67, 0.83] | 0.33 | 0.49 | 0.005 [0.003, 0.007] | 0.055 | 22232 |
| QAOA adam random p=4 | 12 | 50 | 0.66 [0.52, 0.78] | 0.24 | 0.60 | 0.006 [0.003, 0.010] | 0.064 | 22417 |
| QAOA adam random p=4 | 14 | 50 | 0.76 [0.63, 0.86] | 0.13 | 0.52 | 0.003 [0.000, 0.005] | 0.068 | 22500 |
| QAOA adam random p=4 | 16 | 50 | 0.58 [0.44, 0.71] | 0.10 | 1.20 | 0.008 [0.004, 0.011] | 0.075 | 22500 |
| QAOA adam random p=6 | 4 | 100 | 0.93 [0.86, 0.97] | 0.84 | 0.08 | 0.001 [0.000, 0.002] | 0.009 | 32381 |
| QAOA adam random p=6 | 6 | 100 | 0.88 [0.80, 0.93] | 0.58 | 0.16 | 0.002 [0.000, 0.004] | 0.026 | 32241 |
| QAOA adam random p=6 | 8 | 100 | 0.88 [0.80, 0.93] | 0.52 | 0.25 | 0.001 [0.000, 0.002] | 0.029 | 32433 |
| QAOA adam random p=6 | 10 | 100 | 0.80 [0.71, 0.87] | 0.44 | 0.37 | 0.003 [0.002, 0.005] | 0.032 | 32500 |
| QAOA adam random p=6 | 12 | 50 | 0.76 [0.63, 0.86] | 0.33 | 0.48 | 0.003 [0.001, 0.005] | 0.042 | 32500 |
| QAOA adam random p=6 | 14 | 50 | 0.74 [0.60, 0.84] | 0.20 | 0.50 | 0.002 [0.001, 0.002] | 0.047 | 32500 |
| QAOA adam random p=6 | 16 | 50 | 0.68 [0.54, 0.79] | 0.16 | 0.90 | 0.005 [0.002, 0.009] | 0.052 | 32500 |
| QAOA adam random p=8 | 4 | 100 | 0.93 [0.86, 0.97] | 0.87 | 0.08 | 0.001 [-0.000, 0.002] | 0.006 | 42500 |
| QAOA adam random p=8 | 6 | 100 | 0.84 [0.76, 0.90] | 0.65 | 0.20 | 0.003 [0.001, 0.005] | 0.017 | 42500 |
| QAOA adam random p=8 | 8 | 100 | 0.92 [0.85, 0.96] | 0.59 | 0.17 | 0.001 [-0.000, 0.002] | 0.021 | 42500 |
| QAOA adam random p=8 | 10 | 100 | 0.82 [0.73, 0.88] | 0.50 | 0.29 | 0.003 [0.001, 0.004] | 0.024 | 42500 |
| QAOA adam random p=8 | 12 | 50 | 0.86 [0.74, 0.93] | 0.39 | 0.26 | 0.002 [-0.000, 0.004] | 0.033 | 42500 |
| QAOA adam random p=8 | 14 | 50 | 0.78 [0.65, 0.87] | 0.26 | 0.52 | 0.001 [0.000, 0.002] | 0.036 | 42500 |
| QAOA adam random p=8 | 16 | 50 | 0.68 [0.54, 0.79] | 0.20 | 0.70 | 0.005 [0.002, 0.007] | 0.042 | 42500 |
| QAOA COBYLA interp p=2 | 4 | 100 | 0.81 [0.72, 0.87] | 0.52 | 0.21 | 0.007 [0.003, 0.011] | 0.081 | 4475 |
| QAOA COBYLA interp p=2 | 6 | 100 | 0.59 [0.49, 0.68] | 0.24 | 0.52 | 0.019 [0.013, 0.026] | 0.118 | 4995 |
| QAOA COBYLA interp p=2 | 8 | 100 | 0.55 [0.45, 0.64] | 0.16 | 0.76 | 0.021 [0.014, 0.028] | 0.135 | 5000 |
| QAOA COBYLA interp p=2 | 10 | 100 | 0.55 [0.45, 0.64] | 0.09 | 0.81 | 0.014 [0.010, 0.018] | 0.162 | 5000 |
| QAOA COBYLA interp p=2 | 12 | 50 | 0.50 [0.37, 0.63] | 0.04 | 0.92 | 0.018 [0.011, 0.025] | 0.185 | 5000 |
| QAOA COBYLA interp p=2 | 14 | 50 | 0.50 [0.37, 0.63] | 0.02 | 1.04 | 0.016 [0.009, 0.023] | 0.200 | 5000 |
| QAOA COBYLA interp p=2 | 16 | 50 | 0.46 [0.33, 0.60] | 0.01 | 1.20 | 0.012 [0.007, 0.016] | 0.243 | 5000 |
| QAOA COBYLA interp p=4 | 4 | 100 | 0.87 [0.79, 0.92] | 0.70 | 0.15 | 0.003 [0.001, 0.005] | 0.030 | 9475 |
| QAOA COBYLA interp p=4 | 6 | 100 | 0.76 [0.67, 0.83] | 0.42 | 0.34 | 0.007 [0.004, 0.010] | 0.058 | 9995 |
| QAOA COBYLA interp p=4 | 8 | 100 | 0.75 [0.66, 0.82] | 0.34 | 0.51 | 0.008 [0.004, 0.012] | 0.070 | 10000 |
| QAOA COBYLA interp p=4 | 10 | 100 | 0.72 [0.63, 0.80] | 0.23 | 0.58 | 0.008 [0.004, 0.011] | 0.086 | 10000 |
| QAOA COBYLA interp p=4 | 12 | 50 | 0.56 [0.42, 0.69] | 0.14 | 0.98 | 0.011 [0.006, 0.016] | 0.101 | 10000 |
| QAOA COBYLA interp p=4 | 14 | 50 | 0.52 [0.39, 0.65] | 0.06 | 1.00 | 0.010 [0.006, 0.014] | 0.111 | 10000 |
| QAOA COBYLA interp p=4 | 16 | 50 | 0.42 [0.29, 0.56] | 0.03 | 1.46 | 0.014 [0.009, 0.018] | 0.139 | 10000 |
| QAOA COBYLA interp p=6 | 4 | 100 | 0.94 [0.88, 0.97] | 0.80 | 0.07 | 0.001 [-0.000, 0.001] | 0.014 | 14475 |
| QAOA COBYLA interp p=6 | 6 | 100 | 0.88 [0.80, 0.93] | 0.55 | 0.17 | 0.002 [0.000, 0.004] | 0.033 | 14995 |
| QAOA COBYLA interp p=6 | 8 | 100 | 0.89 [0.81, 0.94] | 0.47 | 0.27 | 0.002 [0.000, 0.004] | 0.045 | 15000 |
| QAOA COBYLA interp p=6 | 10 | 100 | 0.78 [0.69, 0.85] | 0.34 | 0.43 | 0.004 [0.002, 0.006] | 0.056 | 15000 |
| QAOA COBYLA interp p=6 | 12 | 50 | 0.66 [0.52, 0.78] | 0.25 | 0.66 | 0.007 [0.003, 0.012] | 0.067 | 15000 |
| QAOA COBYLA interp p=6 | 14 | 50 | 0.68 [0.54, 0.79] | 0.12 | 0.78 | 0.005 [0.002, 0.008] | 0.074 | 15000 |
| QAOA COBYLA interp p=6 | 16 | 50 | 0.58 [0.44, 0.71] | 0.08 | 1.06 | 0.008 [0.004, 0.011] | 0.089 | 15000 |
| QAOA COBYLA interp p=8 | 4 | 100 | 0.95 [0.89, 0.98] | 0.86 | 0.06 | 0.000 [0.000, 0.001] | 0.007 | 19475 |
| QAOA COBYLA interp p=8 | 6 | 100 | 0.87 [0.79, 0.92] | 0.65 | 0.17 | 0.002 [0.000, 0.004] | 0.021 | 19995 |
| QAOA COBYLA interp p=8 | 8 | 100 | 0.91 [0.84, 0.95] | 0.56 | 0.22 | 0.001 [0.000, 0.002] | 0.032 | 20000 |
| QAOA COBYLA interp p=8 | 10 | 100 | 0.84 [0.76, 0.90] | 0.44 | 0.27 | 0.002 [0.001, 0.004] | 0.038 | 20000 |
| QAOA COBYLA interp p=8 | 12 | 50 | 0.78 [0.65, 0.87] | 0.33 | 0.38 | 0.004 [-0.000, 0.008] | 0.048 | 20000 |
| QAOA COBYLA interp p=8 | 14 | 50 | 0.76 [0.63, 0.86] | 0.19 | 0.46 | 0.002 [0.000, 0.003] | 0.053 | 20000 |
| QAOA COBYLA interp p=8 | 16 | 50 | 0.62 [0.48, 0.74] | 0.13 | 1.12 | 0.007 [0.003, 0.010] | 0.063 | 20000 |
| QAOA COBYLA random p=2 | 4 | 100 | 0.81 [0.72, 0.87] | 0.48 | 0.20 | 0.009 [0.004, 0.014] | 0.088 | 2500 |
| QAOA COBYLA random p=2 | 6 | 100 | 0.62 [0.52, 0.71] | 0.19 | 0.45 | 0.016 [0.010, 0.022] | 0.147 | 2500 |
| QAOA COBYLA random p=2 | 8 | 100 | 0.52 [0.42, 0.62] | 0.09 | 0.77 | 0.022 [0.015, 0.029] | 0.192 | 2500 |
| QAOA COBYLA random p=2 | 10 | 100 | 0.57 [0.47, 0.66] | 0.03 | 0.80 | 0.012 [0.008, 0.016] | 0.235 | 2500 |
| QAOA COBYLA random p=2 | 12 | 50 | 0.64 [0.50, 0.76] | 0.01 | 0.60 | 0.011 [0.005, 0.017] | 0.272 | 2500 |
| QAOA COBYLA random p=2 | 14 | 50 | 0.62 [0.48, 0.74] | 0.00 | 0.84 | 0.010 [0.004, 0.017] | 0.294 | 2500 |
| QAOA COBYLA random p=2 | 16 | 50 | 0.58 [0.44, 0.71] | 0.00 | 0.72 | 0.005 [0.003, 0.008] | 0.348 | 2500 |
| QAOA COBYLA random p=4 | 4 | 100 | 0.84 [0.76, 0.90] | 0.51 | 0.17 | 0.006 [0.002, 0.009] | 0.078 | 2500 |
| QAOA COBYLA random p=4 | 6 | 100 | 0.64 [0.54, 0.73] | 0.21 | 0.44 | 0.017 [0.011, 0.023] | 0.134 | 2500 |
| QAOA COBYLA random p=4 | 8 | 100 | 0.57 [0.47, 0.66] | 0.10 | 0.68 | 0.018 [0.012, 0.024] | 0.180 | 2500 |
| QAOA COBYLA random p=4 | 10 | 100 | 0.60 [0.50, 0.69] | 0.04 | 0.78 | 0.009 [0.006, 0.012] | 0.227 | 2500 |
| QAOA COBYLA random p=4 | 12 | 50 | 0.60 [0.46, 0.72] | 0.01 | 0.66 | 0.012 [0.006, 0.019] | 0.264 | 2500 |
| QAOA COBYLA random p=4 | 14 | 50 | 0.60 [0.46, 0.72] | 0.00 | 0.86 | 0.010 [0.004, 0.016] | 0.284 | 2500 |
| QAOA COBYLA random p=4 | 16 | 50 | 0.58 [0.44, 0.71] | 0.00 | 0.90 | 0.006 [0.003, 0.008] | 0.339 | 2500 |
| QAOA COBYLA random p=6 | 4 | 100 | 0.86 [0.78, 0.91] | 0.55 | 0.15 | 0.004 [0.001, 0.007] | 0.067 | 2500 |
| QAOA COBYLA random p=6 | 6 | 100 | 0.62 [0.52, 0.71] | 0.23 | 0.47 | 0.017 [0.011, 0.024] | 0.124 | 2500 |
| QAOA COBYLA random p=6 | 8 | 100 | 0.59 [0.49, 0.68] | 0.12 | 0.69 | 0.015 [0.010, 0.021] | 0.164 | 2500 |
| QAOA COBYLA random p=6 | 10 | 100 | 0.58 [0.48, 0.67] | 0.05 | 0.72 | 0.010 [0.007, 0.013] | 0.212 | 2500 |
| QAOA COBYLA random p=6 | 12 | 50 | 0.60 [0.46, 0.72] | 0.01 | 0.66 | 0.012 [0.006, 0.019] | 0.250 | 2500 |
| QAOA COBYLA random p=6 | 14 | 50 | 0.56 [0.42, 0.69] | 0.00 | 0.92 | 0.012 [0.006, 0.019] | 0.273 | 2500 |
| QAOA COBYLA random p=6 | 16 | 50 | 0.52 [0.39, 0.65] | 0.00 | 0.94 | 0.007 [0.004, 0.010] | 0.326 | 2500 |
| QAOA COBYLA random p=8 | 4 | 100 | 0.89 [0.81, 0.94] | 0.59 | 0.12 | 0.003 [0.001, 0.005] | 0.057 | 2500 |
| QAOA COBYLA random p=8 | 6 | 100 | 0.66 [0.56, 0.75] | 0.24 | 0.41 | 0.015 [0.009, 0.020] | 0.115 | 2500 |
| QAOA COBYLA random p=8 | 8 | 100 | 0.59 [0.49, 0.68] | 0.14 | 0.64 | 0.017 [0.011, 0.022] | 0.149 | 2500 |
| QAOA COBYLA random p=8 | 10 | 100 | 0.57 [0.47, 0.66] | 0.05 | 0.83 | 0.011 [0.007, 0.015] | 0.197 | 2500 |
| QAOA COBYLA random p=8 | 12 | 50 | 0.62 [0.48, 0.74] | 0.02 | 0.66 | 0.011 [0.005, 0.017] | 0.231 | 2500 |
| QAOA COBYLA random p=8 | 14 | 50 | 0.48 [0.35, 0.61] | 0.01 | 1.08 | 0.016 [0.009, 0.023] | 0.259 | 2500 |
| QAOA COBYLA random p=8 | 16 | 50 | 0.56 [0.42, 0.69] | 0.00 | 0.98 | 0.007 [0.004, 0.010] | 0.311 | 2500 |
| QAOA L-BFGS-B interp p=2 | 4 | 100 | 0.86 [0.78, 0.91] | 0.54 | 0.16 | 0.005 [0.002, 0.008] | 0.076 | 773 |
| QAOA L-BFGS-B interp p=2 | 6 | 100 | 0.59 [0.49, 0.68] | 0.27 | 0.53 | 0.019 [0.013, 0.025] | 0.108 | 856 |
| QAOA L-BFGS-B interp p=2 | 8 | 100 | 0.56 [0.46, 0.65] | 0.19 | 0.70 | 0.019 [0.013, 0.025] | 0.117 | 909 |
| QAOA L-BFGS-B interp p=2 | 10 | 100 | 0.59 [0.49, 0.68] | 0.13 | 0.77 | 0.013 [0.008, 0.018] | 0.126 | 958 |
| QAOA L-BFGS-B interp p=2 | 12 | 50 | 0.52 [0.39, 0.65] | 0.08 | 1.08 | 0.017 [0.010, 0.024] | 0.134 | 989 |
| QAOA L-BFGS-B interp p=2 | 14 | 50 | 0.36 [0.24, 0.50] | 0.05 | 1.40 | 0.023 [0.015, 0.030] | 0.132 | 1026 |
| QAOA L-BFGS-B interp p=2 | 16 | 50 | 0.30 [0.19, 0.44] | 0.03 | 1.74 | 0.021 [0.015, 0.027] | 0.145 | 1041 |
| QAOA L-BFGS-B interp p=4 | 4 | 100 | 0.92 [0.85, 0.96] | 0.76 | 0.09 | 0.001 [0.000, 0.002] | 0.022 | 4901 |
| QAOA L-BFGS-B interp p=4 | 6 | 100 | 0.79 [0.70, 0.86] | 0.47 | 0.33 | 0.006 [0.003, 0.009] | 0.044 | 5131 |
| QAOA L-BFGS-B interp p=4 | 8 | 100 | 0.81 [0.72, 0.87] | 0.41 | 0.40 | 0.005 [0.002, 0.007] | 0.048 | 5266 |
| QAOA L-BFGS-B interp p=4 | 10 | 100 | 0.75 [0.66, 0.82] | 0.33 | 0.46 | 0.004 [0.003, 0.006] | 0.054 | 5535 |
| QAOA L-BFGS-B interp p=4 | 12 | 50 | 0.70 [0.56, 0.81] | 0.25 | 0.70 | 0.008 [0.002, 0.014] | 0.062 | 5587 |
| QAOA L-BFGS-B interp p=4 | 14 | 50 | 0.74 [0.60, 0.84] | 0.15 | 0.60 | 0.003 [0.001, 0.006] | 0.063 | 5620 |
| QAOA L-BFGS-B interp p=4 | 16 | 50 | 0.62 [0.48, 0.74] | 0.12 | 0.98 | 0.006 [0.003, 0.009] | 0.068 | 5560 |
| QAOA L-BFGS-B interp p=6 | 4 | 100 | 0.95 [0.89, 0.98] | 0.85 | 0.06 | 0.000 [0.000, 0.001] | 0.008 | 20623 |
| QAOA L-BFGS-B interp p=6 | 6 | 100 | 0.88 [0.80, 0.93] | 0.61 | 0.16 | 0.002 [0.000, 0.004] | 0.023 | 20134 |
| QAOA L-BFGS-B interp p=6 | 8 | 100 | 0.87 [0.79, 0.92] | 0.55 | 0.28 | 0.002 [0.001, 0.003] | 0.026 | 19092 |
| QAOA L-BFGS-B interp p=6 | 10 | 100 | 0.80 [0.71, 0.87] | 0.47 | 0.36 | 0.003 [0.001, 0.004] | 0.028 | 18514 |
| QAOA L-BFGS-B interp p=6 | 12 | 50 | 0.82 [0.69, 0.90] | 0.38 | 0.44 | 0.003 [0.000, 0.005] | 0.036 | 17918 |
| QAOA L-BFGS-B interp p=6 | 14 | 50 | 0.76 [0.63, 0.86] | 0.26 | 0.54 | 0.001 [0.001, 0.002] | 0.037 | 17198 |
| QAOA L-BFGS-B interp p=6 | 16 | 50 | 0.70 [0.56, 0.81] | 0.23 | 0.58 | 0.004 [0.001, 0.006] | 0.040 | 15799 |
| QAOA L-BFGS-B interp p=8 | 4 | 100 | 0.97 [0.92, 0.99] | 0.90 | 0.04 | 0.000 [-0.000, 0.000] | 0.003 | 58751 |
| QAOA L-BFGS-B interp p=8 | 6 | 100 | 0.89 [0.81, 0.94] | 0.69 | 0.15 | 0.002 [0.000, 0.003] | 0.013 | 54686 |
| QAOA L-BFGS-B interp p=8 | 8 | 100 | 0.90 [0.83, 0.94] | 0.65 | 0.25 | 0.001 [0.000, 0.003] | 0.015 | 49680 |
| QAOA L-BFGS-B interp p=8 | 10 | 100 | 0.88 [0.80, 0.93] | 0.57 | 0.22 | 0.001 [0.000, 0.002] | 0.017 | 47204 |
| QAOA L-BFGS-B interp p=8 | 12 | 50 | 0.86 [0.74, 0.93] | 0.50 | 0.30 | 0.002 [-0.000, 0.003] | 0.023 | 42564 |
| QAOA L-BFGS-B interp p=8 | 14 | 50 | 0.80 [0.67, 0.89] | 0.37 | 0.50 | 0.001 [0.000, 0.002] | 0.023 | 41416 |
| QAOA L-BFGS-B interp p=8 | 16 | 50 | 0.78 [0.65, 0.87] | 0.33 | 0.46 | 0.003 [0.000, 0.005] | 0.026 | 32329 |
| QAOA L-BFGS-B random p=2 | 4 | 100 | 0.84 [0.76, 0.90] | 0.54 | 0.18 | 0.006 [0.002, 0.009] | 0.074 | 760 |
| QAOA L-BFGS-B random p=2 | 6 | 100 | 0.59 [0.49, 0.68] | 0.27 | 0.52 | 0.020 [0.014, 0.026] | 0.107 | 816 |
| QAOA L-BFGS-B random p=2 | 8 | 100 | 0.59 [0.49, 0.68] | 0.20 | 0.61 | 0.016 [0.011, 0.021] | 0.115 | 869 |
| QAOA L-BFGS-B random p=2 | 10 | 100 | 0.59 [0.49, 0.68] | 0.13 | 0.86 | 0.013 [0.008, 0.018] | 0.125 | 916 |
| QAOA L-BFGS-B random p=2 | 12 | 50 | 0.52 [0.39, 0.65] | 0.08 | 1.08 | 0.017 [0.010, 0.024] | 0.133 | 945 |
| QAOA L-BFGS-B random p=2 | 14 | 50 | 0.38 [0.26, 0.52] | 0.05 | 1.42 | 0.023 [0.015, 0.030] | 0.132 | 994 |
| QAOA L-BFGS-B random p=2 | 16 | 50 | 0.30 [0.19, 0.44] | 0.03 | 1.76 | 0.021 [0.015, 0.027] | 0.144 | 986 |
| QAOA L-BFGS-B random p=4 | 4 | 100 | 0.90 [0.83, 0.94] | 0.77 | 0.11 | 0.001 [0.000, 0.003] | 0.021 | 4703 |
| QAOA L-BFGS-B random p=4 | 6 | 100 | 0.77 [0.68, 0.84] | 0.47 | 0.34 | 0.007 [0.004, 0.011] | 0.045 | 4719 |
| QAOA L-BFGS-B random p=4 | 8 | 100 | 0.81 [0.72, 0.87] | 0.41 | 0.37 | 0.005 [0.002, 0.007] | 0.048 | 4969 |
| QAOA L-BFGS-B random p=4 | 10 | 100 | 0.75 [0.66, 0.82] | 0.33 | 0.46 | 0.004 [0.003, 0.006] | 0.055 | 5184 |
| QAOA L-BFGS-B random p=4 | 12 | 50 | 0.70 [0.56, 0.81] | 0.24 | 0.70 | 0.008 [0.002, 0.014] | 0.063 | 5242 |
| QAOA L-BFGS-B random p=4 | 14 | 50 | 0.76 [0.63, 0.86] | 0.15 | 0.40 | 0.002 [0.001, 0.003] | 0.065 | 5430 |
| QAOA L-BFGS-B random p=4 | 16 | 50 | 0.64 [0.50, 0.76] | 0.11 | 1.02 | 0.006 [0.003, 0.010] | 0.072 | 5500 |
| QAOA L-BFGS-B random p=6 | 4 | 100 | 0.95 [0.89, 0.98] | 0.86 | 0.05 | 0.000 [-0.000, 0.001] | 0.007 | 20781 |
| QAOA L-BFGS-B random p=6 | 6 | 100 | 0.87 [0.79, 0.92] | 0.61 | 0.17 | 0.002 [0.000, 0.004] | 0.023 | 20753 |
| QAOA L-BFGS-B random p=6 | 8 | 100 | 0.88 [0.80, 0.93] | 0.55 | 0.27 | 0.002 [0.000, 0.003] | 0.026 | 23321 |
| QAOA L-BFGS-B random p=6 | 10 | 100 | 0.78 [0.69, 0.85] | 0.47 | 0.39 | 0.003 [0.002, 0.005] | 0.029 | 24647 |
| QAOA L-BFGS-B random p=6 | 12 | 50 | 0.80 [0.67, 0.89] | 0.37 | 0.34 | 0.002 [0.000, 0.004] | 0.037 | 25159 |
| QAOA L-BFGS-B random p=6 | 14 | 50 | 0.74 [0.60, 0.84] | 0.25 | 0.58 | 0.002 [0.001, 0.003] | 0.039 | 25679 |
| QAOA L-BFGS-B random p=6 | 16 | 50 | 0.66 [0.52, 0.78] | 0.22 | 0.70 | 0.004 [0.002, 0.006] | 0.043 | 25661 |
| QAOA L-BFGS-B random p=8 | 4 | 100 | 0.96 [0.90, 0.98] | 0.92 | 0.05 | 0.000 [-0.000, 0.001] | 0.003 | 43477 |
| QAOA L-BFGS-B random p=8 | 6 | 100 | 0.89 [0.81, 0.94] | 0.69 | 0.14 | 0.001 [0.000, 0.002] | 0.014 | 44389 |
| QAOA L-BFGS-B random p=8 | 8 | 100 | 0.90 [0.83, 0.94] | 0.63 | 0.22 | 0.001 [0.000, 0.002] | 0.017 | 44393 |
| QAOA L-BFGS-B random p=8 | 10 | 100 | 0.87 [0.79, 0.92] | 0.55 | 0.23 | 0.001 [0.001, 0.002] | 0.019 | 45330 |
| QAOA L-BFGS-B random p=8 | 12 | 50 | 0.86 [0.74, 0.93] | 0.47 | 0.20 | 0.001 [-0.000, 0.002] | 0.026 | 45458 |
| QAOA L-BFGS-B random p=8 | 14 | 50 | 0.78 [0.65, 0.87] | 0.34 | 0.52 | 0.001 [0.000, 0.002] | 0.026 | 45940 |
| QAOA L-BFGS-B random p=8 | 16 | 50 | 0.74 [0.60, 0.84] | 0.30 | 0.54 | 0.004 [0.001, 0.006] | 0.030 | 45989 |
| VQPM (max_prob, T=30) | 4 | 100 | 0.77 [0.68, 0.84] | 0.53 | 0.25 | 0.015 [0.006, 0.024] | 0.059 | 2 |
| VQPM (max_prob, T=30) | 6 | 100 | 0.61 [0.51, 0.70] | 0.43 | 0.47 | 0.020 [0.014, 0.027] | 0.039 | 5 |
| VQPM (max_prob, T=30) | 8 | 100 | 0.59 [0.49, 0.68] | 0.43 | 0.61 | 0.019 [0.012, 0.025] | 0.033 | 7 |
| VQPM (max_prob, T=30) | 10 | 100 | 0.64 [0.54, 0.73] | 0.42 | 0.54 | 0.010 [0.007, 0.014] | 0.023 | 10 |
| VQPM (max_prob, T=30) | 12 | 50 | 0.52 [0.39, 0.65] | 0.39 | 1.08 | 0.016 [0.008, 0.023] | 0.025 | 14 |
| VQPM (max_prob, T=30) | 14 | 50 | 0.46 [0.33, 0.60] | 0.29 | 0.98 | 0.016 [0.009, 0.023] | 0.024 | 19 |
| VQPM (max_prob, T=30) | 16 | 50 | 0.40 [0.28, 0.54] | 0.28 | 1.38 | 0.017 [0.011, 0.023] | 0.022 | 22 |
| VQPM (max_prob, T=100) | 4 | 100 | 0.77 [0.68, 0.84] | 0.53 | 0.25 | 0.015 [0.006, 0.024] | 0.059 | 3 |
| VQPM (max_prob, T=100) | 6 | 100 | 0.61 [0.51, 0.70] | 0.43 | 0.47 | 0.020 [0.014, 0.027] | 0.039 | 5 |
| VQPM (max_prob, T=100) | 8 | 100 | 0.59 [0.49, 0.68] | 0.43 | 0.61 | 0.019 [0.012, 0.025] | 0.033 | 9 |
| VQPM (max_prob, T=100) | 10 | 100 | 0.64 [0.54, 0.73] | 0.42 | 0.54 | 0.010 [0.007, 0.014] | 0.023 | 14 |
| VQPM (max_prob, T=100) | 12 | 50 | 0.52 [0.39, 0.65] | 0.39 | 1.08 | 0.016 [0.008, 0.023] | 0.025 | 26 |
| VQPM (max_prob, T=100) | 14 | 50 | 0.46 [0.33, 0.60] | 0.29 | 0.98 | 0.016 [0.009, 0.023] | 0.024 | 42 |
| VQPM (max_prob, T=100) | 16 | 50 | 0.42 [0.29, 0.56] | 0.29 | 1.36 | 0.016 [0.010, 0.021] | 0.021 | 50 |
| VQPM (oracle, T=30) | 4 | 100 | 0.80 [0.71, 0.87] | 0.53 | 0.22 | 0.013 [0.004, 0.022] | 0.060 | 3 |
| VQPM (oracle, T=30) | 6 | 100 | 0.74 [0.65, 0.82] | 0.43 | 0.33 | 0.015 [0.008, 0.022] | 0.052 | 6 |
| VQPM (oracle, T=30) | 8 | 100 | 0.69 [0.59, 0.77] | 0.43 | 0.49 | 0.016 [0.009, 0.023] | 0.048 | 8 |
| VQPM (oracle, T=30) | 10 | 100 | 0.69 [0.59, 0.77] | 0.42 | 0.58 | 0.014 [0.008, 0.021] | 0.046 | 10 |
| VQPM (oracle, T=30) | 12 | 50 | 0.62 [0.48, 0.74] | 0.39 | 0.84 | 0.023 [0.010, 0.036] | 0.062 | 12 |
| VQPM (oracle, T=30) | 14 | 50 | 0.50 [0.37, 0.63] | 0.29 | 0.98 | 0.019 [0.009, 0.028] | 0.051 | 15 |
| VQPM (oracle, T=30) | 16 | 50 | 0.44 [0.31, 0.58] | 0.28 | 1.44 | 0.024 [0.014, 0.034] | 0.063 | 16 |
| VQPM (oracle, T=100) | 4 | 100 | 0.80 [0.71, 0.87] | 0.53 | 0.22 | 0.013 [0.004, 0.022] | 0.060 | 6 |
| VQPM (oracle, T=100) | 6 | 100 | 0.74 [0.65, 0.82] | 0.43 | 0.33 | 0.015 [0.008, 0.022] | 0.052 | 8 |
| VQPM (oracle, T=100) | 8 | 100 | 0.69 [0.59, 0.77] | 0.43 | 0.49 | 0.016 [0.009, 0.023] | 0.048 | 15 |
| VQPM (oracle, T=100) | 10 | 100 | 0.69 [0.59, 0.77] | 0.42 | 0.58 | 0.014 [0.008, 0.021] | 0.046 | 19 |
| VQPM (oracle, T=100) | 12 | 50 | 0.62 [0.48, 0.74] | 0.39 | 0.84 | 0.023 [0.010, 0.036] | 0.062 | 23 |
| VQPM (oracle, T=100) | 14 | 50 | 0.50 [0.37, 0.63] | 0.29 | 0.98 | 0.019 [0.009, 0.028] | 0.051 | 27 |
| VQPM (oracle, T=100) | 16 | 50 | 0.46 [0.33, 0.60] | 0.29 | 1.42 | 0.023 [0.013, 0.033] | 0.063 | 33 |

## Findings and caveats

Written 2026-11-27 from the finished run (100 instances for n = 4..10, 50 for n = 12..16; seed 2026).
Numbers below are for QAOA depth p = 8 unless stated; "found optimum" = the most probable bitstring of
the final QAOA state (VQPM: the returned state) is the exact ground state.

- **QAOA with a good optimizer finds the optimum more often than VQPM at every n, at about 1000x more
  evaluations.** Found-optimum rate n = 4 / 8 / 12 / 16:
  L-BFGS-B random restarts 0.96 / 0.90 / 0.86 / 0.74; L-BFGS-B INTERP 0.97 / 0.90 / 0.86 / 0.78;
  Adam random 0.93 / 0.92 / 0.86 / 0.68; Adam INTERP 0.93 / 0.91 / 0.86 / 0.76;
  VQPM (precision 3, oracle, T = 100) 0.80 / 0.69 / 0.62 / 0.46; VQPM max_prob 0.77 / 0.59 / 0.52 / 0.42.
  Cost per instance: L-BFGS-B ~3e4-6e4 finite-difference-equivalent energy evaluations (5 restarts),
  Adam ~4e4 (random) to ~1.3e5 (INTERP); VQPM 3-33 iterations (one circuit each, but each needs
  ~1e6 shots for 3-decimal marginals, see finite_shots.md).
- **QAOA depends strongly on the optimizer.** At n = 16, p = 8 the found-optimum rate ranges from 0.40
  (gradient descent with INTERP) to 0.78 (L-BFGS-B with INTERP); with random restarts COBYLA reaches
  0.48-0.62 and plain gradient descent 0.52-0.64 for n = 10..16, while L-BFGS-B / Adam reach 0.68-0.87.
  INTERP helps COBYLA a lot (n = 12: 0.62 -> 0.78) but hurts plain gradient descent at large n
  (n = 16: 0.58 random vs 0.40 INTERP). Other depths: qaoa_vs_vqpm_metrics_all_optimizers_p{2,4,6}.png
  (not summarised here).
- **The paper's "VQPM comparable to QAOA" holds only against a weakly optimised QAOA** (COBYLA or gradient
  descent with random starts, which are at VQPM's level for n >= 10); against L-BFGS-B or Adam with
  restarts or INTERP it does not.
- **Probability of the optimum in the final state** declines with n for both: QAOA L-BFGS-B random 0.92 /
  0.63 / 0.47 / 0.30 (n = 4 / 8 / 12 / 16) vs VQPM 0.53 / 0.43 / 0.39 / 0.29, i.e. similar at n = 16.
- **Energy:** the returned states are near-optimal for every method. Mean normalised energy gap of the
  returned state: QAOA L-BFGS-B / Adam 0.000-0.005, COBYLA / gd random 0.003-0.017, VQPM 0.010-0.024.
  Mean normalised expected-energy gap of the final distribution: QAOA L-BFGS-B 0.003-0.030, VQPM
  0.046-0.063 (oracle) / 0.021-0.059 (max_prob).
- **Simulated annealing** solves all these instances: 100 sweeps (n x 100 single-flip proposals) give
  0.98-1.00, 1000 sweeps 1.00, and even 10 sweeps 0.70-0.96.
- Caveats: QAOA expectations are exact (no shot noise) and evaluation counts do not include shots; QAOA
  runs on the paper's scaled QUBO (energies in [-pi/4, pi/4]) with random initial angles in [0, pi);
  learning rates were tuned on separate instances at n = 8 only; VQPM uses the published setting
  (precision 3, pdiff 0.01) -- with precision 6 it would reach ~0.57 at n = 16 (vqpm_grid.md) but needs
  ~1e12 shots per iteration; random Gaussian QUBOs only.

## Regenerate tables and plots from the saved data (no re-run)

```bash
.venv/bin/python experiments/qaoa_vs_vqpm.py --plot-only --out experiments/results/qaoa_vs_vqpm/raw/qaoa_vs_vqpm.jsonl
.venv/bin/python experiments/reports.py
```
