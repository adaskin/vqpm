# Exp. 4: finite shots

_Report generated 2026-11-27T07:51:14 by `experiments/reports.py`; status: finished._

## Purpose

VQPM with marginals estimated from samples; fixed pdiff vs a valid union-bound Hoeffding locking rule; total shot cost.

## Method

Each iteration, M ~ Binomial(shots, P0) samples survive post-selection of the control on 0 and are drawn from p(x) cos^2(phase_x/2)/P0; qubit marginals are the sample frequencies (rounded to 3 decimals). Locking: fixed pdiff = 0.01, or hoeffding_union (lock only if |P0 - P1| > 2 eps + 10^-3, eps = sqrt(ln(2nT/delta)/(2M)), delta = 0.05, T = 100). n = 8..16 step 2, 100 instances, shots per iteration 1e3..1e6, both stop rules, max_iter 100. Returned state = most frequent sample of the last iteration (also reported: locked bits + marginal majority, and the exact argmax).

Metrics (per run, energies in the units of the original, unscaled Q): found optimum = the returned bitstring is the exact ground state (found by enumerating all 2^n energies); bit difference = Hamming distance to the ground state; energy gap = E_found - E_min; normalised energy gap = (E_found - E_min)/(E_max - E_min) in [0, 1]; normalised expected-energy gap = (<E> - E_min)/(E_max - E_min) with <E> the expected energy of the final output distribution (VQPM, QAOA). Intervals: Wilson 95% for rates, mean +- 1.96 s/sqrt(N) for means.

## Provenance

- Invocation 1 of `experiments/results/finite_shots/raw/finite_shots.jsonl`: started 2026-11-25T20:18:13, git commit `61a1e1fe0662bf8c454b52a2427ba08b6a976cf9` (dirty tree), python 3.12.12, numpy 2.5.5, scipy 1.18.1
  - command: `.venv/bin/python experiments/finite_shots.py --n 8 10 12 14 16 --instances 100 --workers 10 --out experiments/results/finite_shots/raw/finite_shots.jsonl`
  - parameters: `{"delta": 0.05, "inst_dir": "experiments/instances", "instances": 100, "max_iter": 100, "n": [8, 10, 12, 14, 16], "out": "experiments/results/finite_shots/raw/finite_shots.jsonl", "pdiff": 0.01, "plot_only": false, "policies": ["fixed", "hoeffding_union"], "precision": 3, "seed": 2026, "shots": [1000, 10000, 100000, 1000000], "workers": 10}`
- Library versions (pinned in requirements.txt): matplotlib 3.11.2, numpy 2.5.5, pytest 9.2.0, qiskit 2.6.2, qiskit-aer 0.18.4, qiskit-ibm-runtime 0.52.1, scipy 1.18.1, python 3.12.12
- Machine: AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 16 logical CPUs, 90 GiB RAM, Linux 6.12.113+deb13-amd64
- End time (last write of the raw data): `finite_shots.jsonl` 2026-11-25T20:19:18
- Wall time `finite_shots.jsonl` (first start to last write, includes waits between invocations): 0.02 h
- Instances: seed 2026, files `experiments/instances/qubo_n<n>_seed2026.npz` (instance i is `default_rng([2026, n, i])`, Q = A + A^T, A_ij ~ N(0, 1); each file stores Q, optimum, E_min, E_max, gap); sha256 listed below

## Where the data is

Paths are relative to the repository root.

| file (repo-relative) | records / size | sha256 |
|---|---|---|
| `experiments/results/finite_shots/raw/finite_shots.jsonl` | 8000 records, 17.8 MiB | `5fc89e03d6ab4363b4c68cfb0d151881255a5dfbca8788543804ba190ae868c9` |
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

- `experiments/results/finite_shots/finite_shots_max_prob.pdf`
- `experiments/results/finite_shots/finite_shots_max_prob.png`
- `experiments/results/finite_shots/finite_shots_metrics_fixed_max_prob.pdf`
- `experiments/results/finite_shots/finite_shots_metrics_fixed_max_prob.png`
- `experiments/results/finite_shots/finite_shots_metrics_fixed_oracle.pdf`
- `experiments/results/finite_shots/finite_shots_metrics_fixed_oracle.png`
- `experiments/results/finite_shots/finite_shots_metrics_hoeffding_union_max_prob.pdf`
- `experiments/results/finite_shots/finite_shots_metrics_hoeffding_union_max_prob.png`
- `experiments/results/finite_shots/finite_shots_metrics_hoeffding_union_oracle.pdf`
- `experiments/results/finite_shots/finite_shots_metrics_hoeffding_union_oracle.png`
- `experiments/results/finite_shots/finite_shots_oracle.pdf`
- `experiments/results/finite_shots/finite_shots_oracle.png`
- `experiments/results/finite_shots/finite_shots_runs.csv` (not committed: large; regenerate)
- `experiments/results/finite_shots/finite_shots_summary.csv`

Raw JSONL files are not in the repository because of their size. The scripts write them to `experiments/results/<experiment>/raw/`, and the summary tables above are computed from them.

## Results

| policy | rule | shots/iter | n | found-opt [CI] | bit diff | norm. energy gap [CI] | runs w/ wrong lock | iters | total shots |
|---|---|---|---|---|---|---|---|---|---|
| fixed | max_prob | 1000 | 8 | 0.02 [0.01, 0.07] | 3.20 | 0.362 [0.322, 0.402] | 0.95 | 2.2 | 2200 |
| fixed | max_prob | 1000 | 10 | 0.00 [0.00, 0.04] | 4.26 | 0.389 [0.353, 0.426] | 0.99 | 2.4 | 2430 |
| fixed | max_prob | 1000 | 12 | 0.00 [0.00, 0.04] | 5.30 | 0.418 [0.386, 0.451] | 1.00 | 2.4 | 2440 |
| fixed | max_prob | 1000 | 14 | 0.00 [0.00, 0.04] | 6.52 | 0.430 [0.401, 0.458] | 1.00 | 2.7 | 2700 |
| fixed | max_prob | 1000 | 16 | 0.00 [0.00, 0.04] | 7.83 | 0.448 [0.423, 0.473] | 1.00 | 2.8 | 2790 |
| fixed | max_prob | 10000 | 8 | 0.11 [0.06, 0.19] | 1.98 | 0.128 [0.106, 0.150] | 0.85 | 3.9 | 38500 |
| fixed | max_prob | 10000 | 10 | 0.07 [0.03, 0.14] | 2.45 | 0.156 [0.133, 0.180] | 0.91 | 4.7 | 46700 |
| fixed | max_prob | 10000 | 12 | 0.01 [0.00, 0.05] | 3.60 | 0.193 [0.169, 0.216] | 0.98 | 5.3 | 53200 |
| fixed | max_prob | 10000 | 14 | 0.00 [0.00, 0.04] | 4.28 | 0.221 [0.200, 0.243] | 1.00 | 5.7 | 57300 |
| fixed | max_prob | 10000 | 16 | 0.00 [0.00, 0.04] | 5.43 | 0.269 [0.244, 0.295] | 0.99 | 5.9 | 58900 |
| fixed | max_prob | 100000 | 8 | 0.52 [0.42, 0.62] | 0.77 | 0.023 [0.016, 0.029] | 0.38 | 5.8 | 585000 |
| fixed | max_prob | 100000 | 10 | 0.52 [0.42, 0.62] | 0.82 | 0.022 [0.016, 0.029] | 0.41 | 8.4 | 836000 |
| fixed | max_prob | 100000 | 12 | 0.30 [0.22, 0.40] | 1.49 | 0.034 [0.026, 0.041] | 0.62 | 10.8 | 1079000 |
| fixed | max_prob | 100000 | 14 | 0.26 [0.18, 0.35] | 1.74 | 0.037 [0.028, 0.045] | 0.67 | 12.8 | 1275000 |
| fixed | max_prob | 100000 | 16 | 0.20 [0.13, 0.29] | 2.17 | 0.046 [0.037, 0.056] | 0.73 | 15.3 | 1527000 |
| fixed | max_prob | 1000000 | 8 | 0.57 [0.47, 0.66] | 0.64 | 0.017 [0.012, 0.023] | 0.22 | 6.4 | 6430000 |
| fixed | max_prob | 1000000 | 10 | 0.60 [0.50, 0.69] | 0.59 | 0.011 [0.007, 0.015] | 0.26 | 9.3 | 9350000 |
| fixed | max_prob | 1000000 | 12 | 0.46 [0.37, 0.56] | 1.10 | 0.017 [0.012, 0.023] | 0.39 | 13.8 | 13770000 |
| fixed | max_prob | 1000000 | 14 | 0.48 [0.38, 0.58] | 1.05 | 0.013 [0.009, 0.017] | 0.36 | 18.9 | 18900000 |
| fixed | max_prob | 1000000 | 16 | 0.37 [0.28, 0.47] | 1.38 | 0.017 [0.012, 0.021] | 0.46 | 23.4 | 23370000 |
| fixed | oracle | 1000 | 8 | 0.02 [0.01, 0.07] | 3.25 | 0.361 [0.324, 0.399] | 0.97 | 2.1 | 2080 |
| fixed | oracle | 1000 | 10 | 0.00 [0.00, 0.04] | 4.31 | 0.396 [0.357, 0.435] | 0.99 | 2.0 | 2020 |
| fixed | oracle | 1000 | 12 | 0.00 [0.00, 0.04] | 5.43 | 0.439 [0.404, 0.474] | 1.00 | 2.0 | 2030 |
| fixed | oracle | 1000 | 14 | 0.00 [0.00, 0.04] | 6.56 | 0.439 [0.409, 0.469] | 1.00 | 2.0 | 2010 |
| fixed | oracle | 1000 | 16 | 0.00 [0.00, 0.04] | 8.01 | 0.473 [0.445, 0.501] | 1.00 | 2.0 | 2000 |
| fixed | oracle | 10000 | 8 | 0.12 [0.07, 0.20] | 2.18 | 0.188 [0.155, 0.220] | 0.86 | 2.9 | 28700 |
| fixed | oracle | 10000 | 10 | 0.07 [0.03, 0.14] | 3.06 | 0.247 [0.208, 0.285] | 0.91 | 2.7 | 26800 |
| fixed | oracle | 10000 | 12 | 0.01 [0.00, 0.05] | 4.63 | 0.309 [0.277, 0.340] | 0.99 | 2.4 | 24200 |
| fixed | oracle | 10000 | 14 | 0.00 [0.00, 0.04] | 5.48 | 0.335 [0.307, 0.362] | 1.00 | 2.4 | 24100 |
| fixed | oracle | 10000 | 16 | 0.00 [0.00, 0.04] | 6.61 | 0.378 [0.348, 0.407] | 0.99 | 2.2 | 22400 |
| fixed | oracle | 100000 | 8 | 0.54 [0.44, 0.63] | 0.81 | 0.029 [0.019, 0.040] | 0.41 | 5.1 | 514000 |
| fixed | oracle | 100000 | 10 | 0.55 [0.45, 0.64] | 1.00 | 0.044 [0.028, 0.060] | 0.41 | 6.8 | 677000 |
| fixed | oracle | 100000 | 12 | 0.34 [0.25, 0.44] | 1.81 | 0.068 [0.052, 0.085] | 0.62 | 6.4 | 636000 |
| fixed | oracle | 100000 | 14 | 0.28 [0.20, 0.37] | 2.63 | 0.101 [0.078, 0.124] | 0.70 | 8.3 | 834000 |
| fixed | oracle | 100000 | 16 | 0.24 [0.17, 0.33] | 3.33 | 0.129 [0.102, 0.155] | 0.75 | 8.2 | 821000 |
| fixed | oracle | 1000000 | 8 | 0.75 [0.66, 0.82] | 0.44 | 0.012 [0.007, 0.017] | 0.22 | 6.7 | 6680000 |
| fixed | oracle | 1000000 | 10 | 0.71 [0.61, 0.79] | 0.56 | 0.017 [0.010, 0.023] | 0.27 | 9.4 | 9400000 |
| fixed | oracle | 1000000 | 12 | 0.53 [0.43, 0.62] | 1.20 | 0.037 [0.024, 0.051] | 0.39 | 10.0 | 10040000 |
| fixed | oracle | 1000000 | 14 | 0.57 [0.47, 0.66] | 1.09 | 0.026 [0.016, 0.036] | 0.36 | 15.1 | 15050000 |
| fixed | oracle | 1000000 | 16 | 0.49 [0.39, 0.59] | 1.91 | 0.056 [0.039, 0.073] | 0.48 | 15.6 | 15620000 |
| hoeffding_union | max_prob | 1000 | 8 | 0.15 [0.09, 0.23] | 1.87 | 0.117 [0.095, 0.138] | 0.74 | 30.9 | 30940 |
| hoeffding_union | max_prob | 1000 | 10 | 0.12 [0.07, 0.20] | 2.36 | 0.148 [0.127, 0.169] | 0.82 | 37.8 | 37810 |
| hoeffding_union | max_prob | 1000 | 12 | 0.03 [0.01, 0.08] | 3.32 | 0.204 [0.177, 0.231] | 0.97 | 45.4 | 45390 |
| hoeffding_union | max_prob | 1000 | 14 | 0.00 [0.00, 0.04] | 4.75 | 0.231 [0.209, 0.253] | 0.99 | 51.0 | 51020 |
| hoeffding_union | max_prob | 1000 | 16 | 0.01 [0.00, 0.05] | 6.01 | 0.285 [0.262, 0.307] | 0.99 | 53.1 | 53090 |
| hoeffding_union | max_prob | 10000 | 8 | 0.40 [0.31, 0.50] | 0.93 | 0.038 [0.028, 0.047] | 0.44 | 18.0 | 180100 |
| hoeffding_union | max_prob | 10000 | 10 | 0.33 [0.25, 0.43] | 1.25 | 0.048 [0.037, 0.058] | 0.52 | 23.8 | 237800 |
| hoeffding_union | max_prob | 10000 | 12 | 0.13 [0.08, 0.21] | 2.06 | 0.067 [0.056, 0.079] | 0.75 | 29.6 | 296300 |
| hoeffding_union | max_prob | 10000 | 14 | 0.07 [0.03, 0.14] | 2.48 | 0.070 [0.058, 0.082] | 0.83 | 35.6 | 355800 |
| hoeffding_union | max_prob | 10000 | 16 | 0.08 [0.04, 0.15] | 3.26 | 0.099 [0.086, 0.112] | 0.90 | 42.0 | 420500 |
| hoeffding_union | max_prob | 100000 | 8 | 0.55 [0.45, 0.64] | 0.74 | 0.019 [0.014, 0.025] | 0.27 | 8.9 | 894000 |
| hoeffding_union | max_prob | 100000 | 10 | 0.49 [0.39, 0.59] | 0.83 | 0.023 [0.014, 0.031] | 0.40 | 12.1 | 1205000 |
| hoeffding_union | max_prob | 100000 | 12 | 0.36 [0.27, 0.46] | 1.27 | 0.026 [0.019, 0.033] | 0.46 | 17.5 | 1752000 |
| hoeffding_union | max_prob | 100000 | 14 | 0.27 [0.19, 0.36] | 1.44 | 0.023 [0.018, 0.029] | 0.60 | 22.1 | 2215000 |
| hoeffding_union | max_prob | 100000 | 16 | 0.21 [0.14, 0.30] | 2.21 | 0.035 [0.029, 0.042] | 0.74 | 26.3 | 2632000 |
| hoeffding_union | max_prob | 1000000 | 8 | 0.52 [0.42, 0.62] | 0.72 | 0.023 [0.016, 0.030] | 0.43 | 3.8 | 3810000 |
| hoeffding_union | max_prob | 1000000 | 10 | 0.62 [0.52, 0.71] | 0.59 | 0.017 [0.010, 0.024] | 0.32 | 4.9 | 4880000 |
| hoeffding_union | max_prob | 1000000 | 12 | 0.47 [0.38, 0.57] | 1.11 | 0.021 [0.015, 0.027] | 0.45 | 6.8 | 6760000 |
| hoeffding_union | max_prob | 1000000 | 14 | 0.34 [0.25, 0.44] | 1.19 | 0.019 [0.015, 0.024] | 0.57 | 9.6 | 9560000 |
| hoeffding_union | max_prob | 1000000 | 16 | 0.36 [0.27, 0.46] | 1.44 | 0.017 [0.013, 0.021] | 0.52 | 11.8 | 11830000 |
| hoeffding_union | oracle | 1000 | 8 | 0.20 [0.13, 0.29] | 2.36 | 0.226 [0.183, 0.269] | 0.76 | 18.2 | 18230 |
| hoeffding_union | oracle | 1000 | 10 | 0.13 [0.08, 0.21] | 3.54 | 0.313 [0.270, 0.356] | 0.84 | 17.6 | 17590 |
| hoeffding_union | oracle | 1000 | 12 | 0.03 [0.01, 0.08] | 4.77 | 0.357 [0.322, 0.392] | 0.97 | 14.6 | 14610 |
| hoeffding_union | oracle | 1000 | 14 | 0.00 [0.00, 0.04] | 6.37 | 0.401 [0.370, 0.432] | 0.99 | 11.5 | 11490 |
| hoeffding_union | oracle | 1000 | 16 | 0.01 [0.00, 0.05] | 7.28 | 0.434 [0.403, 0.466] | 0.99 | 10.6 | 10570 |
| hoeffding_union | oracle | 10000 | 8 | 0.47 [0.38, 0.57] | 0.95 | 0.054 [0.039, 0.068] | 0.45 | 15.5 | 154700 |
| hoeffding_union | oracle | 10000 | 10 | 0.40 [0.31, 0.50] | 1.66 | 0.108 [0.080, 0.135] | 0.55 | 16.8 | 167600 |
| hoeffding_union | oracle | 10000 | 12 | 0.16 [0.10, 0.24] | 3.02 | 0.159 [0.129, 0.189] | 0.81 | 16.0 | 159800 |
| hoeffding_union | oracle | 10000 | 14 | 0.13 [0.08, 0.21] | 3.68 | 0.186 [0.153, 0.220] | 0.85 | 17.5 | 174700 |
| hoeffding_union | oracle | 10000 | 16 | 0.08 [0.04, 0.15] | 5.73 | 0.306 [0.268, 0.343] | 0.90 | 14.9 | 148600 |
| hoeffding_union | oracle | 100000 | 8 | 0.65 [0.55, 0.74] | 0.69 | 0.027 [0.016, 0.038] | 0.32 | 8.3 | 833000 |
| hoeffding_union | oracle | 100000 | 10 | 0.56 [0.46, 0.65] | 0.94 | 0.043 [0.028, 0.058] | 0.40 | 9.7 | 974000 |
| hoeffding_union | oracle | 100000 | 12 | 0.43 [0.34, 0.53] | 1.57 | 0.052 [0.037, 0.067] | 0.49 | 12.8 | 1282000 |
| hoeffding_union | oracle | 100000 | 14 | 0.33 [0.25, 0.43] | 1.86 | 0.064 [0.046, 0.081] | 0.64 | 14.2 | 1419000 |
| hoeffding_union | oracle | 100000 | 16 | 0.22 [0.15, 0.31] | 3.38 | 0.124 [0.099, 0.149] | 0.75 | 11.8 | 1176000 |
| hoeffding_union | oracle | 1000000 | 8 | 0.55 [0.45, 0.64] | 0.74 | 0.024 [0.017, 0.031] | 0.44 | 3.7 | 3680000 |
| hoeffding_union | oracle | 1000000 | 10 | 0.66 [0.56, 0.75] | 0.66 | 0.021 [0.012, 0.030] | 0.32 | 4.6 | 4640000 |
| hoeffding_union | oracle | 1000000 | 12 | 0.49 [0.39, 0.59] | 1.20 | 0.042 [0.028, 0.056] | 0.49 | 5.5 | 5510000 |
| hoeffding_union | oracle | 1000000 | 14 | 0.37 [0.28, 0.47] | 1.37 | 0.041 [0.029, 0.053] | 0.58 | 6.9 | 6930000 |
| hoeffding_union | oracle | 1000000 | 16 | 0.44 [0.35, 0.54] | 1.80 | 0.049 [0.035, 0.063] | 0.53 | 8.3 | 8270000 |

## Findings and caveats

Written 2026-11-27 (finished; 100 instances per n, precision 3, max_iter 100).

- Found-optimum rate (oracle rule) with marginals estimated from samples, fixed pdiff 0.01,
  n = 8 / 12 / 16: 1e3 shots per iteration 0.02 / 0.00 / 0.00; 1e4: 0.12 / 0.01 / 0.00;
  1e5: 0.54 / 0.34 / 0.24; 1e6: 0.75 / 0.53 / 0.49 (exact marginals, same instances: about
  0.69 / 0.59 / 0.38, vqpm_grid.md). Total shots per run at 1e6 per iteration: 0.7e7-1.6e7.
- With fixed pdiff, too few shots make qubits lock on sampling noise in the first 2-3 iterations:
  86-100% of runs have a wrong lock at 1e3-1e4 shots.
- The union-bound Hoeffding rule (delta 0.05) avoids locking on noise and helps at small budgets
  (1e4 shots: n = 8 0.47 vs 0.12; n = 12 0.16 vs 0.01), but it does not remove wrong locks: at 1e6
  shots 49-58% of runs at n = 12-16 still have one, because the bound guarantees the sign of the
  *current* marginal, not agreement with the optimum (the product-state dynamics can favour the wrong
  value). At 1e6 shots it is comparable to fixed pdiff (0.49 vs 0.53 at n = 12, 0.44 vs 0.49 at n = 16)
  with fewer iterations (6-8 vs 10-16) and about half the total shots.
- So VQPM needs ~1e5-1e6 shots per iteration (1e6-1e7 per run) at n = 8-16 to approach its
  exact-marginal performance.
- Caveats: returned state = most frequent sample of the last iteration; no circuit noise; the oracle
  stop uses the exact target probability.

## Regenerate tables and plots from the saved data (no re-run)

```bash
.venv/bin/python experiments/finite_shots.py --plot-only --out experiments/results/finite_shots/raw/finite_shots.jsonl
.venv/bin/python experiments/reports.py
```
