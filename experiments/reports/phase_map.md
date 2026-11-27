# Exp. 3: phase-map ablation

_Report generated 2026-11-27T07:51:13 by `experiments/reports.py`; status: finished._

## Purpose

Sensitivity to the map from QUBO values to eigenphases.

## Method

Maps (vqpm.phase_map_coefficients): paper E*pi/(4S)+pi/4 with S = sum|triu(Q)|; tight (E-L)/(U-L)*pi/2 with L, U the sums of the negative / positive entries of triu(Q); tight_wide (E-L)/(U-L)*0.9*pi; shifted E*pi/(4S)+3pi/4. n = 8..16 step 2, 100 instances, precision 3, pdiff 0.01, both stop rules, budgets 30 and 100 (one trajectory per run, read off as in Exp. 1).

Metrics (per run, energies in the units of the original, unscaled Q): found optimum = the returned bitstring is the exact ground state (found by enumerating all 2^n energies); bit difference = Hamming distance to the ground state; energy gap = E_found - E_min; normalised energy gap = (E_found - E_min)/(E_max - E_min) in [0, 1]; normalised expected-energy gap = (<E> - E_min)/(E_max - E_min) with <E> the expected energy of the final output distribution (VQPM, QAOA). Intervals: Wilson 95% for rates, mean +- 1.96 s/sqrt(N) for means.

## Provenance

- Invocation 1 of `experiments/results/phase_map/raw/phase_map.jsonl`: started 2026-11-25T20:17:37, git commit `61a1e1fe0662bf8c454b52a2427ba08b6a976cf9` (dirty tree), python 3.12.12, numpy 2.5.5, scipy 1.18.1
  - command: `.venv/bin/python experiments/phase_map_ablation.py --n 8 10 12 14 16 --instances 100 --workers 10 --out experiments/results/phase_map/raw/phase_map.jsonl`
  - parameters: `{"budgets": ["30", "100"], "inst_dir": "experiments/instances", "instances": 100, "n": [8, 10, 12, 14, 16], "out": "experiments/results/phase_map/raw/phase_map.jsonl", "pdiff": 0.01, "phase_maps": ["paper", "tight", "tight_wide", "shifted"], "plot_only": false, "precision": "3", "seed": 2026, "workers": 10}`
- Library versions (pinned in requirements.txt): matplotlib 3.11.2, numpy 2.5.5, pytest 9.2.0, qiskit 2.6.2, qiskit-aer 0.18.4, qiskit-ibm-runtime 0.52.1, scipy 1.18.1, python 3.12.12
- Machine: AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 16 logical CPUs, 90 GiB RAM, Linux 6.12.113+deb13-amd64
- End time (last write of the raw data): `phase_map.jsonl` 2026-11-25T20:18:06
- Wall time `phase_map.jsonl` (first start to last write, includes waits between invocations): 0.01 h
- Instances: seed 2026, files `experiments/instances/qubo_n<n>_seed2026.npz` (instance i is `default_rng([2026, n, i])`, Q = A + A^T, A_ij ~ N(0, 1); each file stores Q, optimum, E_min, E_max, gap); sha256 listed below

## Where the data is

Paths are relative to the repository root.

| file (repo-relative) | records / size | sha256 |
|---|---|---|
| `experiments/results/phase_map/raw/phase_map.jsonl` | 2000 records, 10.3 MiB | `9044cb44f6ef7712d7b247b40031e50f63bfaea2b6f405645442edfc5e3ea8fd` |
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

- `experiments/results/phase_map/phase_map_max_iter100.pdf`
- `experiments/results/phase_map/phase_map_max_iter100.png`
- `experiments/results/phase_map/phase_map_max_iter30.pdf`
- `experiments/results/phase_map/phase_map_max_iter30.png`
- `experiments/results/phase_map/phase_map_metrics_max_prob_max_iter100.pdf`
- `experiments/results/phase_map/phase_map_metrics_max_prob_max_iter100.png`
- `experiments/results/phase_map/phase_map_metrics_max_prob_max_iter30.pdf`
- `experiments/results/phase_map/phase_map_metrics_max_prob_max_iter30.png`
- `experiments/results/phase_map/phase_map_metrics_oracle_max_iter100.pdf`
- `experiments/results/phase_map/phase_map_metrics_oracle_max_iter100.png`
- `experiments/results/phase_map/phase_map_metrics_oracle_max_iter30.pdf`
- `experiments/results/phase_map/phase_map_metrics_oracle_max_iter30.png`
- `experiments/results/phase_map/phase_map_runs.csv` (not committed: large; regenerate)
- `experiments/results/phase_map/phase_map_summary.csv`

Raw JSONL files are not in the repository because of their size. The scripts write them to `experiments/results/<experiment>/raw/`, and the summary tables above are computed from them.

## Results

| map | rule | T | n | found-opt [CI] | bit diff | norm. energy gap [CI] | iters |
|---|---|---|---|---|---|---|---|
| paper | oracle | 100 | 8 | 0.69 [0.59, 0.77] | 0.49 | 0.016 [0.009, 0.023] | 15.1 |
| paper | oracle | 100 | 10 | 0.69 [0.59, 0.77] | 0.58 | 0.014 [0.008, 0.021] | 19.5 |
| paper | oracle | 100 | 12 | 0.59 [0.49, 0.68] | 0.86 | 0.020 [0.012, 0.028] | 24.2 |
| paper | oracle | 100 | 14 | 0.48 [0.38, 0.58] | 1.02 | 0.018 [0.012, 0.025] | 32.4 |
| paper | oracle | 100 | 16 | 0.38 [0.29, 0.48] | 1.49 | 0.025 [0.018, 0.031] | 38.7 |
| paper | max_prob | 100 | 8 | 0.59 [0.49, 0.68] | 0.61 | 0.019 [0.012, 0.025] | 9.0 |
| paper | max_prob | 100 | 10 | 0.64 [0.54, 0.73] | 0.54 | 0.010 [0.007, 0.014] | 13.9 |
| paper | max_prob | 100 | 12 | 0.50 [0.40, 0.60] | 1.03 | 0.015 [0.010, 0.020] | 29.5 |
| paper | max_prob | 100 | 14 | 0.46 [0.37, 0.56] | 1.02 | 0.016 [0.011, 0.020] | 45.8 |
| paper | max_prob | 100 | 16 | 0.35 [0.26, 0.45] | 1.44 | 0.017 [0.013, 0.021] | 52.8 |
| tight | oracle | 100 | 8 | 0.59 [0.49, 0.68] | 0.68 | 0.021 [0.014, 0.028] | 9.2 |
| tight | oracle | 100 | 10 | 0.69 [0.59, 0.77] | 0.51 | 0.014 [0.008, 0.020] | 9.1 |
| tight | oracle | 100 | 12 | 0.51 [0.41, 0.61] | 1.05 | 0.024 [0.016, 0.032] | 16.8 |
| tight | oracle | 100 | 14 | 0.49 [0.39, 0.59] | 0.90 | 0.018 [0.012, 0.025] | 24.8 |
| tight | oracle | 100 | 16 | 0.47 [0.38, 0.57] | 1.28 | 0.023 [0.016, 0.030] | 16.5 |
| tight | max_prob | 100 | 8 | 0.51 [0.41, 0.61] | 0.73 | 0.023 [0.016, 0.030] | 5.9 |
| tight | max_prob | 100 | 10 | 0.62 [0.52, 0.71] | 0.58 | 0.013 [0.008, 0.018] | 6.8 |
| tight | max_prob | 100 | 12 | 0.47 [0.38, 0.57] | 1.11 | 0.018 [0.012, 0.023] | 17.0 |
| tight | max_prob | 100 | 14 | 0.44 [0.35, 0.54] | 1.08 | 0.013 [0.009, 0.017] | 21.5 |
| tight | max_prob | 100 | 16 | 0.39 [0.30, 0.49] | 1.34 | 0.015 [0.011, 0.018] | 24.5 |
| shifted | oracle | 100 | 8 | 0.43 [0.34, 0.53] | 0.93 | 0.040 [0.030, 0.049] | 2.2 |
| shifted | oracle | 100 | 10 | 0.43 [0.34, 0.53] | 1.04 | 0.041 [0.031, 0.052] | 3.3 |
| shifted | oracle | 100 | 12 | 0.35 [0.26, 0.45] | 1.33 | 0.039 [0.029, 0.048] | 5.4 |
| shifted | oracle | 100 | 14 | 0.27 [0.19, 0.36] | 1.66 | 0.037 [0.029, 0.045] | 2.9 |
| shifted | oracle | 100 | 16 | 0.30 [0.22, 0.40] | 1.66 | 0.028 [0.022, 0.034] | 5.0 |
| shifted | max_prob | 100 | 8 | 0.43 [0.34, 0.53] | 0.92 | 0.040 [0.030, 0.050] | 1.8 |
| shifted | max_prob | 100 | 10 | 0.43 [0.34, 0.53] | 1.04 | 0.041 [0.031, 0.052] | 2.4 |
| shifted | max_prob | 100 | 12 | 0.35 [0.26, 0.45] | 1.31 | 0.039 [0.029, 0.049] | 4.0 |
| shifted | max_prob | 100 | 14 | 0.25 [0.18, 0.34] | 1.71 | 0.036 [0.028, 0.044] | 4.3 |
| shifted | max_prob | 100 | 16 | 0.27 [0.19, 0.36] | 1.81 | 0.028 [0.022, 0.035] | 5.9 |
| tight_wide | oracle | 100 | 8 | 0.38 [0.29, 0.48] | 1.03 | 0.048 [0.037, 0.060] | 2.2 |
| tight_wide | oracle | 100 | 10 | 0.36 [0.27, 0.46] | 1.23 | 0.054 [0.042, 0.067] | 2.2 |
| tight_wide | oracle | 100 | 12 | 0.27 [0.19, 0.36] | 1.53 | 0.053 [0.042, 0.063] | 4.2 |
| tight_wide | oracle | 100 | 14 | 0.22 [0.15, 0.31] | 1.82 | 0.044 [0.035, 0.053] | 2.6 |
| tight_wide | oracle | 100 | 16 | 0.21 [0.14, 0.30] | 1.97 | 0.039 [0.032, 0.046] | 4.5 |
| tight_wide | max_prob | 100 | 8 | 0.38 [0.29, 0.48] | 1.02 | 0.049 [0.037, 0.060] | 1.7 |
| tight_wide | max_prob | 100 | 10 | 0.36 [0.27, 0.46] | 1.24 | 0.054 [0.042, 0.066] | 2.2 |
| tight_wide | max_prob | 100 | 12 | 0.26 [0.18, 0.35] | 1.54 | 0.052 [0.041, 0.063] | 2.8 |
| tight_wide | max_prob | 100 | 14 | 0.20 [0.13, 0.29] | 1.86 | 0.042 [0.034, 0.051] | 4.0 |
| tight_wide | max_prob | 100 | 16 | 0.20 [0.13, 0.29] | 1.95 | 0.037 [0.030, 0.044] | 4.8 |

## Findings and caveats

Written 2026-11-27 (finished; 100 instances per n, precision 3, pdiff 0.01).

- Found-optimum rate, oracle rule, T = 100, n = 8 / 10 / 12 / 14 / 16:
  paper 0.69 / 0.69 / 0.59 / 0.48 / 0.38; tight 0.59 / 0.69 / 0.51 / 0.49 / 0.47;
  shifted 0.43 / 0.43 / 0.35 / 0.27 / 0.30; tight_wide 0.38 / 0.36 / 0.27 / 0.22 / 0.21.
  (max_prob rule: same ordering, slightly lower.)
- The paper's map is the best or tied for n <= 12; the 'tight' map, which uses the full interval
  [0, pi/2], declines more slowly and is better at n = 16 (0.47 vs 0.38). It also needs fewer
  iterations (17 vs 39 at n = 16).
- Maps that put the phases in a wider interval ('tight_wide', up to 0.9 pi) or near pi ('shifted') are
  clearly worse and lock almost everything in 2-5 iterations, most likely because the first-iteration
  marginal differences are larger, so qubits cross pdiff before the amplification has favoured the
  optimum.
- Caveats: precision 3 only (the setting where rounding stalls the paper map at large n; the 'tight'
  map's advantage may partly be a larger first-step contrast that survives rounding); fixed pdiff.

## Regenerate tables and plots from the saved data (no re-run)

```bash
.venv/bin/python experiments/phase_map_ablation.py --plot-only --out experiments/results/phase_map/raw/phase_map.jsonl
.venv/bin/python experiments/reports.py
```
