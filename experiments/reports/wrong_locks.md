# Exp. 2: wrong-lock statistics (from the Exp. 1 lock logs)

_Report generated 2026-11-27T07:51:13 by `experiments/reports.py`; status: finished._

## Purpose

How often VQPM locks a qubit to the wrong value, when, and with what margin.

## Method

Every lock event of Exp. 1 is compared with the corresponding bit of the exact ground state (matches_optimum). Reported per precision and n over the whole recorded trajectory: wrong locks / all locks, fraction of runs with >= 1 wrong lock, median iteration of the first wrong lock, median margin |P0 - P1| of correct and wrong locks. The failure split per stop rule and budget (wrong lock / budget exhausted / stopped at a wrong state) is in vqpm_grid_summary.csv (fail_* columns).

Metrics (per run, energies in the units of the original, unscaled Q): found optimum = the returned bitstring is the exact ground state (found by enumerating all 2^n energies); bit difference = Hamming distance to the ground state; energy gap = E_found - E_min; normalised energy gap = (E_found - E_min)/(E_max - E_min) in [0, 1]; normalised expected-energy gap = (<E> - E_min)/(E_max - E_min) with <E> the expected energy of the final output distribution (VQPM, QAOA). Intervals: Wilson 95% for rates, mean +- 1.96 s/sqrt(N) for means.

## Provenance

- Invocation 1 of `experiments/results/vqpm_grid/raw/grid.jsonl`: started 2026-11-25T17:45:14, git commit `fbc2114ddcbb368ea55e458fa0dd6cb0485c37d7`, python 3.12.12, numpy 2.5.5, scipy 1.18.1
  - command: `.venv/bin/python experiments/vqpm_grid.py --n 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 --instances 100 --workers 10 --out experiments/results/vqpm_grid/raw/grid.jsonl`
  - parameters: `{"budgets": ["30", "100", "300", "10n"], "inst_dir": "experiments/instances", "instances": 100, "n": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], "out": "experiments/results/vqpm_grid/raw/grid.jsonl", "pdiff": 0.01, "phase_maps": ["paper"], "plot_only": false, "precision": ["3", "6", "exact"], "seed": 2026, "summary_dir": null, "workers": 10}`
- The n = 21, 22 records were added by a second invocation of the same command with `--n 21 22` (it resumes the same file); expected energies of records made before commit ed0b4c4 come from `--backfill-expected-energy` (sidecar file).
- Library versions (pinned in requirements.txt): matplotlib 3.11.2, numpy 2.5.5, pytest 9.2.0, qiskit 2.6.2, qiskit-aer 0.18.4, qiskit-ibm-runtime 0.52.1, scipy 1.18.1, python 3.12.12
- Machine: AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 16 logical CPUs, 90 GiB RAM, Linux 6.12.113+deb13-amd64
- End time (last write of the raw data): `grid.jsonl` 2026-11-25T19:52:56
- Wall time `grid.jsonl` (first start to last write, includes waits between invocations): 2.13 h
- Instances: seed 2026, files `experiments/instances/qubo_n<n>_seed2026.npz` (instance i is `default_rng([2026, n, i])`, Q = A + A^T, A_ij ~ N(0, 1); each file stores Q, optimum, E_min, E_max, gap); sha256 listed below

## Where the data is

Paths are relative to the repository root.

| file (repo-relative) | records / size | sha256 |
|---|---|---|
| `experiments/results/vqpm_grid/raw/grid.jsonl` | 5700 records, 42.3 MiB | `04f126275ecae6a79ac87004419781b94cecade848291b2c0204faf2f2e77200` |
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

- `experiments/results/vqpm_grid/wrong_locks_margins.pdf`
- `experiments/results/vqpm_grid/wrong_locks_margins.png`
- `experiments/results/vqpm_grid/wrong_locks_summary.csv`
- `experiments/results/vqpm_grid/wrong_locks_vs_n.pdf`
- `experiments/results/vqpm_grid/wrong_locks_vs_n.png`

Raw JSONL files are not in the repository because of their size. The scripts write them to `experiments/results/<experiment>/raw/`, and the summary tables above are computed from them.

## Results

| prec | n | locks | wrong/locks | runs w/ wrong lock | median 1st wrong-lock iter | median margin correct | median margin wrong |
|---|---|---|---|---|---|---|---|
| 3 | 4 | 392 | 0.043 | 0.16 | 0.0 | 0.024 | 0.016 |
| 3 | 5 | 489 | 0.063 | 0.25 | 0.0 | 0.020 | 0.014 |
| 3 | 6 | 584 | 0.050 | 0.23 | 1.0 | 0.016 | 0.016 |
| 3 | 7 | 687 | 0.036 | 0.18 | 1.0 | 0.016 | 0.012 |
| 3 | 8 | 779 | 0.051 | 0.22 | 1.0 | 0.014 | 0.012 |
| 3 | 9 | 853 | 0.052 | 0.26 | 2.5 | 0.014 | 0.012 |
| 3 | 10 | 955 | 0.036 | 0.21 | 1.0 | 0.014 | 0.012 |
| 3 | 11 | 1030 | 0.064 | 0.33 | 2.0 | 0.014 | 0.012 |
| 3 | 12 | 1121 | 0.056 | 0.29 | 2.0 | 0.012 | 0.012 |
| 3 | 13 | 1189 | 0.035 | 0.28 | 3.0 | 0.012 | 0.012 |
| 3 | 14 | 1274 | 0.043 | 0.34 | 5.0 | 0.012 | 0.012 |
| 3 | 15 | 1345 | 0.053 | 0.41 | 3.0 | 0.012 | 0.012 |
| 3 | 16 | 1438 | 0.044 | 0.34 | 3.0 | 0.012 | 0.012 |
| 3 | 17 | 1491 | 0.050 | 0.33 | 4.0 | 0.012 | 0.012 |
| 3 | 18 | 1591 | 0.033 | 0.33 | 5.0 | 0.012 | 0.012 |
| 3 | 19 | 1605 | 0.037 | 0.32 | 5.0 | 0.012 | 0.012 |
| 3 | 20 | 1724 | 0.055 | 0.45 | 5.0 | 0.012 | 0.012 |
| 3 | 21 | 1749 | 0.046 | 0.50 | 5.0 | 0.012 | 0.012 |
| 3 | 22 | 1831 | 0.047 | 0.41 | 5.0 | 0.012 | 0.012 |
| 6 | 4 | 400 | 0.058 | 0.21 | 0.0 | 0.024 | 0.014 |
| 6 | 5 | 500 | 0.076 | 0.25 | 0.0 | 0.019 | 0.013 |
| 6 | 6 | 600 | 0.052 | 0.23 | 1.0 | 0.015 | 0.014 |
| 6 | 7 | 699 | 0.052 | 0.23 | 0.0 | 0.014 | 0.012 |
| 6 | 8 | 799 | 0.063 | 0.26 | 1.0 | 0.013 | 0.011 |
| 6 | 9 | 898 | 0.058 | 0.27 | 2.0 | 0.013 | 0.011 |
| 6 | 10 | 999 | 0.037 | 0.22 | 1.5 | 0.012 | 0.011 |
| 6 | 11 | 1100 | 0.059 | 0.32 | 1.0 | 0.012 | 0.011 |
| 6 | 12 | 1197 | 0.065 | 0.33 | 2.0 | 0.012 | 0.011 |
| 6 | 13 | 1297 | 0.051 | 0.35 | 2.0 | 0.012 | 0.010 |
| 6 | 14 | 1395 | 0.057 | 0.34 | 3.0 | 0.012 | 0.011 |
| 6 | 15 | 1497 | 0.069 | 0.47 | 4.0 | 0.011 | 0.011 |
| 6 | 16 | 1597 | 0.062 | 0.42 | 5.0 | 0.011 | 0.010 |
| 6 | 17 | 1692 | 0.063 | 0.34 | 4.0 | 0.011 | 0.010 |
| 6 | 18 | 1795 | 0.052 | 0.41 | 4.0 | 0.011 | 0.010 |
| 6 | 19 | 1894 | 0.063 | 0.43 | 5.0 | 0.011 | 0.010 |
| 6 | 20 | 1996 | 0.072 | 0.44 | 4.0 | 0.011 | 0.010 |
| 6 | 21 | 2094 | 0.058 | 0.49 | 5.0 | 0.011 | 0.010 |
| 6 | 22 | 2196 | 0.068 | 0.42 | 5.5 | 0.011 | 0.010 |
| exact | 4 | 400 | 0.058 | 0.21 | 0.0 | 0.024 | 0.014 |
| exact | 5 | 500 | 0.076 | 0.25 | 0.0 | 0.019 | 0.013 |
| exact | 6 | 600 | 0.052 | 0.23 | 1.0 | 0.015 | 0.014 |
| exact | 7 | 699 | 0.052 | 0.23 | 0.0 | 0.014 | 0.012 |
| exact | 8 | 799 | 0.063 | 0.26 | 1.0 | 0.013 | 0.011 |
| exact | 9 | 898 | 0.058 | 0.27 | 2.0 | 0.013 | 0.011 |
| exact | 10 | 999 | 0.037 | 0.22 | 1.5 | 0.012 | 0.011 |
| exact | 11 | 1100 | 0.059 | 0.32 | 1.0 | 0.012 | 0.011 |
| exact | 12 | 1197 | 0.065 | 0.33 | 2.0 | 0.012 | 0.011 |
| exact | 13 | 1297 | 0.051 | 0.35 | 2.0 | 0.012 | 0.010 |
| exact | 14 | 1395 | 0.057 | 0.34 | 3.0 | 0.012 | 0.011 |
| exact | 15 | 1497 | 0.069 | 0.47 | 4.0 | 0.011 | 0.011 |
| exact | 16 | 1597 | 0.062 | 0.42 | 5.0 | 0.011 | 0.010 |
| exact | 17 | 1693 | 0.063 | 0.34 | 4.0 | 0.011 | 0.010 |
| exact | 18 | 1795 | 0.052 | 0.41 | 4.0 | 0.011 | 0.010 |
| exact | 19 | 1894 | 0.063 | 0.43 | 5.0 | 0.011 | 0.010 |
| exact | 20 | 1996 | 0.072 | 0.44 | 4.0 | 0.011 | 0.010 |
| exact | 21 | 2094 | 0.058 | 0.49 | 5.0 | 0.011 | 0.010 |
| exact | 22 | 2196 | 0.068 | 0.42 | 5.5 | 0.011 | 0.010 |

## Findings and caveats

Written 2026-11-25 from the n = 4..20 grid data (n = 21, 22 partial at the time).

- About 5-7% of all lock events fix a qubit to the wrong value (precision 3: 4.4-5.6%; precision 6 /
  exact: 6.2-7.2% for n = 8..20). Because a run makes about n locks, the fraction of runs with at
  least one wrong lock grows with n: precision 6 0.26 (n = 8), 0.33 (12), 0.42 (16), 0.44 (20).
- Wrong locks happen early: the median first wrong lock is at iteration 1-5, i.e. among the first
  locks of a run, when marginals are still close to 1/2.
- The lock margins of wrong and correct locks are almost the same (median |P0 - P1| ~0.010-0.012 vs
  ~0.011-0.014, just above pdiff = 0.01). A larger fixed threshold would therefore delay locks
  rather than separate right from wrong ones; the wrong locks come from the product-state
  (mean-field) dynamics favouring the wrong value at that moment, not from estimation noise
  (marginals are exact here).
- Caveat: "wrong" means different from the exact ground state; with near-degenerate instances a
  wrong lock can still lead to a low-energy state (see the energy gaps in vqpm_grid.md).

## Regenerate tables and plots from the saved data (no re-run)

```bash
.venv/bin/python experiments/wrong_locks.py --grid experiments/results/vqpm_grid/raw/grid.jsonl
.venv/bin/python experiments/reports.py
```
