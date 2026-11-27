# Exp. 6c: noise-strength sweep at n = 8 (Qiskit Aer, density matrix)

_Report generated 2026-11-27T07:51:14 by `experiments/reports.py`; status: finished._

## Purpose

How much circuit noise VQPM tolerates.

## Method

Circuit VQPM (vqpm_qiskit_noisy/vqpm_qiskit.py, measure_mode='all'): each iteration one circuit prepares prod_q Ry(theta_q)|0>, applies H - controlled-U - H on the control and measures the control and all data qubits; marginals come from the shots with control = 0. Aer density-matrix simulation (shot count nearly free), 1e5 shots per iteration, pdiff 0.1, precision 4, max_iter 100, stop when all qubits are locked; returned state = locked bits + majority of the last marginals. 5 instances (seed 2026) x 3 simulator seeds. Noise: 'none' = ideal sampling; 'depol:p1,p2,ro' = depolarising p1 on sx/x, p2 on CX, symmetric readout error ro, all-to-all (2.7e-4, 4.1e-3, 2.3e-2 are the FakeTorino medians); 'fake:FakeTorino' = calibrated per-qubit noise with routing on the heavy-hex coupling map. 'exact' rows = state-vector VQPM with the same pdiff/precision/budget (stop_rule 'none'). Representative trajectory plot = median-error run, not hand-picked.

Metrics (per run, energies in the units of the original, unscaled Q): found optimum = the returned bitstring is the exact ground state (found by enumerating all 2^n energies); bit difference = Hamming distance to the ground state; energy gap = E_found - E_min; normalised energy gap = (E_found - E_min)/(E_max - E_min) in [0, 1]; normalised expected-energy gap = (<E> - E_min)/(E_max - E_min) with <E> the expected energy of the final output distribution (VQPM, QAOA). Intervals: Wilson 95% for rates, mean +- 1.96 s/sqrt(N) for means.

## Provenance

- Invocation 1 of `experiments/results/noisy/raw/noise_sweep_n8.jsonl`: started 2026-11-25T17:45:17, git commit `fbc2114ddcbb368ea55e458fa0dd6cb0485c37d7`, python 3.12.12, numpy 2.5.5, scipy 1.18.1
  - command: `.venv/bin/python experiments/noisy_runs.py --n 8 --instances 5 --sim-seeds 0 1 2 --shots 100000 --workers 4 --noise none depol:1e-5,1e-4,1e-3 depol:1e-4,1e-3,1e-2 depol:2.7e-4,4.1e-3,2.3e-2 --out experiments/results/noisy/raw/noise_sweep_n8.jsonl`
  - parameters: `{"inst_dir": "experiments/instances", "instances": 5, "locking_rule": "pdiff", "max_iter": 100, "method": "density_matrix", "n": [8], "noise": ["none", "depol:1e-5,1e-4,1e-3", "depol:1e-4,1e-3,1e-2", "depol:2.7e-4,4.1e-3,2.3e-2"], "out": "experiments/results/noisy/raw/noise_sweep_n8.jsonl", "pdiff": 0.1, "plot_only": false, "precision": 4, "seed": 2026, "shots": 100000, "sim_seeds": [0, 1, 2], "threads": 1, "workers": 4}`
- Library versions (pinned in requirements.txt): matplotlib 3.11.2, numpy 2.5.5, pytest 9.2.0, qiskit 2.6.2, qiskit-aer 0.18.4, qiskit-ibm-runtime 0.52.1, scipy 1.18.1, python 3.12.12
- Machine: AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 16 logical CPUs, 90 GiB RAM, Linux 6.12.113+deb13-amd64
- End time (last write of the raw data): `noise_sweep_n8.jsonl` 2026-11-25T18:18:49
- Wall time `noise_sweep_n8.jsonl` (first start to last write, includes waits between invocations): 0.56 h
- Instances: seed 2026, files `experiments/instances/qubo_n<n>_seed2026.npz` (instance i is `default_rng([2026, n, i])`, Q = A + A^T, A_ij ~ N(0, 1); each file stores Q, optimum, E_min, E_max, gap); sha256 listed below

## Where the data is

Paths are relative to the repository root.

| file (repo-relative) | records / size | sha256 |
|---|---|---|
| `experiments/results/noisy/raw/noise_sweep_n8.jsonl` | 65 records, 0.5 MiB | `ad59895d280f19388e7ffd047cab306e04b069fd6bbb1e027b653809998297c4` |
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

- `experiments/results/noisy/noise_sweep_n8_aggregate.pdf`
- `experiments/results/noisy/noise_sweep_n8_aggregate.png`
- `experiments/results/noisy/noise_sweep_n8_metrics.pdf`
- `experiments/results/noisy/noise_sweep_n8_metrics.png`
- `experiments/results/noisy/noise_sweep_n8_runs.csv`
- `experiments/results/noisy/noise_sweep_n8_summary.csv`
- `experiments/results/noisy/noise_sweep_n8_trajectory_representative.pdf`
- `experiments/results/noisy/noise_sweep_n8_trajectory_representative.png`

Raw JSONL files are not in the repository because of their size. The scripts write them to `experiments/results/<experiment>/raw/`, and the summary tables above are computed from them.

## Results

| noise | n | runs | found-opt [CI] | bit diff [CI] | norm. energy gap [CI] | iters | wrong locks/run | shots used |
|---|---|---|---|---|---|---|---|---|
| depol:1e-4,1e-3,1e-2 | 8 | 15 | 0.47 [0.25, 0.70] | 0.87 [0.30, 1.44] | 0.039 [0.018, 0.059] | 100.0 | 0.20 | 10000000 |
| depol:1e-5,1e-4,1e-3 | 8 | 15 | 0.73 [0.48, 0.89] | 0.40 [0.03, 0.77] | 0.018 [0.001, 0.035] | 72.9 | 0.20 | 7293333 |
| depol:2.7e-4,4.1e-3,2.3e-2 | 8 | 15 | 0.27 [0.11, 0.52] | 1.20 [0.69, 1.71] | 0.120 [0.070, 0.171] | 100.0 | 0.00 | 10000000 |
| exact | 8 | 5 | 1.00 [0.57, 1.00] | 0.00 [0.00, 0.00] | 0.000 [-0.000, 0.000] | 55.8 | - | - |
| none | 8 | 15 | 0.67 [0.42, 0.85] | 0.47 [0.09, 0.84] | 0.021 [0.004, 0.039] | 70.9 | 0.20 | 7086667 |

## Findings and caveats

Written 2026-11-25 (finished run; 5 instances x 3 simulator seeds per noise level, n = 8).

- Exact state-vector VQPM with the same settings (pdiff 0.1, precision 4) found the optimum in 5/5
  instances. Ideal circuit sampling with 1e5 shots per iteration: 0.67 [0.42, 0.85]; the gap is most likely
  shot noise (early in a run the marginals move by only ~1e-3 to 1e-2 per iteration, comparable to
  the sampling error), but with 5 instances the difference is within the intervals.
- Depolarising + readout noise (p1, p2, readout):
  (1e-5, 1e-4, 1e-3): 0.73 [0.48, 0.89], normalised energy gap 0.018;
  (1e-4, 1e-3, 1e-2): 0.47 [0.25, 0.70], gap 0.039, all runs hit the 100-iteration cap;
  (2.7e-4, 4.1e-3, 2.3e-2) = FakeTorino medians: 0.27 [0.11, 0.52], gap 0.120, all runs hit the cap.
- So VQPM at n = 8 tolerates two-qubit error ~1e-4 (no loss vs ideal sampling), degrades at 1e-3,
  and at current-hardware error rates (4e-3, with 184 CX per iteration even without routing) it
  degrades strongly (found-optimum 0.27, mean 1.2 wrong bits of 8, normalised energy gap 0.12). Consistent with the estimated circuit fidelity of one
  iteration (gate_counts.md).
- Caveats: 5 instances only (wide intervals); all-to-all connectivity (no SWAP overhead); density-
  matrix simulation, so 1e5 shots per iteration cost nothing extra here but would on hardware
  (7e6-1e7 shots per run).

## Regenerate tables and plots from the saved data (no re-run)

```bash
.venv/bin/python experiments/noisy_runs.py --plot-only --out experiments/results/noisy/raw/noise_sweep_n8.jsonl
.venv/bin/python experiments/reports.py
```
