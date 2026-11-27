# Circuit cost: transpiled controlled-U and noisy-simulation feasibility

_Report generated 2026-11-27T07:51:15 by `experiments/reports.py`; status: finished._

## Purpose

Hardware cost of one VQPM iteration.

## Method

Dense random QUBO (seed [0, n]); controlled-U = P(b) on the control, CP for diagonal terms, MCPhase (two controls) for each pair; transpiled with optimization_level 3 to {cx, rz, sx, x} without coupling map, and onto FakeTorino (CZ, heavy-hex) and FakeSherbrooke (ECR).

Metrics (per run, energies in the units of the original, unscaled Q): found optimum = the returned bitstring is the exact ground state (found by enumerating all 2^n energies); bit difference = Hamming distance to the ground state; energy gap = E_found - E_min; normalised energy gap = (E_found - E_min)/(E_max - E_min) in [0, 1]; normalised expected-energy gap = (<E> - E_min)/(E_max - E_min) with <E> the expected energy of the final output distribution (VQPM, QAOA). Intervals: Wilson 95% for rates, mean +- 1.96 s/sqrt(N) for means.

## Provenance

- Library versions (pinned in requirements.txt): matplotlib 3.11.2, numpy 2.5.5, pytest 9.2.0, qiskit 2.6.2, qiskit-aer 0.18.4, qiskit-ibm-runtime 0.52.1, scipy 1.18.1, python 3.12.12
- Machine: AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 16 logical CPUs, 90 GiB RAM, Linux 6.12.113+deb13-amd64
- End time (last write of the raw data): `controlled_u_gate_counts.json` 2026-11-25T17:08:43

## Where the data is

Paths are relative to the repository root.

| file (repo-relative) | records / size | sha256 |
|---|---|---|
| `experiments/results/gate_counts/controlled_u_gate_counts.json` | 1.8 KiB | `dd159f179802889b2e29578f1bf997f078f38ce2af54ab5b1260420668bf698d` |

Summary tables and plots (committed to git unless noted):

- `experiments/results/gate_counts/controlled_u_gate_counts.json`

Raw JSONL files are not in the repository because of their size. The scripts write them to `experiments/results/<experiment>/raw/`, and the summary tables above are computed from them.

## Results

| n | target | 2q gates | depth | 2q depth | pairs |
|---|---|---|---|---|---|
| 8 | all-to-all | 172 cx | 292 | 167 | 28 |
| 8 | FakeTorino | 407 cz | 995 | 307 | 28 |
| 8 | FakeSherbrooke | 399 ecr | 1483 | 310 | 28 |
| 12 | all-to-all | 400 cx | 694 | 391 | 66 |
| 12 | FakeTorino | 1137 cz | 2321 | 745 | 66 |
| 12 | FakeSherbrooke | 1150 ecr | 3339 | 758 | 66 |
| 16 | all-to-all | 718 cx | 1261 | 705 | 120 |
| 16 | FakeTorino | 2132 cz | 4123 | 1331 | 120 |
| 16 | FakeSherbrooke | 2185 ecr | 5712 | 1304 | 120 |

## Findings and caveats

- One VQPM iteration needs a controlled-U with one doubly controlled phase per QUBO pair: 172 / 400 / 718
  CX (depth 292 / 694 / 1261) for n = 8 / 12 / 16 with all-to-all connectivity, and 407 / 1137 / 2132 CZ
  (depth 995 / 2321 / 4123) on FakeTorino's heavy-hex lattice (FakeSherbrooke: 399 / 1150 / 2185 ECR).
- Estimated fidelity of one full iteration (product of calibrated gate fidelities, no decoherence) on
  FakeTorino: 4% (n = 8), 0.7% (n = 10), 0.1% (n = 12); FakeFez: 19% / 6% / 1%. Current hardware cannot
  run VQPM beyond n ~ 8 without error mitigation or a cheaper controlled-U.
- Simulation cost (measured): routed FakeTorino circuits use extra physical qubits, so density-matrix
  simulation is infeasible beyond n = 8 (17 GB at n = 10); per-shot trajectories were 30x slower than
  density matrix at n = 8. All-to-all depolarising model, density matrix, 1e5 shots: 1.4 s per
  iteration (n = 8), 32 s (n = 10), 13 min (n = 12); hence the n = 12 noisy runs were dropped.

## Regenerate tables and plots from the saved data (no re-run)

```bash
.venv/bin/python vqpm_qiskit_noisy/vqpm_qiskit.py --gate-counts 8 12 16 --gate-count-backends all-to-all FakeTorino FakeSherbrooke --out experiments/results/gate_counts/controlled_u_gate_counts.json
.venv/bin/python experiments/reports.py
```
