# Experiments: index

_Updated 2026-11-27T07:51:15. One-call progress check: `.venv/bin/python experiments/status.py`.._

| experiment | purpose | status | report |
|---|---|---|---|
| Exp. 1: VQPM baseline grid (precision x iteration budget x stop rule) | iteration budget vs precision; baseline | finished | [vqpm_grid.md](vqpm_grid.md) |
| Exp. 2: wrong-lock statistics (from the Exp. 1 lock logs) | wrong locks | finished | [wrong_locks.md](wrong_locks.md) |
| Exp. 3: phase-map ablation | phase maps | finished | [phase_map.md](phase_map.md) |
| Exp. 4: finite shots | finite shots | finished | [finite_shots.md](finite_shots.md) |
| Exp. 5: QAOA (4 optimizers) vs VQPM vs simulated annealing | QAOA baseline | finished | [qaoa_vs_vqpm.md](qaoa_vs_vqpm.md) |
| Exp. 6c: noise-strength sweep at n = 8 (Qiskit Aer, density matrix) | noise | finished | [noise_sweep_n8.md](noise_sweep_n8.md) |
| Exp. 6b: full FakeTorino noise model at n = 8 | noise | finished | [faketorino_n8.md](faketorino_n8.md) |
| Exp. 6a: Torino-calibrated depolarising + readout noise, n = 8, 10 | noise | finished | [depol_torino.md](depol_torino.md) |
| Pilot: QAOA optimizer and learning-rate choice (n = 8, tuning instances) | optimizer choice | finished | [qaoa_optimizer_pilot.md](qaoa_optimizer_pilot.md) |
| Reproduction of the original figures | original figures | finished | [reproduction.md](reproduction.md) |
| Circuit cost: transpiled controlled-U and noisy-simulation feasibility | circuit cost | finished | [gate_counts.md](gate_counts.md) |
