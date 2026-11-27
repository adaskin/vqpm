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
