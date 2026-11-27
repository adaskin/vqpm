Written 2026-11-27 (finished; 5 instances x 3 seeds, n = 8).

- With the full FakeTorino noise model (calibrated per-qubit errors, heavy-hex routing: ~480 two-qubit
  gates per iteration, estimated circuit fidelity ~4%), VQPM never found the optimum (0/15); mean bit
  difference 5.5 of 8 and normalised energy gap 0.55, i.e. no better than a random bitstring. The exact
  VQPM with the same settings found the optimum in 5/5 instances. Current IBM hardware cannot run
  this circuit usefully at n = 8.
