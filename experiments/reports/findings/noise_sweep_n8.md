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
