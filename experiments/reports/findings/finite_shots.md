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
