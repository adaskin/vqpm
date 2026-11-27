Written 2026-11-25 from the n = 4..20 data; n = 21, 22 (finished later) follow the same pattern:
precision 3 0.20 / 0.33, precision 6 or exact at T = 300 0.51 / 0.57 (oracle rule).

- **The 30-iteration cap is not what limits precision-3 VQPM; rounding is.** With marginals rounded
  to 3 decimals the found-optimum rate is the same for every budget (oracle rule: n = 12 0.59,
  n = 16 0.37 -> 0.38, n = 20 0.28 for T = 30 and T = 300). At T = 300, every precision-3 run that
  reaches the budget is stalled (no change of marginals or locks for >= 10 iterations): 17% of runs
  at n = 12, 31% at n = 16, 46% at n = 20. The rounded marginals become a fixed point.
- **With finer rounding the budget matters and the decline with n flattens.** Precision 6 and exact
  marginals give identical results (6-decimal rounding is effectively exact). Oracle rule,
  T = 30 -> T = 300: n = 12 0.62 -> 0.66, n = 16 0.52 -> 0.57, n = 17 0.48 -> 0.66, n = 20 0.41 -> 0.56.
  At T = 300 almost no run hits the budget (<= 1%); mean iterations used 16 (n = 12) to 21 (n = 20).
  Between n = 12 and n = 20 the rate stays around 0.55-0.66 instead of falling to 0.28.
- **The remaining failures are wrong locks.** Failure split at T = 300 (oracle): precision 6, n = 20:
  44% of runs fail by a wrong lock, 0% by not converging; precision 3, n = 20: 45% wrong lock, 27%
  not converged (stalled). About 5-7% of all locks are wrong, and the fraction of runs with at least
  one wrong lock grows with n (0.26 at n = 8, 0.44 at n = 20 for precision 6); see wrong_locks.md.
- **Missed optima are near-optimal in energy.** Mean normalised energy gap (E_found - E_min)/(E_max - E_min)
  at T = 300 is 0.016-0.030 for all n and precisions, even where the found-optimum rate is 0.3-0.6;
  mean Hamming distance 0.4-2.0 bits.
- **max_prob (no oracle) is lower than oracle** at small n (it can stop on a non-optimal dominant state or
  when all qubits lock), e.g. n = 12 precision 6: 0.54 vs 0.66; budgets matter less for it.
- **Shots-equivalent (for the planned figure):** resolving marginals to d decimals needs about 10^(2d)
  shots per iteration, i.e. ~1e6 for precision 3 (total ~1e7-1e8 per run here) and ~1e12 for
  precision 6 (total ~1e13). The precision-6 gains are therefore not reachable with realistic shot
  counts; Exp. 4 (finite shots) measures what is reachable.
- Caveats: exact marginals (no shot noise, no circuit noise); fixed pdiff = 0.01; paper phase map;
  the oracle rule uses the known optimum to stop; random Gaussian QUBOs only.
- Data for the found-optimum-rate figure (found-optimum rate vs n for precision {3, 6, exact} x budget
  {30, 100, 300, 10n}, with a theory curve and the shots caveat): `vqpm_grid_runs.csv` (one row per
  run: success, iterations used, stop reason, stall flag, shots_equiv_total, energy metrics) and
  `vqpm_grid_summary.csv` (rates with Wilson intervals); the per-iteration trajectories are in the
  raw JSONL, so other budgets can be read off without re-running.
