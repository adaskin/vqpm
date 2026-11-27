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
