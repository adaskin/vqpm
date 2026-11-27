Written 2026-11-25 (finished pilot; 20 tuning instances, seed 777, n = 8, never used in the main runs).

- L-BFGS-B with exact adjoint gradients gives the same results as with finite-difference gradients
  (p = 8, random starts: found-optimum 0.75 vs 0.80, P(opt) 0.61 vs 0.60) at about 5x less wall time,
  so the main runs use adjoint gradients and report finite-difference-equivalent evaluation counts.
- L-BFGS-B beats COBYLA with random starts (P(opt) at p = 8: 0.61 vs 0.13; found-optimum 0.75 vs 0.55),
  supporting the paper's statement. With INTERP, COBYLA catches up in found-optimum rate (0.90) but
  not in P(opt) (0.54).
- Adam: best learning rate 0.3 (tried 0.01-0.3): random starts p = 8 found-optimum 0.85, P(opt) 0.60.
- Plain gradient descent (what SGD reduces to on a statevector): best learning rate 1 (tried 0.03-10),
  but clearly worse (random p = 8: 0.55, P(opt) 0.13); lr 10 diverges.
- VQPM on the same instances: found-optimum 0.70 (oracle) / 0.65 (max_prob), P(opt) 0.44, using at
  most 30 iterations. Simulated annealing: 0.95 (10 sweeps), 1.00 (100 sweeps).
- Evaluation cost per instance (5 restarts, p = 8): ~4.5e4 finite-difference-equivalent evaluations
  for L-BFGS-B, 2.5e3 for COBYLA (random), ~4e4 (random) to 2e5 (INTERP chain) for gd / Adam.
