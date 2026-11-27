Written 2026-11-27 from the finished run (100 instances for n = 4..10, 50 for n = 12..16; seed 2026).
Numbers below are for QAOA depth p = 8 unless stated; "found optimum" = the most probable bitstring of
the final QAOA state (VQPM: the returned state) is the exact ground state.

- **QAOA with a good optimizer finds the optimum more often than VQPM at every n, at about 1000x more
  evaluations.** Found-optimum rate n = 4 / 8 / 12 / 16:
  L-BFGS-B random restarts 0.96 / 0.90 / 0.86 / 0.74; L-BFGS-B INTERP 0.97 / 0.90 / 0.86 / 0.78;
  Adam random 0.93 / 0.92 / 0.86 / 0.68; Adam INTERP 0.93 / 0.91 / 0.86 / 0.76;
  VQPM (precision 3, oracle, T = 100) 0.80 / 0.69 / 0.62 / 0.46; VQPM max_prob 0.77 / 0.59 / 0.52 / 0.42.
  Cost per instance: L-BFGS-B ~3e4-6e4 finite-difference-equivalent energy evaluations (5 restarts),
  Adam ~4e4 (random) to ~1.3e5 (INTERP); VQPM 3-33 iterations (one circuit each, but each needs
  ~1e6 shots for 3-decimal marginals, see finite_shots.md).
- **QAOA depends strongly on the optimizer.** At n = 16, p = 8 the found-optimum rate ranges from 0.40
  (gradient descent with INTERP) to 0.78 (L-BFGS-B with INTERP); with random restarts COBYLA reaches
  0.48-0.62 and plain gradient descent 0.52-0.64 for n = 10..16, while L-BFGS-B / Adam reach 0.68-0.87.
  INTERP helps COBYLA a lot (n = 12: 0.62 -> 0.78) but hurts plain gradient descent at large n
  (n = 16: 0.58 random vs 0.40 INTERP). Other depths: qaoa_vs_vqpm_metrics_all_optimizers_p{2,4,6}.png
  (not summarised here).
- **The paper's "VQPM comparable to QAOA" holds only against a weakly optimised QAOA** (COBYLA or gradient
  descent with random starts, which are at VQPM's level for n >= 10); against L-BFGS-B or Adam with
  restarts or INTERP it does not.
- **Probability of the optimum in the final state** declines with n for both: QAOA L-BFGS-B random 0.92 /
  0.63 / 0.47 / 0.30 (n = 4 / 8 / 12 / 16) vs VQPM 0.53 / 0.43 / 0.39 / 0.29, i.e. similar at n = 16.
- **Energy:** the returned states are near-optimal for every method. Mean normalised energy gap of the
  returned state: QAOA L-BFGS-B / Adam 0.000-0.005, COBYLA / gd random 0.003-0.017, VQPM 0.010-0.024.
  Mean normalised expected-energy gap of the final distribution: QAOA L-BFGS-B 0.003-0.030, VQPM
  0.046-0.063 (oracle) / 0.021-0.059 (max_prob).
- **Simulated annealing** solves all these instances: 100 sweeps (n x 100 single-flip proposals) give
  0.98-1.00, 1000 sweeps 1.00, and even 10 sweeps 0.70-0.96.
- Caveats: QAOA expectations are exact (no shot noise) and evaluation counts do not include shots; QAOA
  runs on the paper's scaled QUBO (energies in [-pi/4, pi/4]) with random initial angles in [0, pi);
  learning rates were tuned on separate instances at n = 8 only; VQPM uses the published setting
  (precision 3, pdiff 0.01) -- with precision 6 it would reach ~0.57 at n = 16 (vqpm_grid.md) but needs
  ~1e12 shots per iteration; random Gaussian QUBOs only.
