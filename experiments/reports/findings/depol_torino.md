Written 2026-11-27 (finished; 5 instances x 3 seeds per n, all-to-all connectivity).

- Depolarising + readout noise at FakeTorino's median rates (2.7e-4 1q, 4.1e-3 2q, 2.3e-2 readout),
  i.e. device error rates without routing overhead: found-optimum rate 0.27 at n = 8 (exact VQPM 1.00
  on the same instances) and 0.00 at n = 10 (exact 0.60); mean bit difference 1.2 and 3.1, normalised
  energy gap 0.12 and 0.30. All runs used the full 100 iterations.
- Together with the noise sweep (noise_sweep_n8.md): VQPM needs two-qubit error rates around 1e-4 to
  keep its noiseless performance at n = 8, and the requirement tightens with n because the
  controlled-U grows as n^2 two-qubit gates. n = 12 was not simulated (13 min per iteration).
