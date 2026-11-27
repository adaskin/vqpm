Written 2026-11-27 (finished; 100 instances per n, precision 3, pdiff 0.01).

- Found-optimum rate, oracle rule, T = 100, n = 8 / 10 / 12 / 14 / 16:
  paper 0.69 / 0.69 / 0.59 / 0.48 / 0.38; tight 0.59 / 0.69 / 0.51 / 0.49 / 0.47;
  shifted 0.43 / 0.43 / 0.35 / 0.27 / 0.30; tight_wide 0.38 / 0.36 / 0.27 / 0.22 / 0.21.
  (max_prob rule: same ordering, slightly lower.)
- The paper's map is the best or tied for n <= 12; the 'tight' map, which uses the full interval
  [0, pi/2], declines more slowly and is better at n = 16 (0.47 vs 0.38). It also needs fewer
  iterations (17 vs 39 at n = 16).
- Maps that put the phases in a wider interval ('tight_wide', up to 0.9 pi) or near pi ('shifted') are
  clearly worse and lock almost everything in 2-5 iterations, most likely because the first-iteration
  marginal differences are larger, so qubits cross pdiff before the amplification has favoured the
  optimum.
- Caveats: precision 3 only (the setting where rounding stalls the paper map at large n; the 'tight'
  map's advantage may partly be a larger first-step contrast that survives rounding); fixed pdiff.
