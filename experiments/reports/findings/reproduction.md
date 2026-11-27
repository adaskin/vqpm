- The current code with default options reproduces the published numbers exactly, trial by trial
  (identical to the frozen published vqpm.py in tests/reference, including printed output):
  Fig. 6a mean bit difference 1.16; scaling figure 0.4645 / 0.4008 / 0.1755 at n = 12 / 14 / 16
  (published 0.46 / 0.40 / 0.18).
- Fig. 9: VQPM numbers match (final probability 0.4346 vs 0.43, bit difference 0.43 vs 0.43); QAOA final
  probability 0.5495 vs 0.55 matches; QAOA bit difference 0.32 vs 0.33 differs by one bit in one of
  100 trials. The QAOA class is bit-for-bit identical to the published one in the tests, so the most
  likely cause is a different scipy version (L-BFGS-B was reimplemented in recent scipy); not verified.
