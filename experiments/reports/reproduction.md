# Reproduction of the original figures

_Report generated 2026-11-27T07:51:15 by `experiments/reports.py`; status: finished._

## Purpose

Check that the code with default options reproduces the original figures made with plot_qubit_vs_metrics.py and comparison_to_qaoa.py.

## Method

Fig. 6a: np.random.seed(42), 100 random QUBOs (vqpm.random_qubo), n = 15, pdiff 0.01, precision 3, max_iter 30, locking 'yes'. Scaling figure (VQPM half): seed 42, n = 4..16 step 2, 10 instances, QAOA's RNG draws np.random.rand(2p) for p = 2, 4, 6, 8 mimicked. Fig. 9: comparison_to_qaoa.main1 without plotting (legacy_instances=True). 'legacy' runs the frozen published vqpm.py (tests/reference).

Metrics (per run, energies in the units of the original, unscaled Q): found optimum = the returned bitstring is the exact ground state (found by enumerating all 2^n energies); bit difference = Hamming distance to the ground state; energy gap = E_found - E_min; normalised energy gap = (E_found - E_min)/(E_max - E_min) in [0, 1]; normalised expected-energy gap = (<E> - E_min)/(E_max - E_min) with <E> the expected energy of the final output distribution (VQPM, QAOA). Intervals: Wilson 95% for rates, mean +- 1.96 s/sqrt(N) for means.

## Provenance

- Library versions (pinned in requirements.txt): matplotlib 3.11.2, numpy 2.5.5, pytest 9.2.0, qiskit 2.6.2, qiskit-aer 0.18.4, qiskit-ibm-runtime 0.52.1, scipy 1.18.1, python 3.12.12
- Machine: AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 16 logical CPUs, 90 GiB RAM, Linux 6.12.113+deb13-amd64
- End time (last write of the raw data): `reproduce_published_current.json` 2026-11-25T17:18:43, `reproduce_published_legacy_d1e1d24.json` 2026-11-25T17:18:50, `reproduce_fig9_current.json` 2026-11-25T17:44:56

## Where the data is

Paths are relative to the repository root.

| file (repo-relative) | records / size | sha256 |
|---|---|---|
| `experiments/results/reproduction/reproduce_published_current.json` | 1.4 KiB | `39042703446d0f1717270775015e90b8dad599e110e4fb9f8225a2a2acea8a37` |
| `experiments/results/reproduction/reproduce_published_legacy_d1e1d24.json` | 1.4 KiB | `911bce982e96ce3bc0f923be217497767c1768eb901c672265996ffb18b50582` |
| `experiments/results/reproduction/reproduce_fig9_current.json` | 0.2 KiB | `994ed93ae94061f62288f43dde8494e6ba193b5b0571c6a182ae4580c0450631` |

Summary tables and plots (committed to git unless noted):

- `experiments/results/reproduction/reproduce_fig9_current.json`
- `experiments/results/reproduction/reproduce_published_current.json`
- `experiments/results/reproduction/reproduce_published_legacy_d1e1d24.json`

Raw JSONL files are not in the repository because of their size. The scripts write them to `experiments/results/<experiment>/raw/`, and the summary tables above are computed from them.

## Results

- `reproduce_published_current.json` (current): Fig. 6a mean bit difference 1.1600 (published 1.16)
  scaling-figure VQPM mean final target probability: n=4: 0.4024, n=6: 0.3524, n=8: 0.4022, n=10: 0.2132, n=12: 0.4645, n=14: 0.4008, n=16: 0.1755 (published 0.46 / 0.40 / 0.18 at n = 12 / 14 / 16)
- `reproduce_published_legacy_d1e1d24.json` (legacy): Fig. 6a mean bit difference 1.1600 (published 1.16)
  scaling-figure VQPM mean final target probability: n=4: 0.4024, n=6: 0.3524, n=8: 0.4022, n=10: 0.2132, n=12: 0.4645, n=14: 0.4008, n=16: 0.1755 (published 0.46 / 0.40 / 0.18 at n = 12 / 14 / 16)
- `reproduce_fig9_current.json`: Fig. 9 VQPM prob 0.4346 (0.43), QAOA prob 0.5495 (0.55), VQPM bits 0.43 (0.43), QAOA bits 0.32 (0.33)

## Findings and caveats

- The current code with default options reproduces the published numbers exactly, trial by trial
  (identical to the frozen published vqpm.py in tests/reference, including printed output):
  Fig. 6a mean bit difference 1.16; scaling figure 0.4645 / 0.4008 / 0.1755 at n = 12 / 14 / 16
  (published 0.46 / 0.40 / 0.18).
- Fig. 9: VQPM numbers match (final probability 0.4346 vs 0.43, bit difference 0.43 vs 0.43); QAOA final
  probability 0.5495 vs 0.55 matches; QAOA bit difference 0.32 vs 0.33 differs by one bit in one of
  100 trials. The QAOA class is bit-for-bit identical to the published one in the tests, so the most
  likely cause is a different scipy version (L-BFGS-B was reimplemented in recent scipy); not verified.

## Regenerate tables and plots from the saved data (no re-run)

```bash
.venv/bin/python experiments/reproduce_published.py --impl current
.venv/bin/python experiments/reproduce_published.py --impl legacy
.venv/bin/python experiments/reproduce_published.py --impl current --checks fig9
.venv/bin/python experiments/reports.py
```
