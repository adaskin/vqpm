# Theory checks and classical runs

Scripts behind the appendices of the paper, the classical runs of VQPM up to $n=100$ (Sec. 3 and Sec. 4.2) and
several figures. Each script prints its results; the printed output is committed next to it as `<script>.out`, so
every number can be checked without running anything. Run the scripts from the repository root, for example

```bash
python3 theory_checks/theory_checks.py > theory_checks/theory_checks.out
```

Large raw outputs are committed gzipped (`*.json.gz`). The scripts read either the `.json` or the `.json.gz`
file, and the uncompressed copies are ignored by git. Figures are written to `paper/figures/` with the shared
style of `figstyle.py` (matplotlib defaults, DejaVu Sans).

## Instances

- `theory_checks.py`, `shot_counts.py`, `meanfield.py`, `large_n*.py`: `random_qubo(n)` of `vqpm.py` after
  `np.random.seed(2026 + n)`, 100 instances per $n$.
- `small_n_flow.py`, `noise_rerun.py`: the seed-2026 instances of `experiments/` (the instances of the QAOA
  comparison).
- Both draw $Q=G+G^T$ with $G_{ij}\sim\mathcal N(0,1)$.

For $n\ge30$ the reference string $x_{\mathrm{ref}}$ is the best of 32 simulated-annealing chains, each followed by
1-flip descent; for $n\le22$ it is the exact optimum.

## Scripts

| script | what it computes | output | run time |
|---|---|---|---|
| `theory_checks.py` | phase map and its window, acceptance rates, coherent power iteration against VQPM, the exact update without locking, locking and wrong locks, the Hoeffding threshold, the Shor relaxation and moment matrix (needs `cvxpy`) | `theory_checks.out` | more than half an hour on one core |
| `shot_counts.py` | shots on a device against classical draws per iteration | `shot_counts.out` | |
| `meanfield.py` | first-order (mirror-descent) form of the VQPM update and locking on top of it; checks it against the exact update at n = 8 and 12 | `meanfield.out` | |
| `large_n.py` | VQPM by classical sampling at n = 20 to 100 (no $2^n$ state vector) | `large_n.json.gz`, `large_n.out` | about 40 min on 4 cores |
| `large_n_sweep.py` | the same without rounding, for locking thresholds from $10^{-8}$ to $10^{-2}$ | `large_n_sweep.json.gz`, `large_n_sweep.out` | about an hour on 4 cores |
| `large_n_report.py` | tables and figure of the two scripts above | `large_n_report.out`, `paper/figures/app_large_n.pdf` | seconds |
| `large_n_flow.py` | first-order update without locking and with thresholds from $10^{-6}$ to 0.2, n = 20 to 100 | `large_n_flow.json.gz`, `large_n_flow.out` | about 95 min on 4 cores |
| `large_n_flow_report.py` | tables and figure of the first-order runs, compared with the sampled runs | `large_n_flow_report.out`, `paper/figures/app_flow.pdf` | seconds |
| `small_n_flow.py` | first-order update (all n) and exact update (n up to 14) without and with locking on the QAOA-comparison instances, n = 4 to 22 | `small_n_flow.json.gz`, `small_n_flow.out` | about 20 min on 4 cores |
| `scaling_summary.py` | found-optimum rate, iterations and cost per instance of VQPM, QAOA and simulated annealing from n = 4 to 100 | `scaling_summary.out`, `paper/figures/scaling_summary.pdf` | seconds |
| `qaoa_figure.py` | figure of the QAOA comparison from `experiments/results/qaoa_vs_vqpm/` | `paper/figures/qaoa_vs_vqpm_p8_summary.pdf` | seconds |
| `noise_rerun.py` | circuit VQPM under noise with $p_{diff}=0.01$ (Qiskit Aer, calls `experiments/noisy_runs.py`) | `noise_rerun.jsonl` | about an hour on 4 cores |
| `noise_figure.py` | table and figure of the noise study | `noise_figure.out`, `paper/figures/noisy_locking.pdf` | seconds |
| `figstyle.py` | shared matplotlib style | | |

The order in which the long runs depend on each other: `large_n_sweep.py` fixes the reference strings that
`large_n_flow.py` reads, and `scaling_summary.py` reads the outputs of `small_n_flow.py`, `large_n_flow.py`,
`large_n.py` and the experiment summaries.
