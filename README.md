# Variational Quantum Power Method (VQPM) for QUBO problems

Code, data and LaTeX source of the paper

> A. Daskin, *From Theory to Practice: Analyzing Variational Quantum Power Method for Quantum Optimization of
> QUBO Problems*, [arXiv:2505.12990](https://arxiv.org/abs/2505.12990).

Every figure and table of the paper can be regenerated from this repository. The
[figure and table index](#figure-and-table-index) below lists, for each of them, the script, the command and the
data it reads.

## What VQPM does

VQPM looks for the minimum of a QUBO energy $E(x)=\sum_{i\le j}Q_{ij}x_ix_j$, $x\in\{0,1\}^n$.

1. **Phase map.** Every energy is mapped to an eigenphase of a diagonal unitary $\mathcal U$,
   $\lambda(x)=\pi/4+(\pi/4)\,E(x)/S$ with $S=\sum_{i\le j}|Q_{ij}|$, so that all phases lie in $[0,\pi/2]$ and
   lower energy means a smaller phase.
2. **One iteration.** A product state with single-qubit marginals $\pi_i=\Pr(x_i=1)$ is prepared with $R_y$
   gates. A control qubit in $\ket{+}$ applies $\mathcal U$, and a Hadamard on the control followed by the outcome
   $0$ applies $(I+\mathcal U)/2$. This weights each string by $\cos^2(\lambda(x)/2)$, which favours low energies.
   The marginals of the data qubits are measured on the shots with control $0$.
3. **Update.** The next input is the product state with the measured marginals.
4. **Locking.** When $|P_0-P_1|$ of a qubit exceeds the locking threshold $p_{diff}$, the qubit is fixed to its
   more likely value. This cuts the number of iterations from about $10^3$ to tens.

Because $\mathcal U$ is diagonal and the input is a product state, every shot can also be drawn classically at
$O(n^2)$ cost. The repository uses this to run VQPM up to $n=100$ (`theory_checks/`), and it contains a
first-order form of the update, which is mirror descent on the mean energy of the product distribution.

## Repository layout

| path | content |
|---|---|
| `vqpm.py` | VQPM simulator with state vectors: phase map, update, locking rules, Hoeffding thresholds, finite shots, stop rules. Run alone it solves one random instance. |
| `qubo.py` | QUBO helpers (energies, random instances), re-exported from `vqpm.py` |
| `baselines.py` | simulated annealing |
| `comparison_to_qaoa.py` | state-vector QAOA with L-BFGS-B, COBYLA, gradient descent or Adam, exact adjoint gradients, random restarts and INTERP initialisation; also an older small QAOA comparison (`main1`, `main2`) whose output is in `figures/` |
| `run_multiple.py`, `plot_qubit_vs_metrics.py` | many random instances at once; the run figures of Sec. 2 of the paper |
| `misc_plots.py` | figure of the weight $\cos^2(\lambda/2)$ |
| `vqpm_qiskit_noisy/` | VQPM as a Qiskit circuit: ideal and noisy (Aer) runs, transpiled gate counts. See its [README](vqpm_qiskit_noisy/README.md). |
| `experiments/` | experiment scripts with resumable output, the stored instances, result tables and one report per experiment. See its [README](experiments/README.md). |
| `theory_checks/` | scripts behind the appendices, the classical-sampling runs up to $n=100$ and several figures, each with its printed output. See its [README](theory_checks/README.md). |
| `paper/` | LaTeX source and figures of the paper, and the script that writes the index below. See its [README](paper/README.md). |
| `tests/` | pytest suite; `tests/reference/` holds a frozen copy of the original `vqpm.py` (commit d1e1d24) used to check that the default code path is unchanged |
| `figures/` | output of `comparison_to_qaoa.py main1/main2` (not used in the paper) |
| `vqpm_maxcut/` | a Max-Cut variant of VQPM with a moment-matrix certificate (not used in the paper) |

## Installation

Python 3.12 was used. The direct dependencies are pinned in `requirements.txt`, the full environment in
`requirements-lock.txt`.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests        # a few seconds
```

Qiskit is needed only for `vqpm_qiskit_noisy/` and the noise runs. One table of the theory appendix (the SDP
bound in `theory_checks/theory_checks.py`) also needs `cvxpy`; without it that part is skipped.

## Quick start

```python
import numpy as np
from vqpm import random_qubo, adjust_unitary_phase, vqpm_for_qubo

np.random.seed(1)
n = 10
Q = random_qubo(n)                                   # Q = G + G^T, G_ij ~ N(0, 1)
Q_scaled, u, phases, target = adjust_unitary_phase(n, Q)   # u: diagonal of U; target: index of the optimum

state, max_prob, q_states, p0, iters, probs = vqpm_for_qubo(
    u, n,
    expected_idx=target,     # used only for reporting with stop_rule="max_prob"
    max_iter=30, pdiff=0.01, precision=3,
    stop_rule="max_prob",    # stop when one string has probability >= 1/2; no knowledge of the optimum
    verbose=False,
)
print(np.binary_repr(state, n), state == target)
```

`python vqpm.py` runs a similar example and prints a summary.

### Main options of `vqpm_for_qubo`

| option | meaning | default |
|---|---|---|
| `max_iter` | iteration budget $T$; an integer or a string such as `"10n"` | 20 |
| `pdiff` | locking threshold $p_{diff}$ | 0.01 |
| `precision` | decimals of the measured marginals; `None` keeps them exact | 3 |
| `locking` | `"yes"`, `"no"` or `"only_when_p0_increases"` | `"yes"` |
| `dynamic_pdiff_policy` | `None` (fixed threshold), `"hoeffding"` (threshold that changes with the iteration), `"scaledby2"`, or `"hoeffding_union"` (union-bound threshold, needs `shots`) | `None` |
| `qubit_weights` | per-qubit scale of the threshold, e.g. `compute_qubit_weights(Q)` (influence scores) | `None` |
| `stop_rule` | `"oracle"` (stop when the known optimum `expected_idx` reaches probability 1/2), `"max_prob"` (no oracle) or `"none"` | `"oracle"` |
| `shots`, `rng` | estimate the marginals from this many circuit runs per iteration, post-selected on control 0 | `None` (exact) |
| `return_log` | also return a dict with the lock log, the iterations used, the shots used and the stop reason | `False` |

The defaults reproduce the original code exactly; the tests check this against `tests/reference/`.

## Reproducing the paper

The scripts run from the repository root. Most of the figures and tables are made from stored results in a few
seconds; the commands that simulate say so and give their run time.

**Instances.** All instances are random QUBOs $Q=G+G^T$ with $G_{ij}\sim\mathcal N(0,1)$, from three seeds:

- the run figures of Sec. 2 (`plot_qubit_vs_metrics.py`) use `np.random.seed(42)`, as in the original code;
- the experiments of Secs. 4 and 5 use `experiments/common.get_instances(n, count, 2026)`, stored in
  `experiments/instances/`, so that VQPM, QAOA and simulated annealing see the same instances;
- the theory checks and the classical-sampling runs use `random_qubo` with seed $2026+n$.

**Long runs.** The raw outputs of the long runs are committed (gzipped where large), so the figures and tables
can be remade without them. The longest is the QAOA comparison (about 34 hours of wall time on a workstation).
The iteration-budget grid and the noisy Qiskit runs take one to three hours each, and the classical-sampling runs
to $n=100$ about 40 to 95 minutes each on 4 cores. The experiment reports give the wall time of every run.

**Paper.** `cd paper && latexmk -pdf main.tex`. After a build, `python3 paper/update_figure_index.py`
refreshes the figure and table numbers in the index below.

### Figure and table index

The original scripts write their figures into `figures/`; the copies used by the paper are in `paper/figures/`.
Experiment reports in `experiments/reports/` give the method, the parameters and the raw files of each experiment.

<!-- FIGURE-INDEX:START -->

| item | content | script | how to run | data and output |
|---|---|---|---|---|
| Fig. 1 | the VQPM circuit and loop | drawn in LaTeX | part of the paper build | paper/graphical.tex |
| Fig. 2 | cosine weight and its powers | misc_plots.py | `python3 misc_plots.py` | paper/figures/cosx.pdf, cosxvsx.png |
| Fig. 3 | one VQPM run that ends in a nearby local minimum | vqpm.py (single-run convergence plot) | original single run of vqpm.py | paper/figures/convergence-long.png |
| Fig. 4 | VQPM without locking, 100 runs at n = 15 | plot_qubit_vs_metrics.py | `python3 plot_qubit_vs_metrics.py` (options in `__main__`: locking 'no', max_iter 500, seed 42) | paper/figures/fixed-pdiff-no-locking[12].pdf |
| Fig. 5a, 5b | locking jumps, correct and incorrect locks (two panels) | vqpm.py (single-run convergence plot) | original single runs of vqpm.py | paper/figures/jumps[12].png |
| Fig. 6 | VQPM with fixed p_diff = 0.01, three decimals, 100 runs at n = 15 | plot_qubit_vs_metrics.py | options: locking 'yes', max_iter 30, file_name figures/fixed-pdiff-yes | paper/figures/fixed-pdiff-yes[12].pdf |
| Fig. 7 | VQPM with the Hoeffding threshold, 100 runs at n = 15 | plot_qubit_vs_metrics.py | options: dynamic_pdiff_policy 'hoeffding', max_iter 30 | paper/figures/dynamic-pdiff-hoeffding[12].pdf |
| Fig. 8 | VQPM with influence scores, 100 runs at n = 15 | plot_qubit_vs_metrics.py | options: dynamic_pdiff_policy 'hoeffding', compute_qubit_weights_in_Q True, max_iter 30 | paper/figures/dynamic-pdiff-hoeffding-with-weights[12].pdf |
| Fig. 9 | first-order update without and with locking, n = 20 to 100 | theory_checks/large_n_flow.py, theory_checks/large_n_flow_report.py | `python3 theory_checks/large_n_flow.py` (about 95 min on 4 cores), then `large_n_flow_report.py` | theory_checks/large_n_flow.json.gz; paper/figures/app_flow.pdf |
| Fig. 10 | the same runs, three metrics | theory_checks/qaoa_figure.py | `python3 theory_checks/qaoa_figure.py` | experiments/results/qaoa_vs_vqpm/qaoa_vs_vqpm_summary.csv; paper/figures/qaoa_vs_vqpm_p8_summary.pdf |
| Fig. 11 | found-optimum rate and iterations, n = 4 to 100 | theory_checks/small_n_flow.py, theory_checks/scaling_summary.py | `python3 theory_checks/small_n_flow.py` (about 20 min on 4 cores), then `scaling_summary.py` | theory_checks/small_n_flow.json.gz, large_n_flow.json.gz, large_n.json.gz, experiments/results/*/..._summary.csv; output theory_checks/scaling_summary.out, paper/figures/scaling_summary.pdf |
| Fig. 12 | marginal differences and locks under noise | theory_checks/noise_figure.py | `python3 theory_checks/noise_figure.py` | theory_checks/noise_rerun.jsonl; paper/figures/noisy_locking.pdf |
| Fig. 13 | first-iteration differences and outcomes, n = 20 to 100 | theory_checks/large_n_report.py | `python3 theory_checks/large_n_report.py` (after `large_n.py` and `large_n_sweep.py`) | paper/figures/app_large_n.pdf |
| Table 1 | locking thresholds on the first-order update, n = 20 to 100 | theory_checks/large_n_flow_report.py | `python3 theory_checks/large_n_flow_report.py` | theory_checks/large_n_flow_report.out |
| Table 2 | QAOA (four optimizers), VQPM and simulated annealing, n = 4 to 16 | experiments/qaoa_vs_vqpm.py | `python3 experiments/qaoa_vs_vqpm.py --n 4 6 8 10 12 14 16` | experiments/results/qaoa_vs_vqpm/qaoa_vs_vqpm_summary.csv; report experiments/reports/qaoa_vs_vqpm.md |
| Table 3 | VQPM with M shots per iteration, n = 8 to 16 | experiments/finite_shots.py | `python3 experiments/finite_shots.py --n 8 10 12 14 16 --instances 100` | experiments/reports/finite_shots.md |
| Table 4 | size of the transpiled controlled-U | vqpm_qiskit_noisy/vqpm_qiskit.py | `python3 vqpm_qiskit_noisy/vqpm_qiskit.py --gate-counts 8 12 16 --gate-count-backends all-to-all FakeTorino FakeSherbrooke` | experiments/results/gate_counts/controlled_u_gate_counts.json; report experiments/reports/gate_counts.md |
| Table 5 | circuit VQPM under gate and readout noise (Qiskit Aer) | theory_checks/noise_rerun.py (uses experiments/noisy_runs.py), theory_checks/noise_figure.py | `python3 theory_checks/noise_rerun.py` (about 1 h on 4 cores), then `noise_figure.py` | theory_checks/noise_rerun.jsonl, noise_figure.out; p_diff = 0.1 rows also in experiments/reports/noise_sweep_n8.md, depol_torino.md, faketorino_n8.md |
| Table 6 | cost per instance of all methods at n = 16 and n = 100 | theory_checks/scaling_summary.py, theory_checks/small_n_flow.py | `python3 theory_checks/scaling_summary.py` (after `small_n_flow.py`); device rows from the table of finite shots | theory_checks/scaling_summary.out, small_n_flow.out; experiments/reports/finite_shots.md |
| Table 7 | parameters of VQPM and the values used in the paper | vqpm.py | arguments of `vqpm_for_qubo` (see Main options above) | values from the sections cited in the table |
| Table 8 | phase map, rates and coherent-iteration counts | theory_checks/theory_checks.py | `python3 theory_checks/theory_checks.py` (more than half an hour on one core; the SDP table needs cvxpy) | theory_checks/theory_checks.out |
| Table 9 | contrast and acceptance of four phase windows | theory_checks/theory_checks.py | `python3 theory_checks/theory_checks.py` | theory_checks/theory_checks.out |
| Table 10 | exact VQPM update without locking | theory_checks/theory_checks.py | `python3 theory_checks/theory_checks.py` | theory_checks/theory_checks.out |
| Table 11 | locking, rounding and wrong locks | theory_checks/theory_checks.py | `python3 theory_checks/theory_checks.py` | theory_checks/theory_checks.out |
| Table 12 | Shor relaxation and moment matrix | theory_checks/theory_checks.py | `python3 theory_checks/theory_checks.py` | theory_checks/theory_checks.out |
| Table 13 | device shots against classical draws per iteration | theory_checks/shot_counts.py | `python3 theory_checks/shot_counts.py` | theory_checks/shot_counts.out |
| Table 14 | first-iteration differences, n = 20 to 100 | theory_checks/large_n.py, theory_checks/large_n_sweep.py, theory_checks/large_n_report.py | `large_n.py` (about 40 min on 4 cores), `large_n_sweep.py` (about an hour), then `large_n_report.py` | theory_checks/large_n.json.gz, large_n_sweep.json.gz; theory_checks/large_n_report.out |
| Table 15 | VQPM by classical sampling, n = 20 to 100 | theory_checks/large_n_report.py | `python3 theory_checks/large_n_report.py` (after `large_n.py` and `large_n_sweep.py`) | theory_checks/large_n_report.out |
| Table 16 | first-order update without locking, n = 20 to 100 | theory_checks/large_n_flow_report.py | `python3 theory_checks/large_n_flow_report.py` (after `large_n_flow.py`) | theory_checks/large_n_flow_report.out |
| Table 17 | signs read by locking thresholds | theory_checks/large_n_flow_report.py | `python3 theory_checks/large_n_flow_report.py` (after `large_n_flow.py`) | theory_checks/large_n_flow_report.out |
| Table 18 | sampled VQPM without rounding for small thresholds | theory_checks/large_n_report.py | `python3 theory_checks/large_n_report.py` (after `large_n.py` and `large_n_sweep.py`) | theory_checks/large_n_report.out |

Numbers quoted in the text:

| section | content | script | output or report |
|---|---|---|---|
| Sec. 1.1 | phase maps | experiments/phase_map_ablation.py | experiments/reports/phase_map.md |
| Sec. 2.3 | iteration budget and rounding | experiments/vqpm_grid.py | experiments/reports/vqpm_grid.md |
| Sec. 2.4 | wrong locks | experiments/vqpm_grid.py, experiments/wrong_locks.py | experiments/reports/wrong_locks.md |
| Sec. 3.2 | first-order against exact update | theory_checks/meanfield.py | theory_checks/meanfield.out |
| Sec. 4.1 | QAOA optimizer and learning-rate pilot | experiments/qaoa_vs_vqpm.py | experiments/reports/qaoa_optimizer_pilot.md |
| Sec. 5.3 | noise runs with p_diff = 0.1 | experiments/noisy_runs.py | experiments/reports/noise_sweep_n8.md, depol_torino.md, faketorino_n8.md |

<!-- FIGURE-INDEX:END -->

## Citation

```bibtex
@article{daskin2025vqpm,
  title   = {From Theory to Practice: Analyzing Variational Quantum Power Method for Quantum Optimization of QUBO Problems},
  author  = {Daskin, Ammar},
  journal = {arXiv preprint arXiv:2505.12990},
  year    = {2025}
}
```

## License and acknowledgment

Apache License 2.0, see `LICENSE`. Parts of the code and of the documentation were written with the help of AI
assistants.
