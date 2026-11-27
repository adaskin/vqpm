"""Writes the figure and table index of the repository README: for every figure and table of the paper, the
script that makes it, how to run it and the data it reads. Figure, table and section numbers are taken from the
labels of the compiled paper, so build the paper first. From the repository root:

    (cd paper && latexmk -pdf main.tex)
    python3 paper/update_figure_index.py [paper/main.aux]

The index is written between the FIGURE-INDEX markers of README.md. Paths are relative to the repository root.
"""

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TC = "theory_checks/"
EX = "experiments/"
START, END = "<!-- FIGURE-INDEX:START -->", "<!-- FIGURE-INDEX:END -->"

# label, short description, script(s) and command, data / output
ITEMS = [
    ("fig:Circuit Summary of VQPM", "the VQPM circuit and loop", "drawn in LaTeX", "part of the paper build",
     "paper/graphical.tex"),
    ("tab:params", "parameters of VQPM and the values used in the paper", "vqpm.py",
     "arguments of `vqpm_for_qubo` (see Main options above)", "values from the sections cited in the table"),
    ("fig:cosx", "cosine weight and its powers", "misc_plots.py", "`python3 misc_plots.py`", "paper/figures/cosx.pdf, cosxvsx.png"),
    ("fig-losing-eigenstate", "one VQPM run that ends in a nearby local minimum",
     "vqpm.py (single-run convergence plot)", "original single run of vqpm.py", "paper/figures/convergence-long.png"),
    ("fig:both-no-locking", "VQPM without locking, 100 runs at n = 15",
     "plot_qubit_vs_metrics.py", "`python3 plot_qubit_vs_metrics.py` (options in `__main__`: locking 'no', "
     "max_iter 500, seed 42)", "paper/figures/fixed-pdiff-no-locking[12].pdf"),
    ("fig:jumps-correct,fig:jumps-incorrect", "locking jumps, correct and incorrect locks (two panels)",
     "vqpm.py (single-run convergence plot)", "original single runs of vqpm.py", "paper/figures/jumps[12].png"),
    ("fig:both-pdiff-fixed", "VQPM with fixed p_diff = 0.01, three decimals, 100 runs at n = 15",
     "plot_qubit_vs_metrics.py", "options: locking 'yes', max_iter 30, file_name figures/fixed-pdiff-yes",
     "paper/figures/fixed-pdiff-yes[12].pdf"),
    ("fig:app-flow", "first-order update without and with locking, n = 20 to 100",
     TC + "large_n_flow.py, " + TC + "large_n_flow_report.py",
     "`python3 theory_checks/large_n_flow.py` (about 95 min on 4 cores), then `large_n_flow_report.py`",
     TC + "large_n_flow.json.gz; paper/figures/app_flow.pdf"),
    ("tab:app-lockflow", "locking thresholds on the first-order update, n = 20 to 100",
     TC + "large_n_flow_report.py", "`python3 theory_checks/large_n_flow_report.py`",
     TC + "large_n_flow_report.out"),
    ("fig:both-hoeffding", "VQPM with the Hoeffding threshold, 100 runs at n = 15", "plot_qubit_vs_metrics.py",
     "options: dynamic_pdiff_policy 'hoeffding', max_iter 30", "paper/figures/dynamic-pdiff-hoeffding[12].pdf"),
    ("fig:both-influence", "VQPM with influence scores, 100 runs at n = 15", "plot_qubit_vs_metrics.py",
     "options: dynamic_pdiff_policy 'hoeffding', compute_qubit_weights_in_Q True, max_iter 30",
     "paper/figures/dynamic-pdiff-hoeffding-with-weights[12].pdf"),
    ("tab:qaoa", "QAOA (four optimizers), VQPM and simulated annealing, n = 4 to 16",
     EX + "qaoa_vs_vqpm.py", "`python3 experiments/qaoa_vs_vqpm.py --n 4 6 8 10 12 14 16`",
     EX + "results/qaoa_vs_vqpm/qaoa_vs_vqpm_summary.csv; report " + EX + "reports/qaoa_vs_vqpm.md"),
    ("fig:qaoa", "the same runs, three metrics", TC + "qaoa_figure.py", "`python3 theory_checks/qaoa_figure.py`",
     EX + "results/qaoa_vs_vqpm/qaoa_vs_vqpm_summary.csv; paper/figures/qaoa_vs_vqpm_p8_summary.pdf"),
    ("fig:scaling", "found-optimum rate and iterations, n = 4 to 100",
     TC + "small_n_flow.py, " + TC + "scaling_summary.py",
     "`python3 theory_checks/small_n_flow.py` (about 20 min on 4 cores), then `scaling_summary.py`",
     TC + "small_n_flow.json.gz, large_n_flow.json.gz, large_n.json.gz, " + EX + "results/*/..._summary.csv; "
     "output " + TC + "scaling_summary.out, paper/figures/scaling_summary.pdf"),
    ("tab:cost", "cost per instance of all methods at n = 16 and n = 100",
     TC + "scaling_summary.py, " + TC + "small_n_flow.py", "`python3 theory_checks/scaling_summary.py` (after `small_n_flow.py`); device rows from "
     "the table of finite shots", TC + "scaling_summary.out, small_n_flow.out; " + EX + "reports/finite_shots.md"),
    ("tab:shots", "VQPM with M shots per iteration, n = 8 to 16", EX + "finite_shots.py",
     "`python3 experiments/finite_shots.py --n 8 10 12 14 16 --instances 100`", EX + "reports/finite_shots.md"),
    ("tab:gates", "size of the transpiled controlled-U", "vqpm_qiskit_noisy/vqpm_qiskit.py",
     "`python3 vqpm_qiskit_noisy/vqpm_qiskit.py --gate-counts 8 12 16 --gate-count-backends all-to-all FakeTorino "
     "FakeSherbrooke`", EX + "results/gate_counts/controlled_u_gate_counts.json; report " + EX + "reports/gate_counts.md"),
    ("tab:noise", "circuit VQPM under gate and readout noise (Qiskit Aer)",
     TC + "noise_rerun.py (uses " + EX + "noisy_runs.py), " + TC + "noise_figure.py",
     "`python3 theory_checks/noise_rerun.py` (about 1 h on 4 cores), then `noise_figure.py`",
     TC + "noise_rerun.jsonl, noise_figure.out; p_diff = 0.1 rows also in " + EX + "reports/noise_sweep_n8.md, "
     "depol_torino.md, faketorino_n8.md"),
    ("fig:noisy", "marginal differences and locks under noise", TC + "noise_figure.py",
     "`python3 theory_checks/noise_figure.py`", TC + "noise_rerun.jsonl; paper/figures/noisy_locking.pdf"),
    ("tab:app-phase", "phase map, rates and coherent-iteration counts", TC + "theory_checks.py",
     "`python3 theory_checks/theory_checks.py` (more than half an hour on one core; the SDP table needs cvxpy)", TC + "theory_checks.out"),
    ("tab:app-windows", "contrast and acceptance of four phase windows", TC + "theory_checks.py", "`python3 theory_checks/theory_checks.py`",
     TC + "theory_checks.out"),
    ("tab:app-exact", "exact VQPM update without locking", TC + "theory_checks.py", "`python3 theory_checks/theory_checks.py`", TC + "theory_checks.out"),
    ("tab:app-locking", "locking, rounding and wrong locks", TC + "theory_checks.py", "`python3 theory_checks/theory_checks.py`",
     TC + "theory_checks.out"),
    ("tab:app-sdp", "Shor relaxation and moment matrix", TC + "theory_checks.py", "`python3 theory_checks/theory_checks.py`", TC + "theory_checks.out"),
    ("tab:app-shots", "device shots against classical draws per iteration", TC + "shot_counts.py",
     "`python3 theory_checks/shot_counts.py`", TC + "shot_counts.out"),
    ("tab:app-round1", "first-iteration differences, n = 20 to 100",
     TC + "large_n.py, " + TC + "large_n_sweep.py, " + TC + "large_n_report.py",
     "`large_n.py` (about 40 min on 4 cores), `large_n_sweep.py` (about an hour), then `large_n_report.py`",
     TC + "large_n.json.gz, large_n_sweep.json.gz; " + TC + "large_n_report.out"),
    ("tab:app-large", "VQPM by classical sampling, n = 20 to 100", TC + "large_n_report.py", "`python3 theory_checks/large_n_report.py` (after `large_n.py` and `large_n_sweep.py`)",
     TC + "large_n_report.out"),
    ("fig:app-large", "first-iteration differences and outcomes, n = 20 to 100", TC + "large_n_report.py", "`python3 theory_checks/large_n_report.py` (after `large_n.py` and `large_n_sweep.py`)",
     "paper/figures/app_large_n.pdf"),
    ("tab:app-flow", "first-order update without locking, n = 20 to 100", TC + "large_n_flow_report.py",
     "`python3 theory_checks/large_n_flow_report.py` (after `large_n_flow.py`)", TC + "large_n_flow_report.out"),
    ("tab:app-read", "signs read by locking thresholds", TC + "large_n_flow_report.py", "`python3 theory_checks/large_n_flow_report.py` (after `large_n_flow.py`)",
     TC + "large_n_flow_report.out"),
    ("tab:app-sweep", "sampled VQPM without rounding for small thresholds", TC + "large_n_report.py", "`python3 theory_checks/large_n_report.py` (after `large_n.py` and `large_n_sweep.py`)",
     TC + "large_n_report.out"),
]


def labels(aux=None):
    num = {}
    aux = aux or os.path.join(HERE, "main.aux")
    if not os.path.exists(aux):
        return num
    for line in open(aux):
        m = re.match(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", line)
        if m:
            num[m.group(1)] = m.group(2)
    return num


def kind(lab):
    return "Table" if lab.startswith("tab") else "Fig."


def numbers(lab, num):
    return ", ".join(num.get(l, "?") for l in lab.split(","))



TEXT = [
    ("sec:qpi", "phase maps", EX + "phase_map_ablation.py", EX + "reports/phase_map.md"),
    ("sec:locking", "iteration budget and rounding", EX + "vqpm_grid.py", EX + "reports/vqpm_grid.md"),
    ("sec:incorrectlocking", "wrong locks", EX + "vqpm_grid.py, " + EX + "wrong_locks.py", EX + "reports/wrong_locks.md"),
    ("sec:gradient", "first-order against exact update", TC + "meanfield.py", TC + "meanfield.out"),
    ("sec:smalln", "QAOA optimizer and learning-rate pilot", EX + "qaoa_vs_vqpm.py", EX + "reports/qaoa_optimizer_pilot.md"),
    ("sec:gatenoise", "noise runs with p_diff = 0.1", EX + "noisy_runs.py",
     EX + "reports/noise_sweep_n8.md, depol_torino.md, faketorino_n8.md"),
]


def index_markdown(num):
    md = [START, "",
          "| item | content | script | how to run | data and output |", "|---|---|---|---|---|"]
    def order(item):
        lab = item[0]
        first = num.get(lab.split(",")[0], "")
        m = re.match(r"\d+", first)
        return (kind(lab) != "Fig.", int(m.group()) if m else 10**6)

    for lab, what, script, cmd, data in sorted(ITEMS, key=order):
        md.append("| %s %s | %s | %s | %s | %s |" % (kind(lab), numbers(lab, num), what, script, cmd, data))
    md += ["", "Numbers quoted in the text:", "", "| section | content | script | output or report |", "|---|---|---|---|"]
    for lab, what, script, rep in TEXT:
        md.append("| Sec. %s | %s | %s | %s |" % (num.get(lab, "?"), what, script, rep))
    md += ["", END]
    return "\n".join(md)


if __name__ == "__main__":
    import sys
    num = labels(sys.argv[1] if len(sys.argv) > 1 else None)
    readme = os.path.join(ROOT, "README.md")
    text = open(readme).read()
    if START not in text or END not in text:
        sys.exit("README.md has no %s ... %s block" % (START, END))
    a, b = text.index(START), text.index(END) + len(END)
    open(readme, "w").write(text[:a] + index_markdown(num) + text[b:])
    missing = sum(any(l not in num for l in lab.split(",")) for lab, *_ in ITEMS)
    print("wrote the figure index of README.md (%d items, %d without a number)" % (len(ITEMS), missing))
