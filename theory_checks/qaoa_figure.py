"""Figure: QAOA (p = 8, four optimizers), VQPM and simulated annealing for n = 4..16.

Run from the repository root:
    python3 theory_checks/qaoa_figure.py
Reads experiments/results/qaoa_vs_vqpm/qaoa_vs_vqpm_summary.csv (report experiments/reports/qaoa_vs_vqpm.md) and writes
paper/figures/qaoa_vs_vqpm_p8_summary.pdf. No new runs.
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "paper", "figures", "qaoa_vs_vqpm_p8_summary.pdf")
R = list(csv.DictReader(open(os.path.join(HERE, "..", "experiments", "results", "qaoa_vs_vqpm", "qaoa_vs_vqpm_summary.csv"))))


def series(**match):
    out = {int(r["n"]): r for r in R if all(r[k] == v for k, v in match.items())}
    return dict(sorted(out.items()))


INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"
import figstyle  # noqa: E402
figstyle.apply()
# fixed categorical order: blue, orange, aqua, yellow, magenta, green (validated, light surface)
C = {"vqpm": "#257de0", "L-BFGS-B": "#f47230", "sa": "#00b894", "COBYLA": "#f2a600", "gd": "#ee7fac",
     "adam": "#008a17"}
NAME = {"L-BFGS-B": "L-BFGS-B", "COBYLA": "COBYLA", "gd": "gradient descent", "adam": "Adam"}

lines = []
for opt, mk in [("L-BFGS-B", "s"), ("adam", "v"), ("COBYLA", "D"), ("gd", "P")]:
    for init, ls, mfc in [("random", "-", None), ("interp", "--", "white")]:
        lines.append((series(kind="qaoa", variant=opt, init_or_budget=init, p="8"), C[opt], ls, mk, mfc,
                      "QAOA, %s, %s" % (NAME[opt], "restarts" if init == "random" else "INTERP"),
                      opt == "L-BFGS-B" and init == "random"))
lines.append((series(kind="vqpm", variant="oracle", init_or_budget="100"), C["vqpm"], "-", "o", None,
              "VQPM, oracle rule", True))
lines.append((series(kind="vqpm", variant="max_prob", init_or_budget="100"), C["vqpm"], "--", "o", "white",
              "VQPM, max-prob rule", False))
lines.append((series(kind="sa", variant="100"), C["sa"], "-", "^", None, "simulated annealing, 100 sweeps", False))

panels = [("found_optimum_rate", "found-optimum rate", "(a)"), ("mean_bit_diff", "mean bit difference", "(b)"),
          ("mean_norm_energy_gap", "mean normalized energy gap", "(c)")]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.3))
handles = []
for ax, (key, ylab, tag) in zip(axes, panels):
    for d, col, ls, mk, mfc, lab, err in lines:
        ns = list(d)
        y = [float(d[n][key]) for n in ns]
        kw = dict(color=col, ls=ls, lw=1.8 if ls == "-" else 1.2, marker=mk, ms=4, label=lab)
        if mfc:
            kw["mfc"] = mfc
        if err:
            base = key[:-5] if key.endswith("_rate") else key
            lo = [float(d[n][key]) - float(d[n][base + "_lo"]) for n in ns]
            hi = [float(d[n][base + "_hi"]) - float(d[n][key]) for n in ns]
            hd = ax.errorbar(ns, y, yerr=[lo, hi], capsize=2, elinewidth=0.8, **kw)
        else:
            hd = ax.plot(ns, y, **kw)[0]
        if ax is axes[0]:
            handles.append((hd, lab))
    ax.set_xticks([4, 8, 12, 16])
    ax.set_xlabel("$n$")
    ax.set_ylabel(ylab)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_title(tag, loc="left", fontsize=9, color=INK)
axes[0].set_ylim(0.25, 1.03)
fig.legend([h for h, _ in handles], [lab for _, lab in handles], loc="lower center", ncol=3, handlelength=2.6, columnspacing=1.2)
fig.tight_layout(rect=(0, 0.25, 1, 1))
fig.savefig(FIG)
print("wrote", os.path.relpath(FIG, os.path.join(HERE, "..")))
