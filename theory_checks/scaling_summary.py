"""Found-optimum rate and iterations of VQPM, QAOA and simulated annealing from n = 4 to 100.

Run from the repository root:
    python3 theory_checks/scaling_summary.py > theory_checks/scaling_summary.out
Writes paper/figures/scaling_summary.pdf.

Sources (no new runs):
  experiments/results/{qaoa_vs_vqpm,vqpm_grid}/*_summary.csv
      (instance seed 2026; reports experiments/reports/qaoa_vs_vqpm.md, vqpm_grid.md).
  small_n_flow.json        first-order update with and without locking, n = 4..22 (seed-2026 instances).
  large_n_flow.json(.gz)   first-order update with and without locking, n = 20..100 (seed 2026 + n).
  large_n.json(.gz)        sampled VQPM with the paper settings and the simulated-annealing reference.
The two instance sets come from the same distribution (Q = A + A^T, A_ij ~ N(0, 1)) but differ.
"""

import csv
import gzip
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "paper", "figures", "scaling_summary.pdf")


def load_json(name):
    p = os.path.join(HERE, name)
    return json.load(open(p)) if os.path.exists(p) else json.load(gzip.open(p + ".gz", "rt"))


def rows(name):
    exp = name.replace("_summary.csv", "")
    return list(csv.DictReader(open(os.path.join(HERE, "..", "experiments", "results", exp, name))))


def series(rs, **match):
    out = {}
    for r in rs:
        if all(r[k] == v for k, v in match.items()):
            out[int(r["n"])] = r
    return dict(sorted(out.items()))


Q = rows("qaoa_vs_vqpm_summary.csv")
G = rows("vqpm_grid_summary.csv")
qaoa_lb = series(Q, kind="qaoa", variant="L-BFGS-B", init_or_budget="random", p="8")
qaoa_lbi = series(Q, kind="qaoa", variant="L-BFGS-B", init_or_budget="interp", p="8")
qaoa_co = series(Q, kind="qaoa", variant="COBYLA", init_or_budget="random", p="8")
sa100 = series(Q, kind="sa", variant="100")
vq_small = series(Q, kind="vqpm", variant="oracle", init_or_budget="100")
grid_ex = series(G, phase_map="paper", precision="exact", stop_rule="oracle", budget="300")
grid_p3 = series(G, phase_map="paper", precision="3", stop_rule="oracle", budget="100")

SM = load_json("small_n_flow.json")      # first-order update on the seed-2026 instances, n = 4..22
SNS = sorted({r["n"] for r in SM})
sby = {n: [r for r in SM if r["n"] == n] for n in SNS}
FL = load_json("large_n_flow.json")
LN = load_json("large_n.json")
NS = sorted({r["n"] for r in FL})
by = {n: [r for r in FL if r["n"] == n] for n in NS}
lby = {n: [r for r in LN if r["n"] == n] for n in NS}


def rate(xs):
    return float(np.mean(xs))


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else float("nan")


flow = {p: {n: rate([r["runs"][p]["success"] for r in by[n]]) for n in NS} for p in ["none", "0.2", "0.01"]}
flow_t = {p: {n: med([r["runs"][p]["t_half"] for r in by[n] if r["runs"][p]["success"]]) for n in NS}
          for p in ["none", "0.2", "0.01"]}
small = {p: {n: rate([r["first_order"][p]["success"] for r in sby[n]]) for n in SNS} for p in ["none", "0.2", "0.01"]}
small_t = {p: {n: med([r["first_order"][p]["t_half"] for r in sby[n] if r["first_order"][p]["success"]]) for n in SNS}
           for p in ["none", "0.2", "0.01"]}
small_end = {p: {n: med([r["first_order"][p]["rounds"] for r in sby[n]]) for n in SNS} for p in ["none", "0.2", "0.01"]}
flow_end = {p: {n: med([r["runs"][p]["rounds"] for r in by[n]]) for n in NS} for p in ["none", "0.2", "0.01"]}
# Found-optimum rate as in the experiment reports: the returned string (locked bits plus the more likely
# value of each free qubit) is x_ref, whether or not x_ref reached probability 1/2 before the budget ran out.
# For the three-decimal loop most runs stall, so this differs from the pass test ("success").
def returned_ref(r, m):
    return abs(r[m]["E_out"] - r["E_ref"]) <= 1e-9 * max(1.0, abs(r["E_ref"]))


paper_large = {n: rate([returned_ref(r, "paper") for r in lby[n]]) for n in NS}
paper_pass = {n: rate([r["paper"]["success"] for r in lby[n]]) for n in NS}
device_large = {n: rate([returned_ref(r, "device") for r in lby[n]]) for n in NS}
sa_large = {n: rate([r["sa_hits"] / 32 for r in lby[n]]) for n in NS}


def f(d, n, key="found_optimum_rate"):
    v = d.get(n)
    if v is None:
        return "   -  "
    return "%6.2f" % (float(v[key]) if isinstance(v, dict) else v)


print("Found-optimum rate (fraction of instances); n <= 22: seed-2026 instances (100 per n, 50 for QAOA/SA at n >= 12),")
print("n >= 20: theory-check instances (100 per n), reference x_ref.")
print("1st-order columns: seed-2026 instances for n <= 22 except n = 20 (theory-check instances, as for n >= 30);")
print("the seed-2026 values at n = 20 are listed below the table.")
print("   n | QAOA LBFGS rnd | QAOA LBFGS interp | QAOA COBYLA rnd | SA 100 sw | VQPM p3 T100 | VQPM exact T300 |"
      " 1st-order none | 0.2 | 0.01 | sampled p3 T30 | device 2e5 | SA chain (3000 sw)")
for n in sorted(set(qaoa_lb) | set(grid_ex) | set(NS)):
    print("%4d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s" % (
        n, f(qaoa_lb, n), f(qaoa_lbi, n), f(qaoa_co, n), f(sa100, n), f(grid_p3, n), f(grid_ex, n),
        f(small["none"] if n in small["none"] and n not in NS else flow["none"], n),
        f(small["0.2"] if n in small["0.2"] and n not in NS else flow["0.2"], n),
        f(small["0.01"] if n in small["0.01"] and n not in NS else flow["0.01"], n), f(paper_large, n), f(device_large, n),
        f(sa_large, n)))
print("1st-order on the seed-2026 instances at n = 20: none %.2f, 0.2 %.2f, 0.01 %.2f" % tuple(
    small[p][20] for p in ["none", "0.2", "0.01"]))
print("sampled p3 T30 for n >= 20: returned string = x_ref (column above) vs x_ref reaching probability 1/2:",
      ", ".join("n=%d %.2f/%.2f" % (n, paper_large[n], paper_pass[n]) for n in NS))

print("\nIterations")
print("   n | VQPM exact T300 mean iters | QAOA LBFGS rnd evals | SA 100 sweeps proposals |"
      " 1st-order rounds to x_ref (median, passing): none / 0.2 / 0.01 | rounds to end (median): none / 0.2 / 0.01")
for n in sorted(set(qaoa_lb) | set(grid_ex) | set(NS)):
    a = "%8.1f" % float(grid_ex[n]["mean_iterations"]) if n in grid_ex else "      - "
    b = "%9.0f" % float(qaoa_lb[n]["mean_evals"]) if n in qaoa_lb and "mean_evals" in qaoa_lb[n] else "       - "
    c = "%6.0f" % float(sa100[n]["mean_evals"]) if n in sa100 and "mean_evals" in sa100[n] else "     - "
    if n in NS:
        d = " / ".join("%7.0f" % flow_t[p][n] for p in ["none", "0.2", "0.01"])
        e = " / ".join("%7.0f" % flow_end[p][n] for p in ["none", "0.2", "0.01"])
    elif n in SNS:
        d = " / ".join("%7.0f" % small_t[p][n] for p in ["none", "0.2", "0.01"])
        e = " / ".join("%7.0f" % small_end[p][n] for p in ["none", "0.2", "0.01"])
    else:
        d = e = "-"
    print("%4d | %s | %s | %s | %s | %s" % (n, a, b, c, d, e))

# ------------------------------------------------------------------ figure
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit(0)

import figstyle  # noqa: E402
from figstyle import GRID, INK, MUTED  # noqa: E402
figstyle.apply()
# VQPM: blue ordinal ramp as in Fig. app_flow (ink = no locking, darker = larger threshold);
# QAOA: orange; simulated annealing: aqua.
C_NONE, C_02, C_001, C_P3 = INK, "#0e3d7e", "#338ced", "#84b9f3"
C_QAOA, C_SA = "#e76624", "#00a888"


def xy(d, key="found_optimum_rate"):
    ns = sorted(d)
    return ns, [float(d[n][key]) if isinstance(d[n], dict) else d[n] for n in ns]


def join(lo, hi):
    """small-n series for n < 20, large-n series from n = 20 on (the two instance sets)."""
    d = {n: v for n, v in lo.items() if n < 20}
    d.update(hi)
    return d


fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
NT = [4, 8, 16, 32, 64, 100]
a1.axvline(20, color=MUTED, lw=0.8, ls=":", zorder=0)
x, y = xy(qaoa_lb)
a1.plot(x, y, "-", color=C_QAOA, marker="s", label="QAOA $p=8$, L-BFGS-B")
x, y = xy(qaoa_co)
a1.plot(x, y, "--", color=C_QAOA, lw=1.3, marker="s", mfc="white", label="QAOA $p=8$, COBYLA")
x, y = xy(sa100)
a1.plot(x, y, "-", color=C_SA, marker="^", label="SA, 100 sweeps")
x, y = xy(sa_large)
a1.plot(x, y, ":", color=C_SA, marker="^", mfc="white", label="SA, one chain of 3000 sweeps")
x, y = xy(join(small["none"], flow["none"]))
a1.plot(x, y, "-", color=C_NONE, marker="x", ms=5, label="VQPM, no locking")
x, y = xy(join(small["0.2"], flow["0.2"]))
a1.plot(x, y, "-", color=C_02, marker="D", ms=4, label=r"VQPM, $p_{\mathrm{diff}}=0.2$")
x, y = xy(join(small["0.01"], flow["0.01"]))
a1.plot(x, y, "-", color=C_001, marker="o", ms=4, label=r"VQPM, $p_{\mathrm{diff}}=10^{-2}$")
x, y = xy(grid_ex)
a1.plot(x, y, "none", color=C_001, marker="o", ms=6, mfc="none", mew=1.0,
        label=r"VQPM code, $p_{\mathrm{diff}}=10^{-2}$, 300 it.")
x, y = xy(join(grid_p3, paper_large))
a1.plot(x, y, "--", color=C_P3, lw=1.3, marker="o", ms=4, mfc="white", label=r"VQPM code, $p_{\mathrm{diff}}=10^{-2}$, 3 decimals")
a1.set_xscale("log")
a1.set_xticks(NT)
a1.set_xticklabels([str(v) for v in NT])
a1.minorticks_off()
a1.set_xlabel("$n$")
a1.set_ylabel("found-optimum rate")
a1.set_ylim(-0.02, 1.02)
a1.grid(True, axis="y", color=GRID, lw=0.6)
a1.set_title("(a)", loc="left")

for p, col, mk, lab in [("none", C_NONE, "x", "no locking"), ("0.2", C_02, "D", r"$p_{\mathrm{diff}}=0.2$"),
                        ("0.01", C_001, "o", r"$p_{\mathrm{diff}}=10^{-2}$")]:
    x, y = xy(join(small_t[p], flow_t[p]))
    a2.plot(x, y, "-", color=col, marker=mk, ms=4 if mk != "x" else 5, label=lab)
x, y = xy(grid_ex, "mean_iterations")
a2.plot(x, y, "none", color=C_001, marker="o", ms=6, mfc="none", mew=1.0)
a2.axvline(20, color=MUTED, lw=0.8, ls=":", zorder=0)
a2.set_xscale("log")
a2.set_yscale("log")
a2.set_xticks(NT)
a2.set_xticklabels([str(v) for v in NT])
a2.minorticks_off()
a2.set_xlabel("$n$")
a2.set_ylabel("iterations to reach the optimum")
a2.grid(True, which="major", axis="y", color=GRID, lw=0.6)
a2.set_title("(b)", loc="left")

h, l = a1.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, handlelength=2.6, columnspacing=1.5)
fig.tight_layout(rect=(0, 0.26, 1, 1))
fig.savefig(FIG)
print("\nwrote", os.path.relpath(FIG, os.path.join(HERE, "..")))
