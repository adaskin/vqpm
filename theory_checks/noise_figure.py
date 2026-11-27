"""Table and figure of the noise study from the noisy Aer runs of noise_rerun.py (no new runs).

Run from the repository root:
    python3 theory_checks/noise_figure.py > theory_checks/noise_figure.out
Reads theory_checks/noise_rerun.jsonl; writes paper/figures/noisy_locking.pdf.

Free-qubit difference |P0 - P1| = |2 pi - 1| of every qubit that is not yet locked, from the marginals of the
next input (rounded); a qubit locks in the iteration in which the measured difference exceeds p_diff.
"""

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "paper", "figures", "noisy_locking.pdf")
R = [json.loads(l) for l in open(os.path.join(HERE, "noise_rerun.jsonl"))]
MED = "depol:2.7e-4,4.1e-3,2.3e-2"
NOISES = ["none", "depol:1e-5,1e-4,1e-3", "depol:1e-4,1e-3,1e-2", MED]
NAME = {"none": "sampling, no noise", "depol:1e-5,1e-4,1e-3": "(1e-5, 1e-4, 1e-3)",
        "depol:1e-4,1e-3,1e-2": "(1e-4, 1e-3, 1e-2)", MED: "(2.7e-4, 4.1e-3, 2.3e-2), FakeTorino medians"}


def sel(n, pdiff, noise, shots=100000, kind="circuit"):
    out = []
    for r in R:
        if r["n"] != n or abs(r["pdiff"] - pdiff) > 1e-12 or r["kind"] != kind:
            continue
        if kind == "circuit" and (r["noise"] != noise or r["shots"] != shots):
            continue
        out.append(r)
    return sorted(out, key=lambda r: (r["instance"], r.get("sim_seed", 0)))


def wilson(k, m, z=1.96):
    if m == 0:
        return float("nan"), float("nan")
    p = k / m
    d = 1 + z * z / m
    c = (p + z * z / (2 * m)) / d
    h = z * math.sqrt(p * (1 - p) / m + z * z / (4 * m * m)) / d
    return c - h, c + h


def stats(rs, n):
    k = sum(r["success"] for r in rs)
    lo, hi = wilson(k, len(rs))
    wrong = [sum(1 for e in r["lock_log"] if e.get("matches_optimum") is False) for r in rs]
    return dict(m=len(rs), k=k, rate=k / len(rs) if rs else float("nan"), lo=lo, hi=hi,
                bits=np.mean([r["bit_diff"] for r in rs]) if rs and "bit_diff" in rs[0] else float("nan"),
                iters=np.mean([r["iterations"] for r in rs]), locks=np.mean([r["n_locked_history"][-1] for r in rs]),
                wrong_runs=np.mean([w > 0 for w in wrong]), wrong=np.mean(wrong))


def free_diffs(r):
    """per iteration, the differences |2 pi - 1| of the free qubits (NaN once locked)."""
    H = np.array(r["prepared_history"], float)
    D = np.abs(2 * H - 1)
    D[(H == 0) | (H == 1)] = np.nan
    return D


print("Circuit VQPM under noise (Qiskit Aer, density matrices); 5 instances x 3 simulator seeds at n = 8,")
print("5 instances x 1 seed at n = 10; at most 100 iterations. Exact = state-vector VQPM, same settings.")
print("rate [Wilson 95%] | mean bit difference | mean iterations | mean locked qubits at the end |"
      " runs with a wrong lock | mean wrong locks")
for n in (8, 10):
    for pdiff, prec in ((0.1, 4), (0.01, 3)):
        ex = sel(n, pdiff, None, kind="exact")
        rows = [("exact marginals", ex)] if ex else []
        for noise in NOISES:
            rs = sel(n, pdiff, noise)
            if rs:
                rows.append((NAME[noise], rs))
        rs = sel(n, pdiff, MED, shots=1000000)
        if rs:
            rows.append((NAME[MED] + ", 1e6 shots", rs))
        if not rows:
            continue
        print("\nn = %d, p_diff = %g, %d decimals" % (n, pdiff, prec))
        for lab, rs in rows:
            s = stats(rs, n)
            print("  %-52s %2d/%2d = %.2f [%.2f, %.2f] | %.2f | %5.1f | %4.1f | %.2f | %.2f" % (
                lab, s["k"], s["m"], s["rate"], s["lo"], s["hi"], s["bits"], s["iters"], s["locks"], s["wrong_runs"],
                s["wrong"]))

# level of the free differences: median over free qubits and iterations 5..end, per run, then median over runs
print("\nMedian free-qubit difference |P0 - P1| after iteration 5 (median over runs), n = 8")
for pdiff in (0.1, 0.01):
    for noise, shots in [("none", 100000), (MED, 100000), (MED, 1000000)]:
        rs = sel(8, pdiff, noise, shots)
        if not rs:
            continue
        v = [np.nanmedian(free_diffs(r)[5:]) for r in rs if np.isfinite(free_diffs(r)[5:]).any()]
        print("  p_diff %-5g %-30s shots %7d: %.4f (%d runs with free qubits after iteration 5)" % (
            pdiff, noise, shots, np.median(v) if v else float("nan"), len(v)))

# ------------------------------------------------------------------ figure
import figstyle  # noqa: E402
from figstyle import GRID, INK, MUTED  # noqa: E402

plt = figstyle.apply()
C_NONE, C_MED, C_MED6 = "#257de0", "#e76624", "#8a3fb3"
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.7))


def traj(ax, r, col, lab, pdiff):
    D = free_diffs(r)
    t = np.arange(1, len(D) + 1)
    for q in range(D.shape[1]):
        ax.plot(t, D[:, q], "-", color=col, lw=0.9, alpha=0.85)
        ok = np.flatnonzero(np.isfinite(D[:, q]))
        if len(ok) and ok[-1] < len(D) - 1:            # locked after iteration ok[-1] + 1
            ax.plot(ok[-1] + 2, pdiff, "o", color=col, ms=4)
    ax.plot([], [], "-", color=col, lw=1.5, label=lab)


ax = axes[0]
r0 = sel(8, 0.1, "none")[0]
r1 = sel(8, 0.1, MED)[0]
traj(ax, r0, C_NONE, "no noise (dots: qubit locks)", 0.1)
traj(ax, r1, C_MED, "noise at the FakeTorino medians", 0.1)
ax.axhline(0.1, color=INK, lw=1.0, ls="--")
ax.text(99, 0.115, "$p_{\\mathrm{diff}}=0.1$", ha="right", va="bottom", color=INK)
ax.set_yscale("log")
ax.set_ylim(1e-3, 2.0)
ax.set_xlim(0, 101)
ax.set_xlabel("iteration")
ax.set_ylabel("$|P_0-P_1|$ of each free qubit")
ax.set_title("(a) one run, $n=8$, $p_{\\mathrm{diff}}=0.1$", loc="left")
ax.grid(True, axis="y", color=GRID, lw=0.6)
ax.legend(loc="upper right")

ax = axes[1]
for pdiff, noise, shots, col, ls, lab in [
        (0.1, "none", 100000, C_NONE, "--", "no noise, $p_{\\mathrm{diff}}=0.1$"),
        (0.1, MED, 100000, C_MED, "--", "noise, $p_{\\mathrm{diff}}=0.1$"),
        (0.01, "none", 100000, C_NONE, "-", "no noise, $p_{\\mathrm{diff}}=0.01$"),
        (0.01, MED, 100000, C_MED, "-", "noise, $p_{\\mathrm{diff}}=0.01$"),
        (0.01, MED, 1000000, C_MED6, "-", "noise, $p_{\\mathrm{diff}}=0.01$, $10^6$ shots")]:
    rs = sel(8, pdiff, noise, shots)
    if not rs:
        continue
    L = 100
    nl = np.array([(r["n_locked_history"] + [r["n_locked_history"][-1]] * L)[:L] for r in rs], float)
    ax.plot(np.arange(1, L + 1), nl.mean(0), ls, color=col, label=lab)
ax.set_xlabel("iteration")
ax.set_ylabel("locked qubits (mean of 15 runs)")
ax.set_xlim(0, 101)
ax.set_ylim(-0.2, 8.4)
ax.grid(True, axis="y", color=GRID, lw=0.6)
ax.set_title("(b) all runs, $n=8$", loc="left")
ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.42))
fig.tight_layout()
fig.savefig(FIG)
print("\nwrote", os.path.relpath(FIG, os.path.join(HERE, "..")))
