"""Tables and figure for the product-state map with and without locking, from large_n_flow.json.

    python3 theory_checks/large_n_flow_report.py > theory_checks/large_n_flow_report.out
Writes paper/figures/app_flow.pdf. Also compares with the sampled VQPM runs of large_n_sweep.json.
"""

import gzip
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    p = os.path.join(HERE, name)
    return json.load(open(p)) if os.path.exists(p) else json.load(gzip.open(p + ".gz", "rt"))


R = load("large_n_flow.json")
SW = {(r["n"], r["k"]): r for r in load("large_n_sweep.json")}
NS = sorted({r["n"] for r in R})
P = list(R[0]["runs"].keys())                      # "none", "0.2", ..., "1e-06"
by = {n: [r for r in R if r["n"] == n] for n in NS}
pct = lambda a: 100.0 * np.mean(a) if len(a) else float("nan")  # noqa: E731
med = lambda a: float(np.median(a)) if len(a) else float("nan")  # noqa: E731


def row(label, vals, fmt):
    print("%8s | " % label + " ".join(fmt % v for v in vals))


print("Unlocked product-state map (first-order form), from the uniform start, 100 instances per n")
print("  n | ends at x_ref | still moving after 3e6 rounds | settled runs ending at a strict 1-flip local min |"
      " Hamming to x_ref (median, misses) | energy gap (median, misses) | rounds to converge (median) | /S |"
      " t_half (median, hits) | /S")
for n in NS:
    u = [r["runs"]["none"] for r in by[n]]
    S = [r["S"] for r in by[n]]
    miss = [x for x in u if not x["is_ref"]]
    hit = [(x, s) for x, s in zip(u, S) if x["t_half"]]
    print("%3d | %4.0f%% | %3d | %4.0f%% | %4.1f | %.4f | %8.0f | %5.1f | %7.0f | %5.1f" % (
        n, pct([x["is_ref"] for x in u]), sum(x["rounds"] >= 3 * 10**6 for x in u),
        pct([x["local_min"] for x in u if x["rounds"] < 3 * 10**6]),
        med([x["ham_ref"] for x in miss]), med([x["gap"] for x in miss]), med([x["rounds"] for x in u]),
        med([x["rounds"] / s for x, s in zip(u, S)]), med([x["t_half"] for x, s in hit]),
        med([x["t_half"] / s for x, s in hit])))

print("\nWith locking (no rounding): runs that pass the code's test (x_ref reaches probability 1/2 before any"
      " lock contradicts it), %")
print("     p_diff | " + " ".join("%6d" % n for n in NS))
for p in P:
    row(p, [pct([r["runs"][p]["success"] for r in by[n]]) for n in NS], "%5.0f%%")

print("\nRuns ending at the unlocked end point x_inf, %")
for p in P[1:]:
    row(p, [pct([r["runs"][p]["is_inf"] for r in by[n]]) for n in NS], "%5.0f%%")

print("\nLocks that agree with x_inf, % (mean over runs)")
for p in P[1:]:
    row(p, [100 * np.mean([r["runs"][p]["locks_agree_inf"] for r in by[n]]) for n in NS], "%5.0f%%")

print("\nLocks that agree with x_ref, % (mean over runs)")
for p in P[1:]:
    row(p, [100 * np.mean([r["runs"][p]["locks_agree_ref"] for r in by[n]]) for n in NS], "%5.0f%%")

print("\nOf the instances whose unlocked map ends at x_ref: share whose locked run still passes, %")
for p in P[1:]:
    row(p, [pct([r["runs"][p]["success"] for r in by[n] if r["runs"]["none"]["is_ref"]]) for n in NS], "%5.0f%%")
print("Of the instances whose unlocked map ends elsewhere: share whose locked run passes, %")
for p in P[1:]:
    row(p, [pct([r["runs"][p]["success"] for r in by[n] if not r["runs"]["none"]["is_ref"]]) for n in NS], "%5.0f%%")

print("\nSplit of the passing runs relative to the unlocked map (counts per 100): kept = unlocked ends at x_ref and the"
      " locked run passes; lost = unlocked ends at x_ref, locked run fails; rescued = unlocked misses, locked passes")
for p in P[1:]:
    kept = [sum(r["runs"]["none"]["is_ref"] and r["runs"][p]["success"] for r in by[n]) for n in NS]
    lost = [sum(r["runs"]["none"]["is_ref"] and not r["runs"][p]["success"] for r in by[n]) for n in NS]
    resc = [sum((not r["runs"]["none"]["is_ref"]) and r["runs"][p]["success"] for r in by[n]) for n in NS]
    print("%8s | " % p + " ".join("%3d/%3d/%3d" % v for v in zip(kept, lost, resc)))
print("\nUnlocked runs that pass the code's test (t_half reached) but do not end at x_ref, count per 100")
row("none", [sum(bool(r["runs"]["none"]["success"]) and not r["runs"]["none"]["is_ref"] for r in by[n]) for n in NS], "%6d")

print("\nRounds until all qubits are locked or the free ones have converged (median)")
for p in P:
    row(p, [med([r["runs"][p]["rounds"] for r in by[n]]) for n in NS], "%6.0f")
print("\nRound in which x_ref first reaches probability 1/2, passing runs (median)")
for p in P:
    row(p, [med([r["runs"][p]["t_half"] for r in by[n] if r["runs"][p]["success"]]) for n in NS], "%6.0f")
print("\nRound of the first lock (median)")
for p in P[1:]:
    row(p, [med([r["runs"][p]["first_lock"] for r in by[n]]) for n in NS], "%6.0f")


print("\nSigns that locking reads, on the unlocked run: the sign of 2 pi_i - 1 when |2 pi_i - 1| first exceeds p,")
print("compared with the centre gradient -sign g_i(1/2) (= sign of round-1 D_i), the linear prediction")
print("delta(t) = -(I - A^t) J^{-1} g0 at the same round, the unlocked end point x_inf and x_ref (share of qubits).")
print("tau = 4/(eta0 |mu_min(J)|); 'flips' = reads against the centre gradient, 'lin on flips' = share of them predicted")
for n in NS:
    L = [r["linear"] for r in by[n]]
    tau = [l["tau"] for l in L]
    print("n = %3d  tau median %.0f = %.2f n^1.5 = %.2f S | x_inf agrees with grad %.2f, x_ref with grad %.2f" % (
        n, med(tau), med([t / n**1.5 for t in tau]), med([l["tau"] / r["S"] for l, r in zip(L, by[n])]),
        np.mean([l["inf_vs_grad"] for l in L]), np.mean([l["ref_vs_grad"] for l in L])))
    for p in L[0]["read"]:
        D = [l["read"][p] for l in L]
        tt = [np.median([v for v in d["t"] if v > 0]) / l["tau"] for d, l in zip(D, L)]
        lf = [d["lin_on_flips"] for d in D if d["lin_on_flips"] is not None]
        print("   p=%-7s t/tau %6.3f | grad %.3f  lin %.3f  inf %.3f  ref %.3f | lin on flips %.2f of %d" % (
            p, med(tt), *[np.mean([d[q] for d in D]) for q in ("grad", "lin", "inf", "ref")],
            np.mean(lf) if lf else float("nan"), sum(d["n_flips"] for d in D)))


def slope(xs, ys):
    xs, ys = np.array(xs, float), np.array(ys, float)
    ok = np.isfinite(ys) & (ys > 0)
    return np.polyfit(np.log(xs[ok]), np.log(ys[ok]), 1)[0]


ns = np.array(NS, float)
print("\nfitted exponent of tau in n: %.2f" % slope(NS, [med([r["linear"]["tau"] for r in by[n]]) for n in NS]))
print("\nfitted exponents in n: unlocked t_half %.2f, unlocked rounds %.2f, S %.2f" % (
    slope(NS, [med([r["runs"]["none"]["t_half"] for r in by[n] if r["runs"]["none"]["t_half"]]) for n in NS]),
    slope(NS, [med([r["runs"]["none"]["rounds"] for r in by[n]]) for n in NS]),
    slope(NS, [med([r["S"] for r in by[n]]) for n in NS])))
for p in ("0.2", "0.1", "0.01", "0.001"):
    print("  p_diff %s: rounds to finish %.2f, t_half (passing) %.2f, first lock %.2f" % (
        p, slope(NS, [med([r["runs"][p]["rounds"] for r in by[n]]) for n in NS]),
        slope(NS, [med([r["runs"][p]["t_half"] for r in by[n] if r["runs"][p]["success"]]) for n in NS]),
        slope(NS, [med([r["runs"][p]["first_lock"] for r in by[n]]) for n in NS])))

print("\nCheck against sampled VQPM (large_n_sweep.json, same instances): pass rate first-order / sampled, and"
      " per-instance agreement")
for p in ("0.01", "0.001", "0.0001", "1e-05"):
    vals = ["%3.0f/%3.0f" % (pct([r["runs"][p]["success"] for r in by[n]]),
                             pct([SW[(n, r["k"])]["sweep"][p]["success"] for r in by[n]])) for n in NS]
    agree = 100 * np.mean([r["runs"][p]["success"] == SW[(r["n"], r["k"])]["sweep"][p]["success"] for r in R])
    print("%8s | %s | agreement %.1f%%" % (p, " ".join(vals), agree))

# ------------------------------------------------------------------ figure
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit(0)

INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"
import figstyle  # noqa: E402
figstyle.apply()
# ordinal ramps: n (orange ramp), p_diff (blue ramp, darker = larger threshold)
NCOL = {20: "#f8a77e", 60: "#e76624", 100: "#9a3a0c"}
PCOL = {"none": INK, "0.2": "#0e3d7e", "0.1": "#1b64be", "0.01": "#338ced", "0.001": "#84b9f3", "0.0001": "#b6d5f8"}
PLAB = {"none": "no locking", "0.2": r"$p_{\mathrm{diff}}=0.2$", "0.1": "0.1", "0.01": r"$10^{-2}$",
        "0.001": r"$10^{-3}$", "0.0001": r"$10^{-4}$"}
fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(7.2, 5.6))

# (a) overlap with x_ref along the unlocked map, against rounds / S
for n in (20, 60, 100):
    for ends, ls in ((True, "-"), (False, "--")):
        rs = [r for r in by[n] if r["runs"]["none"]["is_ref"] == ends]
        if not rs:
            continue
        cs = np.logspace(-2, 2.5, 28)
        curves = []
        for r in rs:
            lg = {t: o for (t, o, nl, b) in r["runs"]["none"]["log"]}
            last = r["runs"]["none"]["log"][-1][1]
            curves.append([lg.get(t, last) for t in r["log_at"][:len(cs)]])
        L = min(len(c) for c in curves)
        x = cs[:L]
        y = np.median(np.array([c[:L] for c in curves]), axis=0)
        a3_lab = "n=%d, %s" % (n, "ends at $x_{\\mathrm{ref}}$" if ends else "ends elsewhere")
        a1.plot(x, y, ls, color=NCOL[n], lw=2)
a1.set_xscale("log")
a1.set_xlabel("round / $S$")
a1.set_ylabel(r"overlap with $x_{\mathrm{ref}}$")
a1.set_ylim(-0.02, 1.02)
hs = [a1.plot([], [], "-", color=NCOL[n], lw=2, label="$n=%d$" % n)[0] for n in (20, 60, 100)]
hs += [a1.plot([], [], ls, color=INK, lw=1.5, label=lab)[0]
       for ls, lab in (("-", "ends at $x_{\\mathrm{ref}}$"), ("--", "ends elsewhere"))]
a1.legend(handles=hs, loc="upper left")
a1.grid(True, axis="y", color=GRID, lw=0.6)
a1.set_title("(a)", loc="left")

# (b) pass rate versus n
marks = {"none": "o", "0.2": "s", "0.1": "^", "0.01": "D", "0.001": "v", "0.0001": "P"}
for p in ("none", "0.2", "0.1", "0.01", "0.001", "0.0001"):
    a2.plot(NS, [pct([r["runs"][p]["success"] for r in by[n]]) for n in NS], "-", color=PCOL[p], lw=2,
            marker=marks[p], ms=4, label=PLAB[p])
for p in ("0.01", "0.001"):
    a2.plot(NS, [pct([SW[(n, r["k"])]["sweep"][p]["success"] for r in by[n]]) for n in NS], "none",
            marker="o", ms=7, mfc="none", mec=PCOL[p], mew=1)
a2.set_xticks(NS)
a2.set_xlabel("$n$")
a2.set_ylabel(r"runs reaching $x_{\mathrm{ref}}$ (%)")
a2.set_ylim(-3, 100)
a2.legend(fontsize=9, loc="upper right", ncol=2)
a2.grid(True, axis="y", color=GRID, lw=0.6)
a2.set_title("(b)", loc="left")

# (c) signs that locking would read on the unlocked run, versus p_diff
RCOL = {"grad": "#1b64be", "lin": "#e76624", "inf": "#1a9e8f", "ref": INK}
RLAB = {"grad": "centre gradient (round-1 $D_i$)", "lin": "linear prediction", "inf": "unlocked end point",
        "ref": r"$x_{\mathrm{ref}}$"}
rp = list(by[NS[0]][0]["linear"]["read"].keys())
rv = [float(p) for p in rp]
for n, ls in ((100, "-"), (20, "--")):
    L = [r["linear"]["read"] for r in by[n]]
    for q in ("grad", "lin", "inf", "ref"):
        a3.plot(rv, [100 * np.mean([l[p][q] for l in L]) for p in rp], ls, color=RCOL[q], lw=2 if ls == "-" else 1.2,
                marker="o" if ls == "-" else None, ms=3, label=RLAB[q] if n == 100 else None)
a3.plot([], [], "-", color=INK, lw=2, label="$n=100$")
a3.plot([], [], "--", color=INK, lw=1.2, label="$n=20$")
a3.set_xscale("log")
a3.set_xlabel(r"$p_{\mathrm{diff}}$ (sign read when $|2\pi_i-1|$ first exceeds it)")
a3.set_ylabel("signs agreeing (%)")
a3.set_ylim(50, 101)
a3.legend(fontsize=9, loc="lower left", ncol=2)
a3.grid(True, axis="y", color=GRID, lw=0.6)
a3.set_title("(c)", loc="left")

# (d) rounds versus n
for p in ("none", "0.2", "0.1", "0.01", "0.001"):
    a4.plot(ns, [med([r["runs"][p]["t_half"] for r in by[n] if r["runs"][p]["success"]]) for n in NS], "-",
            color=PCOL[p], lw=2, marker=marks[p], ms=4, label=PLAB[p])
ref = med([r["runs"]["none"]["t_half"] for r in by[20] if r["runs"]["none"]["t_half"]]) * (ns / 20) ** 2
a4.plot(ns, ref, ":", color=INK, lw=1)
a4.text(ns[-1] * 1.04, ref[-1], "$\\propto n^2$", fontsize=9, color=INK, va="center")
a4.set_xscale("log")
a4.set_yscale("log")
a4.set_xticks([20, 30, 50, 100])
a4.set_xticklabels(["20", "30", "50", "100"])
a4.minorticks_off()
a4.set_xlim(18, 130)
a4.set_xlabel("$n$")
a4.set_ylabel(r"rounds until $\Pr(x_{\mathrm{ref}})\geq1/2$")
a4.set_ylim(1.5, 1e8)
a4.legend(fontsize=9, loc="upper left", ncol=2)
a4.grid(True, which="major", axis="y", color=GRID, lw=0.6)
a4.set_title("(d)", loc="left")

fig.tight_layout()
out = os.path.join(HERE, "..", "paper", "figures", "app_flow.pdf")
fig.savefig(out)
if os.environ.get("PREVIEW"):
    fig.savefig(os.environ["PREVIEW"], dpi=110)
print("figure written to", os.path.relpath(out, os.path.dirname(HERE)))
