"""Tables and figure for the larger instances, from large_n.json and large_n_sweep.json.

    python3 theory_checks/large_n_report.py > theory_checks/large_n_report.out

Also recomputes round-1 P1 - P0 exactly (enumeration) for n = 6 ... 18 on the same instance
family, so the figure covers n = 6 ... 100.  Writes paper/figures/app_large_n.pdf.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
from vqpm import random_qubo  # noqa: E402

_js = os.path.join(HERE, "large_n.json")
if os.path.exists(_js):
    R = json.load(open(_js))
else:
    import gzip
    R = json.load(gzip.open(_js + ".gz", "rt"))
NS = sorted({r["n"] for r in R})
by = {n: [r for r in R if r["n"] == n] for n in NS}
pct = lambda a: 100.0 * np.mean(a)  # noqa: E731


def exact_round1(n, count=100, seed=2026):
    k = np.arange(2**n)
    B = ((k[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1).astype(float)
    np.random.seed(seed + n)
    med, mx = [], []
    for _ in range(count):
        T = np.triu(random_qubo(n))
        S = np.abs(T).sum()
        E = np.einsum("ij,ij->i", B @ T, B)
        w = np.cos((np.pi / 4 + np.pi / 4 * E / S) / 2) ** 2
        D = np.abs(2 * (w @ B) / w.sum() - 1)
        med.append(np.median(D))
        mx.append(D.max())
    return np.median(med), np.median(mx)


print("Reference solutions")
for n in NS:
    rs = by[n]
    line = "n=%3d  instances %d  all 32 annealing chains agree: %.0f%%" % (
        n, len(rs), pct([r["sa_hits"] == 32 for r in rs]))
    if "sa_is_exact" in rs[0]:
        line += "  annealing = exhaustive optimum: %.0f%%" % pct([r["sa_is_exact"] for r in rs])
    print(line)

print("\nRound 1 from the uniform start: D_i = P1 - P0 (200k draws, control variate)")
print("  n | median_i|D_i| | max_i|D_i| | |D_i|<1e-3 (erased by 3-decimal rounding) | |D_i|>=0.01 |"
      " sign agrees with x_ref | strongest qubit agrees | first-order formula: rel. error (median), sign agrees")
r1 = {}
for n in NS:
    D = [np.abs(np.array(r["round1"]["D"])) for r in by[n]]
    F = [np.array(r["round1"]["D_formula"]) for r in by[n]]
    Ds = [np.array(r["round1"]["D"]) for r in by[n]]
    med = np.median([np.median(d) for d in D])
    mx = np.median([d.max() for d in D])
    fmed = np.median([np.median(np.abs(f)) for f in F])
    erased = pct(np.concatenate([d < 1e-3 for d in D]))
    lockable = pct(np.concatenate([d >= 1e-2 for d in D]))
    agree = 100 * np.mean([r["round1"]["agree"] for r in by[n]])
    top = pct([r["round1"]["agree_top"] for r in by[n]])
    relerr = np.median([np.max(np.abs(f - d)) / np.max(np.abs(d)) for f, d in zip(F, Ds)])
    fagree = 100 * np.mean([np.mean(np.sign(f) == np.sign(d)) for f, d in zip(F, Ds)])
    r1[n] = (med, mx, fmed)
    print("%3d | %.2e | %.2e | %5.1f%% | %5.1f%% | %5.1f%% | %5.1f%% | %.3f, %.1f%%"
          % (n, med, mx, erased, lockable, agree, top, relerr, fagree))
ns = np.array(NS, float)
print("fit |D| ~ n^a over n=%d..%d: median a = %.2f, max a = %.2f" % (
    NS[0], NS[-1], np.polyfit(np.log(ns), np.log([r1[n][0] for n in NS]), 1)[0],
    np.polyfit(np.log(ns), np.log([r1[n][1] for n in NS]), 1)[0]))

MODES = [("paper", "paper settings (3 decimals, 30 rounds)"),
         ("norounding", "no rounding, 30 rounds"),
         ("norounding_long", "no rounding, 300 rounds"),
         ("device", "device, 2e5 shots/round, 3 decimals, 30 rounds")]
print("\nFull runs (p_diff = 0.01, oracle stop on x_ref)")
summary = {}
for m, label in MODES:
    print("\n[%s]" % label)
    print("  n | found x_ref | wrong lock | stalled | rounds (median) | locked at end (median %% of n) |"
          " output gap (E_out-E_ref)/|E_ref| median | output 1-flip local min")
    for n in NS:
        L = [r[m] for r in by[n]]
        gaps = [(l["E_out"] - r["E_ref"]) / abs(r["E_ref"]) for l, r in zip(L, by[n])]
        found = pct([l["success"] for l in L])
        wrong = pct([l["wrong_lock"] for l in L])
        summary[(m, n)] = (found, wrong)
        print("%3d | %5.0f%% | %5.0f%% | %5.0f%% | %5.1f | %5.0f%% | %.3f | %5.0f%%"
              % (n, found, wrong, pct([l["stalled"] for l in L]), np.median([l["rounds"] for l in L]),
                 np.median([100 * l["n_locked"] / n for l in L]), np.median(gaps), pct([l["local_min"] for l in L])))
    if m == "paper":
        print("  stalled in round 1 with nothing locked: " + ", ".join(
            "n=%d %.0f%%" % (n, pct([r[m]["stalled"] and r[m]["rounds"] == 1 and r[m]["n_locked"] == 0
                                    for r in by[n]])) for n in NS))
    if m == "device":
        print("  device: wrong locks that a noiseless run (norounding, 30) did not make: " + ", ".join(
            "n=%d %.0f%%" % (n, pct([r[m]["wrong_lock"] and not r["norounding"]["wrong_lock"] for r in by[n]]))
            for n in NS))

print("\nMax |P1 - P0| over free qubits per round, median over instances, rounds 1..10")
for m, label in MODES[1:2] + MODES[3:]:
    for n in (20, 60, 100):
        if n in by:
            rows = [r[m]["maxD"] for r in by[n]]
            vals = [np.median([x[t] for x in rows if len(x) > t]) if any(len(x) > t for x in rows) else np.nan
                    for t in range(10)]
            print("  %-12s n=%3d: %s" % (m, n, " ".join("%.1e" % v for v in vals)))

# ------------------------------------------------------------------ p_diff sweep, no rounding
_sj = os.path.join(HERE, "large_n_sweep.json")
if os.path.exists(_sj):
    SW = json.load(open(_sj))
else:
    import gzip
    SW = json.load(gzip.open(_sj + ".gz", "rt"))
PD = list(SW[0]["sweep"].keys())
sby = {n: [r for r in SW if r["n"] == n] for n in NS}
assert all(r["E_ref"] == q["E_ref"] for r, q in zip(sorted(SW, key=lambda r: (r["n"], r["k"])), R))


def tab(title, f, fmt):
    print("\n" + title)
    print("  n | " + " | ".join("%7s" % p for p in PD))
    for n in NS:
        print("%3d | " % n + " | ".join(fmt % f(sby[n], p) for p in PD))


print("\n\np_diff sweep without rounding (exact marginals at n=20, sampled above), budget ceil(1.5 n^1.5) rounds")
print("budgets: " + ", ".join("n=%d %d" % (n, sby[n][0]["budget"]) for n in NS))
tab("reached x_ref (%)", lambda rs, p: pct([r["sweep"][p]["success"] for r in rs]), "%6.0f%%")
tab("wrong lock (%)", lambda rs, p: pct([r["sweep"][p]["wrong_lock"] for r in rs]), "%6.0f%%")
tab("hit the round budget (%)", lambda rs, p: pct([r["sweep"][p]["hit_budget"] for r in rs]), "%6.0f%%")
tab("qubits locked in round 1 (%)", lambda rs, p: 100 * np.mean([r["sweep"][p]["locks_round1"] / r["n"] for r in rs]), "%6.0f%%")
tab("median rounds used", lambda rs, p: np.median([r["sweep"][p]["rounds"] for r in rs]), "%7.1f")
tab("median round of the first lock", lambda rs, p: np.median([r["sweep"][p]["first_lock"] or np.nan for r in rs]), "%7.1f")


def succ_rounds(rs, p):
    v = [r["sweep"][p]["rounds"] for r in rs if r["sweep"][p]["success"]]
    return np.median(v) if v else np.nan


tab("median rounds of the successful runs", succ_rounds, "%7.1f")
tab("median output gap (E_out - E_ref)/|E_ref|",
    lambda rs, p: np.median([(r["sweep"][p]["E_out"] - r["E_ref"]) / abs(r["E_ref"]) for r in rs]), "%7.3f")

print("\nRound-1 resolution: draws per round to resolve the median |D_i| at three standard deviations")
print("  n | W | std(w) | median |D_i| | device, post-selected 9/(W D^2) | classical, control variate 9 Var(w)/(W^2 D^2)")
res = {}
for n in NS:
    q = [r["round1"] for r in sby[n]]
    res[n] = [np.median([x[k] for x in q]) for k in ("W", "std_w", "D_median", "device_shots_3sigma",
                                                       "classical_draws_3sigma")]
    print("%3d | %.3f | %.2e | %.2e | %.2e | %.2e" % (n, *res[n]))


def slope(xs, ys):
    xs, ys = np.array(xs, float), np.array(ys, float)
    ok = np.isfinite(ys) & (ys > 0)
    return np.polyfit(np.log(xs[ok]), np.log(ys[ok]), 1)[0]


print("fitted exponents in n: device shots %.2f, classical draws %.2f, std(w) %.2f" % (
    slope(NS, [res[n][3] for n in NS]), slope(NS, [res[n][4] for n in NS]), slope(NS, [res[n][1] for n in NS])))
fl = {p: [np.median([r["sweep"][p]["first_lock"] or np.nan for r in sby[n]]) for n in NS] for p in PD}
sr = {p: [succ_rounds(sby[n], p) for n in NS] for p in PD}
print("fitted exponents in n: first-lock round (p_diff=1e-2) %.2f; successful-run rounds p_diff=1e-2 %.2f,"
      " p_diff=1e-3 %.2f" % (slope(NS, fl["0.01"]), slope(NS, sr["0.01"]), slope(NS, sr["0.001"])))
print("successful runs per n: " + ", ".join("n=%d %d/%d" % (n, sum(r["sweep"]["0.01"]["success"] for r in sby[n]),
                                                       sum(r["sweep"]["0.001"]["success"] for r in sby[n])) for n in NS)
      + "  (p_diff 1e-2 / 1e-3)")

# ------------------------------------------------------------------ figure
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit(0)

small = [6, 8, 10, 12, 14, 16, 18]
ex = {n: exact_round1(n) for n in small}
print("\nExact round-1 |D| for n = 6..18 (median of per-instance median, max): " +
      ", ".join("n=%d %.1e/%.1e" % (n, *ex[n]) for n in small))

INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"
C = ["#257de0", "#f47230"]                                   # categorical slots 1, 2
RAMP = {"0.01": "#0f4a96", "0.001": "#2370cd", "0.0001": "#519ced", "low": "#9dc8f7"}   # ordinal: p_diff
import figstyle  # noqa: E402
figstyle.apply()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.9))
xs_s, xs_l = np.array(small, float), ns


def logx(ax, ticks=(6, 10, 20, 50, 100)):
    ax.set_xscale("log")
    ax.set_xticks(list(ticks))
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("$n$")


# (a) round-1 differences
a1.plot(xs_s, [ex[n][1] for n in small], "-", color=C[0], lw=2, marker="o", ms=4, label="largest $|D_i|$")
a1.plot(xs_l, [r1[n][1] for n in NS], "-", color=C[0], lw=2, marker="o", ms=4, mfc="white")
a1.plot(xs_s, [ex[n][0] for n in small], "-", color=C[1], lw=2, marker="s", ms=4, label="median $|D_i|$")
a1.plot(xs_l, [r1[n][0] for n in NS], "-", color=C[1], lw=2, marker="s", ms=4, mfc="white")
a1.plot(xs_l, [r1[n][2] for n in NS], ":", color=INK, lw=1.2, label="first-order formula")
for k in (2, 3, 4, 5):
    a1.axhline(10.0 ** -k, color=MUTED, lw=0.7, ls="--")
    a1.text(104, 10.0 ** -k, r"$10^{-%d}$" % k, color=MUTED, fontsize=9.5, ha="left", va="center")
logx(a1)
a1.set_yscale("log")
a1.set_ylim(5e-6, 5e-2)
a1.set_ylabel(r"round-1 $|P_1-P_0|$")
a1.legend(fontsize=9.5, loc="lower left")
a1.grid(True, which="major", axis="x", color=GRID, lw=0.6)
a1.set_title("(a)", loc="left")

# (b) draws per round to resolve the median |D_i| at 3 sigma
a2.plot(ns, [res[n][3] for n in NS], "-", color=C[0], lw=2, marker="o", ms=4, label="device, post-selected shots")
a2.plot(ns, [res[n][4] for n in NS], "-", color=C[1], lw=2, marker="s", ms=4, label="classical, control variate")
for col, expo, lab in ((3, 3, "$\\propto n^3$"), (4, 1, "$\\propto n$")):
    ref = res[NS[0]][col] * (ns / ns[0]) ** expo
    a2.plot(ns, ref, ":", color=INK, lw=1)
    a2.text(ns[-1] * 1.04, ref[-1], lab, color=INK, fontsize=9, va="center")
logx(a2, (20, 30, 50, 100))
a2.set_xlim(18, 130)
a2.set_yscale("log")
a2.set_ylabel("draws per round (3$\\sigma$, median $|D_i|$)")
a2.legend(fontsize=9.5, loc="center", bbox_to_anchor=(0.45, 0.5))
a2.grid(True, which="major", axis="y", color=GRID, lw=0.6)
a2.set_title("(b)", loc="left")

# panels for the threshold sweep moved to app_flow (large_n_flow_report.py)
fig.tight_layout()
out = os.path.join(HERE, "..", "paper", "figures", "app_large_n.pdf")
fig.savefig(out)
if os.environ.get("PREVIEW"):
    fig.savefig(os.environ["PREVIEW"], dpi=110)
print("figure written to", os.path.relpath(out, os.path.dirname(HERE)))
