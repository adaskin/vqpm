"""Experiment 2: wrong-lock statistics from the lock logs of vqpm_grid.py.

Reads the grid JSONL (no new simulation) and reports, per precision and n:
wrong-lock rate per lock, fraction of runs with a wrong lock (over the whole
recorded trajectory and up to the stop of each stop rule / budget), the
iteration of the first wrong lock, and the lock margin |P0 - P1| of correct
vs wrong locks. The per-run failure split (wrong lock / not converged /
wrong stop) is in vqpm_grid_summary.csv.

Example:
    python experiments/wrong_locks.py --grid experiments/results/vqpm_grid/raw/grid.jsonl
"""

import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default="experiments/results/vqpm_grid/raw/grid.jsonl")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    _, recs = common.load_jsonl(args.grid)
    out_dir = args.out_dir or os.path.dirname(os.path.dirname(args.grid))
    os.makedirs(out_dir, exist_ok=True)

    g = collections.defaultdict(list)
    for r in recs:
        g[(r["precision"], r["n"])].append(r)
    rows = []
    for (prec, n), rs in sorted(g.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        locks = [e for r in rs for e in r["lock_log"]]
        wrong = [e for e in locks if e["matches_optimum"] is False]
        right = [e for e in locks if e["matches_optimum"] is True]
        first = [min(e["iteration"] for e in r["lock_log"] if e["matches_optimum"] is False)
                 for r in rs if any(e["matches_optimum"] is False for e in r["lock_log"])]
        first_lock = [min(e["iteration"] for e in r["lock_log"]) for r in rs if r["lock_log"]]
        rows.append({
            "precision": prec, "n": n, "runs": len(rs), "locks": len(locks),
            "wrong_lock_rate_per_lock": len(wrong) / max(1, len(locks)),
            "runs_with_wrong_lock": len(first) / len(rs),
            "mean_wrong_locks_per_run": len(wrong) / len(rs),
            "median_first_wrong_lock_iter": float(np.median(first)) if first else np.nan,
            "median_first_lock_iter": float(np.median(first_lock)) if first_lock else np.nan,
            "median_margin_correct": float(np.median([e["margin"] for e in right])) if right else np.nan,
            "median_margin_wrong": float(np.median([e["margin"] for e in wrong])) if wrong else np.nan,
            "frac_wrong_locks_at_first_lock_iter": (
                np.mean([e["iteration"] == min(x["iteration"] for x in r["lock_log"])
                         for r in rs for e in r["lock_log"] if e["matches_optimum"] is False])
                if wrong else np.nan),
        })
    base = os.path.join(out_dir, "wrong_locks")
    common.write_csv(base + "_summary.csv", rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    precs = sorted({r["precision"] for r in rows}, key=lambda s: (s == "exact", s))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for prec in precs:
        s = sorted([r for r in rows if r["precision"] == prec], key=lambda r: r["n"])
        x = [r["n"] for r in s]
        axes[0].plot(x, [r["wrong_lock_rate_per_lock"] for r in s], "o-", label=f"precision {prec}")
        axes[1].plot(x, [r["runs_with_wrong_lock"] for r in s], "o-", label=f"precision {prec}")
        axes[2].plot(x, [r["median_first_wrong_lock_iter"] for r in s], "o-", label=f"precision {prec}")
    axes[0].set_title("Wrong locks / all locks (whole trajectory)")
    axes[1].set_title("Runs with >= 1 wrong lock")
    axes[2].set_title("Median iteration of first wrong lock")
    for ax in axes:
        ax.set_xlabel("n")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(base + "_vs_n." + ext, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(precs), figsize=(5 * len(precs), 4), squeeze=False)
    for a, prec in enumerate(precs):
        locks = [e for r in recs if r["precision"] == prec for e in r["lock_log"]]
        ok = [e["margin"] for e in locks if e["matches_optimum"]]
        bad = [e["margin"] for e in locks if e["matches_optimum"] is False]
        bins = np.logspace(-3, 0, 31)
        axes[0, a].hist([ok, bad], bins=bins, label=["correct", "wrong"], density=True)
        axes[0, a].set_xscale("log")
        axes[0, a].set_title(f"Lock margin |P0-P1|, precision {prec}")
        axes[0, a].legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(base + "_margins." + ext, dpi=150)
    plt.close(fig)
    print(f"wrote {base}_summary.csv and plots")
    import reports

    reports.on_finish("wrong_locks")


if __name__ == "__main__":
    main()
