"""Experiment 3: phase-map ablation.

Maps (vqpm.phase_map_coefficients): paper E*pi/(4S)+pi/4, tight
(E-L)/(U-L)*pi/2, tight_wide (E-L)/(U-L)*0.9pi, shifted E*pi/(4S)+3pi/4.
Both stop rules, fixed pdiff, one trajectory per (n, instance, map) read off
at each budget (same machinery as vqpm_grid.py).

Example:
    python experiments/phase_map_ablation.py --n 8 10 12 14 16 --instances 100 --workers 14
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import numpy as np  # noqa: E402

import vqpm  # noqa: E402
import vqpm_grid as G  # noqa: E402


def plot(summ, out_prefix, budget):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [("found_optimum_rate", "Found-optimum rate"), ("mean_p_optimum", "Mean P(optimum)"),
               ("mean_norm_energy_error", "Mean normalised energy error"),
               ("mean_iterations", "Mean iterations used")]
    fig, axes = plt.subplots(2, len(metrics), figsize=(4.2 * len(metrics), 7), squeeze=False)
    for a, rule in enumerate(G.STOP_RULES):
        for m, (key, title) in enumerate(metrics):
            ax = axes[a, m]
            for pm in vqpm.PHASE_MAPS:
                s = sorted([r for r in summ if r["phase_map"] == pm and r["stop_rule"] == rule
                            and r["budget"] == budget], key=lambda r: r["n"])
                if s:
                    ax.plot([r["n"] for r in s], [r[key] for r in s], "o-", label=pm)
            ax.set_title(f"{title}\n{rule}, max_iter={budget}", fontsize=10)
            ax.set_xlabel("n")
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_prefix}.{ext}", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, nargs="+", default=[8, 10, 12, 14, 16])
    ap.add_argument("--instances", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--phase-maps", nargs="+", default=list(vqpm.PHASE_MAPS), choices=vqpm.PHASE_MAPS)
    ap.add_argument("--precision", default="3")
    ap.add_argument("--budgets", nargs="+", default=["30", "100"])
    ap.add_argument("--pdiff", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--inst-dir", default=common.DEFAULT_INSTANCE_DIR)
    ap.add_argument("--out", default="experiments/results/phase_map/raw/phase_map.jsonl")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    prec = G.parse_precision(args.precision)
    budgets = [b if b.endswith("n") else int(b) for b in args.budgets]
    if not args.plot_only:
        tasks = []
        for n in sorted(args.n, reverse=True):
            common.get_instances(n, args.instances, args.seed, args.inst_dir)
            for i in range(args.instances):
                for pm in args.phase_maps:
                    tasks.append(dict(key=f"n{n}_i{i}_p{G.prec_label(prec)}_pm{pm}_pd{args.pdiff}",
                                      n=n, i=i, precision=prec, count=args.instances, seed=args.seed,
                                      inst_dir=args.inst_dir, budgets=budgets, pdiff=args.pdiff,
                                      phase_map=pm))
        store = common.JsonlStore(args.out, meta=common.run_metadata(args))
        secs = common.run_tasks(G.run_task, tasks, store, workers=args.workers, log_every=20)
        print(f"done in {secs:.1f}s", flush=True)

    _, recs = common.load_jsonl(args.out)
    rows = G.flatten(recs, inst_dir=args.inst_dir)
    sdir = os.path.dirname(os.path.dirname(args.out)) if os.path.basename(
        os.path.dirname(args.out)) == "raw" else os.path.dirname(args.out)
    os.makedirs(sdir, exist_ok=True)
    base = os.path.join(sdir, "phase_map")
    summ = G.summarise(rows)
    common.write_csv(base + "_summary.csv", summ)
    common.write_csv(base + "_runs.csv", rows)
    for b in args.budgets:
        plot(summ, f"{base}_max_iter{b}", str(b))
        for rule in G.STOP_RULES:
            series = []
            for pm in vqpm.PHASE_MAPS:
                byx = {}
                for r in rows:
                    if r["phase_map"] == pm and r["stop_rule"] == rule and r["budget"] == str(b):
                        byx.setdefault(r["n"], []).append(r)
                if byx:
                    series.append((pm, {"marker": "o", "ms": 3}, byx))
            common.plot_metric_panels(series, f"{base}_metrics_{rule}_max_iter{b}",
                                      f"Phase-map ablation: {rule}, max_iter={b}")
    print(f"wrote {base}_summary.csv and plots", flush=True)
    import reports

    reports.on_finish("phase_map")


if __name__ == "__main__":
    main()
