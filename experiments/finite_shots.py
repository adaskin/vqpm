"""Experiment 4: finite-shot VQPM.

Marginals are estimated from `shots` circuit runs per iteration
(post-selection on the control, multinomial sampling; see
vqpm.vqpm_for_qubo(shots=...)). Locking: fixed pdiff, or the union-bound
Hoeffding threshold ('hoeffding_union', delta). Both stop rules; each run
is simulated directly (the sampled max-probability rule depends on the
samples). The same sampling seed is used for both stop rules.

Recorded: found optimum (most frequent sample of the last iteration;
also via locked bits + marginal majority, and via the exact output
distribution), P(optimum), energy error, iterations, shots used,
post-selected shots, locks and wrong locks.

Example:
    python experiments/finite_shots.py --n 8 10 12 14 16 --instances 100 \\
        --shots 1000 10000 100000 1000000 --workers 14
"""

import argparse
import collections
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import numpy as np  # noqa: E402

import vqpm  # noqa: E402

RULES = ("oracle", "max_prob")


def run_task(task):
    n, i = task["n"], task["i"]
    inst = common.get_instances(n, task["count"], task["seed"], task["inst_dir"])
    Q = inst["Q"][i]
    opt, e_min, e_max = int(inst["optimum"][i]), float(inst["e_min"][i]), float(inst["e_max"][i])
    _, u, _, _ = vqpm.adjust_unitary_phase(n, Q)
    E = vqpm.qubo_energies(n, Q)
    pol = task["policy"]
    out = []
    for rule in RULES:
        t0 = time.time()
        r = vqpm.vqpm_for_qubo(
            u, n, expected_idx=opt, max_iter=task["max_iter"], pdiff=task["pdiff"],
            precision=task["precision"],
            dynamic_pdiff_policy="hoeffding_union" if pol == "hoeffding_union" else None,
            delta=task["delta"], stop_rule=rule, shots=task["shots"],
            rng=[task["seed"], n, i, task["shots"], 7], return_log=True, verbose=False,
            energies=E)
        info = r[6]
        wrong = [e for e in info["lock_log"] if e["matches_optimum"] is False]
        rec = {
            "key": f"{task['key']}_{rule}",
            "n": n, "instance": i, "seed": task["seed"], "shots": task["shots"], "policy": pol,
            "stop_rule": rule,
            "max_iter": task["max_iter"], "precision": task["precision"], "pdiff": task["pdiff"],
            "delta": task["delta"],
            "found_state": int(r[0]),
            "success": int(r[0]) == opt,
            "success_marginals": info["found_state_marginals"] == opt,
            "success_exact_argmax": info["found_state_exact"] == opt,
            "target_prob": float(r[5][-1]) if len(r[5]) else float("nan"),
            "final_target_prob": info.get("final_target_prob"),
            "iterations": info["iterations"],
            "stop_reason": info["stop_reason"],
            "shots_used": info["shots_used"],
            "postselected_shots": info["postselected_shots"],
            "n_locks": info["n_locked"],
            "n_wrong_locks": len(wrong),
            "first_wrong_lock_iter": min((e["iteration"] for e in wrong), default=None),
            "lock_log": info["lock_log"],
            "seconds": time.time() - t0,
        }
        rec.update(common.outcome_metrics(n, Q, int(r[0]), opt, e_min, e_max,
                                          info.get("final_expected_energy")))
        out.append(rec)
    return out


def summarise(recs):
    g = collections.defaultdict(list)
    for r in recs:
        g[(r["policy"], r["stop_rule"], r["shots"], r["n"])].append(r)
    rows = []
    for (pol, rule, shots, n), rs in sorted(g.items()):
        rows.append({
            "policy": pol, "stop_rule": rule, "shots": shots, "n": n, "runs": len(rs),
            **common.metric_summary(rs),
            "found_optimum_rate": np.mean([r["success"] for r in rs]),
            "found_optimum_rate_marginals": np.mean([r["success_marginals"] for r in rs]),
            "found_optimum_rate_exact_argmax": np.mean([r["success_exact_argmax"] for r in rs]),
            "mean_p_optimum": np.mean([r["final_target_prob"] if r["final_target_prob"] is not None
                                       else r["target_prob"] for r in rs]),
            "mean_norm_energy_error": np.mean([r["norm_energy_error"] for r in rs]),
            "mean_bit_diff": np.mean([r["bit_diff"] for r in rs]),
            "mean_iterations": np.mean([r["iterations"] for r in rs]),
            "mean_shots_used": np.mean([r["shots_used"] for r in rs]),
            "mean_locks": np.mean([r["n_locks"] for r in rs]),
            "runs_with_wrong_lock": np.mean([r["n_wrong_locks"] > 0 for r in rs]),
            "wrong_lock_rate_per_lock": sum(r["n_wrong_locks"] for r in rs) / max(1, sum(r["n_locks"] for r in rs)),
            "hit_budget_rate": np.mean([r["stop_reason"] == "max_iter" for r in rs]),
        })
    return rows


def plot(summ, out_prefix):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pols = sorted({r["policy"] for r in summ})
    ns = sorted({r["n"] for r in summ})
    metrics = [("found_optimum_rate", "Found-optimum rate"), ("runs_with_wrong_lock", "Runs with a wrong lock"),
               ("mean_shots_used", "Mean total shots"), ("mean_iterations", "Mean iterations")]
    for rule in RULES:
        fig, axes = plt.subplots(len(pols), len(metrics), figsize=(4.2 * len(metrics), 3.4 * len(pols)),
                                 squeeze=False)
        for a, pol in enumerate(pols):
            for m, (key, title) in enumerate(metrics):
                ax = axes[a, m]
                for n in ns:
                    s = sorted([r for r in summ if r["policy"] == pol and r["stop_rule"] == rule
                                and r["n"] == n], key=lambda r: r["shots"])
                    ax.plot([r["shots"] for r in s], [r[key] for r in s], "o-", label=f"n={n}")
                ax.set_xscale("log")
                if key == "mean_shots_used":
                    ax.set_yscale("log")
                ax.set_xlabel("shots per iteration")
                ax.set_title(f"{title}\n{pol}, {rule}", fontsize=10)
                ax.grid(alpha=0.3)
        axes[0, 0].legend(fontsize=8)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(f"{out_prefix}_{rule}.{ext}", dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, nargs="+", default=[8, 10, 12, 14, 16])
    ap.add_argument("--instances", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--shots", type=int, nargs="+", default=[1000, 10000, 100000, 1000000])
    ap.add_argument("--policies", nargs="+", default=["fixed", "hoeffding_union"],
                    choices=["fixed", "hoeffding_union"])
    ap.add_argument("--pdiff", type=float, default=0.01)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--precision", type=int, default=3)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--inst-dir", default=common.DEFAULT_INSTANCE_DIR)
    ap.add_argument("--out", default="experiments/results/finite_shots/raw/finite_shots.jsonl")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    if not args.plot_only:
        tasks = []
        for n in sorted(args.n, reverse=True):
            common.get_instances(n, args.instances, args.seed, args.inst_dir)
            for i in range(args.instances):
                for shots in sorted(args.shots, reverse=True):
                    for pol in args.policies:
                        tasks.append(dict(key=f"n{n}_i{i}_s{shots}_{pol}_T{args.max_iter}",
                                          n=n, i=i, shots=shots, policy=pol, count=args.instances,
                                          seed=args.seed, inst_dir=args.inst_dir, pdiff=args.pdiff,
                                          delta=args.delta, precision=args.precision,
                                          max_iter=args.max_iter))
        store = common.JsonlStore(args.out, meta=common.run_metadata(args))
        # each task writes one record per stop rule; resume on the oracle key
        store.done = {k[: -len("_oracle")] for k in store.done if k.endswith("_oracle")} & {
            k[: -len("_max_prob")] for k in store.done if k.endswith("_max_prob")}
        secs = common.run_tasks(run_task, tasks, store, workers=args.workers, log_every=20)
        print(f"done in {secs:.1f}s", flush=True)

    _, recs = common.load_jsonl(args.out)
    common.attach_energy_metrics(recs, args.seed, args.inst_dir)
    sdir = os.path.dirname(os.path.dirname(args.out)) if os.path.basename(
        os.path.dirname(args.out)) == "raw" else os.path.dirname(args.out)
    os.makedirs(sdir, exist_ok=True)
    base = os.path.join(sdir, "finite_shots")
    summ = summarise(recs)
    common.write_csv(base + "_summary.csv", summ)
    common.write_csv(base + "_runs.csv", [{k: v for k, v in r.items() if k != "lock_log"} for r in recs])
    plot(summ, base)
    for pol in sorted({r["policy"] for r in recs}):
        for rule in RULES:
            series = []
            for n in sorted({r["n"] for r in recs}):
                byx = {}
                for r in recs:
                    if r["policy"] == pol and r["stop_rule"] == rule and r["n"] == n:
                        byx.setdefault(r["shots"], []).append(r)
                series.append((f"n={n}", {"marker": "o", "ms": 3}, byx))
            common.plot_metric_panels(series, f"{base}_metrics_{pol}_{rule}",
                                      f"Finite shots: {pol}, {rule}", xlabel="shots per iteration",
                                      logx=True)
    print(f"wrote {base}_summary.csv and plots", flush=True)
    import reports

    reports.on_finish("finite_shots")


if __name__ == "__main__":
    main()
