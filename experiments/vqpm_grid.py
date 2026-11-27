"""Experiment 1 (+2, +budget sweep): VQPM baseline grid.

For every n, instance and rounding precision, VQPM is run ONCE with
stop_rule='none' for the largest iteration budget (exact marginals, fixed
pdiff, fast-forward at fixed points) while recording the trajectory and the
lock log. The outcome under each stop rule ('oracle', 'max_prob') and each
budget (e.g. 30, 100, 300, '10n') is then read off that trajectory with
common.derive_result; this is exact because the dynamics do not depend on
the stop rule or on the budget (tested in tests/test_vqpm_options.py).

Recorded per (n, instance, precision, stop rule, budget): found optimum,
P(optimum) at the end, relative / normalised energy error of the returned
state, Hamming distance, iterations used, stop reason, locks, wrong locks,
first wrong-lock iteration, failure type (wrong_lock / not_converged /
wrong_stop) and a stall flag (no change of marginals or locks over the last
10 iterations before the budget ran out).

Examples:
    python experiments/vqpm_grid.py --n 4 6 8 --instances 5 --out experiments/results/smoke/grid.jsonl
    python experiments/vqpm_grid.py --n $(seq 4 20) --instances 100 --workers 14 \\
        --out experiments/results/vqpm_grid/raw/grid.jsonl
    python experiments/vqpm_grid.py --plot-only --out experiments/results/vqpm_grid/raw/grid.jsonl
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402  (sets single-threaded BLAS first)
import numpy as np  # noqa: E402

import vqpm  # noqa: E402

STOP_RULES = ("oracle", "max_prob")


def parse_precision(s):
    return None if s in ("none", "None", "exact") else int(s)


def prec_label(p):
    return "exact" if p is None else str(p)


def run_task(task):
    n, i, prec = task["n"], task["i"], task["precision"]
    inst = common.get_instances(n, task["count"], task["seed"], task["inst_dir"])
    Q = inst["Q"][i]
    opt, e_min, e_max = int(inst["optimum"][i]), float(inst["e_min"][i]), float(inst["e_max"][i])
    _, u, _, target = vqpm.adjust_unitary_phase(n, Q, phase_map=task["phase_map"])
    assert int(target) == opt
    budgets = [vqpm.resolve_max_iter(b, n) for b in task["budgets"]]
    T = max(budgets)
    import time

    t0 = time.time()
    E = vqpm.qubo_energies(n, Q)  # original (unscaled) units
    r = vqpm.vqpm_for_qubo(
        u, n, expected_idx=opt, max_iter=T, pdiff=task["pdiff"], precision=prec,
        stop_rule="none", return_log=True, record_trajectory=True,
        fast_forward_fixed_point=True, verbose=False, energies=E,
    )
    secs = time.time() - t0
    info = r[6]
    traj = info["trajectory"]
    results = []
    for label, b in zip(task["budgets"], budgets):
        for rule in STOP_RULES:
            d = common.derive_result(traj, info["lock_log"], n, opt, rule, b)
            d.update(common.outcome_metrics(n, Q, d["found_state"], opt, e_min, e_max,
                                            d["expected_energy"]))
            if d["expected_energy_is_state"]:
                d["expected_energy"] = d["energy"]
                d["norm_expected_energy_gap"] = d["norm_energy_gap"]
            d.update(stop_rule=rule, budget=str(label), budget_iters=b)
            results.append(d)
    # store the trajectory up to the fixed point only (the rest repeats)
    L = info["fast_forwarded_from"] or len(traj["target_prob"])
    return {
        "key": task["key"],
        "n": n, "instance": i, "seed": task["seed"], "precision": prec_label(prec),
        "pdiff": task["pdiff"], "phase_map": task["phase_map"], "optimum": opt,
        "gap": float(inst["gap"][i]), "max_budget": T, "seconds": secs,
        "fixed_point_from": info["fast_forwarded_from"],
        "lock_log": info["lock_log"],
        "trajectory": {k: v[: L + 1] for k, v in traj.items()},
        "results": results,
    }


def flatten(recs, sidecar=None, inst_dir=common.DEFAULT_INSTANCE_DIR):
    """One row per (record, stop rule, budget). Energy metrics are recomputed
    from the stored instances (original Q units); expected energies of runs
    made before they were recorded come from the backfill sidecar."""
    sidecar = sidecar or {}
    rows = []
    for r in recs:
        side = sidecar.get(r["key"])
        for k, d in enumerate(r["results"]):
            row = {k2: r[k2] for k2 in ("n", "instance", "precision", "pdiff", "phase_map", "gap", "seed")}
            row.update({k2: v for k2, v in d.items()})
            if row.get("expected_energy") is None and side is not None:
                row["expected_energy"] = side["expected"][k]
                row["expected_energy_is_state"] = side["is_state"][k]
            rows.append(row)
    for row in rows:
        # resolving a marginal to d decimals needs ~10^(2d) shots per iteration
        if row["precision"] == "exact":
            row["shots_equiv_per_iter"] = None
            row["shots_equiv_total"] = None
        else:
            row["shots_equiv_per_iter"] = 10 ** (2 * int(row["precision"]))
            row["shots_equiv_total"] = row["shots_equiv_per_iter"] * row["iterations"]
    seeds = {row["seed"] for row in rows}
    for sd in seeds:
        common.attach_energy_metrics([row for row in rows if row["seed"] == sd], sd, inst_dir)
    return rows


def sidecar_path(out):
    return os.path.splitext(out)[0] + "_expected_energy.jsonl"


def load_sidecar(out):
    p = sidecar_path(out)
    if not os.path.exists(p):
        return {}
    _, recs = common.load_jsonl(p)
    return {r["key"]: r for r in recs}


def backfill_task(task):
    """Recompute the expected energy of the final state for a record made
    before energies were recorded (deterministic re-run, same outcome)."""
    rec = run_task(task)
    return {"key": task["key"],
            "expected": [d["expected_energy"] for d in rec["results"]],
            "is_state": [d["expected_energy_is_state"] for d in rec["results"]],
            "found_state": [d["found_state"] for d in rec["results"]]}


def summarise(rows, group=("phase_map", "precision", "stop_rule", "budget", "n")):
    """Per-group summary; includes common.metric_summary (found-optimum
    rate, bit difference and energy gaps with 95% intervals)."""
    import collections

    g = collections.defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in group)].append(r)
    out = []
    for key, rs in sorted(g.items(), key=lambda kv: tuple(
            (x if not isinstance(x, str) else (len(x), x)) for x in kv[0])):
        fails = [r for r in rs if not r["success"]]
        out.append({
            **dict(zip(group, key)),
            "runs": len(rs),
            **common.metric_summary(rs),
            "found_optimum_rate": np.mean([r["success"] for r in rs]),
            "mean_p_optimum": np.mean([r["target_prob"] for r in rs]),
            "mean_rel_energy_error": np.nanmean([r["rel_energy_error"] for r in rs]),
            "mean_norm_energy_error": np.mean([r["norm_energy_error"] for r in rs]),
            "mean_bit_diff": np.mean([r["bit_diff"] for r in rs]),
            "mean_iterations": np.mean([r["iterations"] for r in rs]),
            "mean_shots_equiv_total": (np.mean([r["shots_equiv_total"] for r in rs])
                                       if rs[0].get("shots_equiv_total") is not None else np.nan),
            "hit_budget_rate": np.mean([r["stop_reason"] == "max_iter" for r in rs]),
            "stalled_rate": np.mean([r["stalled"] for r in rs]),
            "mean_locks": np.mean([r["n_locks"] for r in rs]),
            "runs_with_wrong_lock": np.mean([r["n_wrong_locks"] > 0 for r in rs]),
            "wrong_lock_rate_per_lock": (sum(r["n_wrong_locks"] for r in rs)
                                         / max(1, sum(r["n_locks"] for r in rs))),
            "mean_first_wrong_lock_iter": (np.mean([r["first_wrong_lock_iter"] for r in rs
                                                    if r["first_wrong_lock_iter"] is not None])
                                           if any(r["first_wrong_lock_iter"] is not None for r in rs)
                                           else np.nan),
            "fail_wrong_lock": sum(r["failure"] == "wrong_lock" for r in fails) / len(rs),
            "fail_not_converged": sum(r["failure"] == "not_converged" for r in fails) / len(rs),
            "fail_wrong_stop": sum(r["failure"] == "wrong_stop" for r in fails) / len(rs),
        })
    return out


def plot(rows, out_prefix):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summ = summarise(rows)
    precs = sorted({r["precision"] for r in summ}, key=lambda s: (s == "exact", s))
    bud_sorted = sorted({r["budget"] for r in rows},
                        key=lambda s: (s.endswith("n"), int(s[:-1]) if s.endswith("n") else int(s)))
    for prec in precs:
        for rule in STOP_RULES:
            series = []
            for b in bud_sorted:
                byx = {}
                for r in rows:
                    if r["precision"] == prec and r["stop_rule"] == rule and r["budget"] == b:
                        byx.setdefault(r["n"], []).append(r)
                series.append((f"max_iter={b}", {"marker": "o", "ms": 3}, byx))
            common.plot_metric_panels(series, f"{out_prefix}_metrics_{rule}_prec{prec}",
                                      f"VQPM grid: stop rule {rule}, precision {prec}")
    budgets = sorted({r["budget"] for r in summ}, key=lambda s: (s.endswith("n"), int(s[:-1]) if s.endswith("n") else int(s)))
    metrics = [("found_optimum_rate", "Found-optimum rate"),
               ("mean_p_optimum", "Mean P(optimum)"),
               ("mean_norm_energy_error", "Mean normalised energy error"),
               ("hit_budget_rate", "Runs hitting the budget")]
    for rule in STOP_RULES:
        fig, axes = plt.subplots(len(precs), len(metrics), figsize=(4.2 * len(metrics), 3.2 * len(precs)),
                                 squeeze=False, sharex=True)
        for a, prec in enumerate(precs):
            for m, (key, title) in enumerate(metrics):
                ax = axes[a, m]
                for b in budgets:
                    s = [r for r in summ if r["precision"] == prec and r["stop_rule"] == rule and r["budget"] == b]
                    s.sort(key=lambda r: r["n"])
                    ax.plot([r["n"] for r in s], [r[key] for r in s], "o-", label=f"max_iter={b}")
                ax.set_title(f"{title}\nprecision={prec}", fontsize=10)
                ax.grid(alpha=0.3)
                if a == len(precs) - 1:
                    ax.set_xlabel("n")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"VQPM grid, stop rule = {rule}")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(f"{out_prefix}_{rule}.{ext}", dpi=150)
        plt.close(fig)

    # experiment 2: wrong locks and failure split
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for prec in precs:
        for rule in STOP_RULES:
            b = budgets[-1]
            s = sorted([r for r in summ if r["precision"] == prec and r["stop_rule"] == rule
                        and r["budget"] == b], key=lambda r: r["n"])
            lab = f"prec={prec}, {rule}"
            axes[0].plot([r["n"] for r in s], [r["runs_with_wrong_lock"] for r in s], "o-", label=lab)
            axes[1].plot([r["n"] for r in s], [r["wrong_lock_rate_per_lock"] for r in s], "o-", label=lab)
            axes[2].plot([r["n"] for r in s], [r["mean_first_wrong_lock_iter"] for r in s], "o-", label=lab)
    axes[0].set_title(f"Runs with >= 1 wrong lock (max_iter={budgets[-1]})")
    axes[1].set_title("Wrong locks / all locks")
    axes[2].set_title("Mean iteration of first wrong lock")
    for ax in axes:
        ax.set_xlabel("n")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_prefix}_wrong_locks.{ext}", dpi=150)
    plt.close(fig)

    # failure split (stacked) for the oracle rule, largest budget, each precision
    fig, axes = plt.subplots(1, len(precs), figsize=(5 * len(precs), 4), squeeze=False)
    for a, prec in enumerate(precs):
        ax = axes[0, a]
        for rule, hatch in zip(STOP_RULES, ("", "//")):
            s = sorted([r for r in summ if r["precision"] == prec and r["stop_rule"] == rule
                        and r["budget"] == budgets[-1]], key=lambda r: r["n"])
            ns = np.array([r["n"] for r in s], dtype=float) + (-0.2 if rule == "oracle" else 0.2)
            bottom = np.zeros(len(s))
            for key, col in (("fail_wrong_lock", "C3"), ("fail_not_converged", "C1"), ("fail_wrong_stop", "C7")):
                v = np.array([r[key] for r in s])
                ax.bar(ns, v, width=0.4, bottom=bottom, color=col, hatch=hatch, edgecolor="k", linewidth=0.3,
                       label=f"{key[5:]} ({rule})")
                bottom += v
        ax.set_title(f"Failure split, precision={prec}, max_iter={budgets[-1]}")
        ax.set_xlabel("n")
        ax.set_ylabel("fraction of runs")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_prefix}_failures.{ext}", dpi=150)
    plt.close(fig)

    # margins of correct vs wrong locks
    return summ


def lock_margin_table(recs):
    rows = []
    for r in recs:
        for e in r["lock_log"]:
            rows.append({"n": r["n"], "instance": r["instance"], "precision": r["precision"],
                         **{k: e[k] for k in ("iteration", "qubit", "value", "margin", "threshold",
                                               "matches_optimum")}})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, nargs="+", default=list(range(4, 21)))
    ap.add_argument("--instances", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--precision", nargs="+", default=["3", "6", "exact"])
    ap.add_argument("--budgets", nargs="+", default=["30", "100", "300", "10n"])
    ap.add_argument("--pdiff", type=float, default=0.01)
    ap.add_argument("--phase-maps", nargs="+", default=["paper"], choices=vqpm.PHASE_MAPS)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--inst-dir", default=common.DEFAULT_INSTANCE_DIR)
    ap.add_argument("--out", default="experiments/results/vqpm_grid/raw/grid.jsonl")
    ap.add_argument("--summary-dir", default=None,
                    help="where summary CSV/plots go (default: parent of raw/, or next to --out)")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--backfill-expected-energy", action="store_true",
                    help="re-run records saved without expected energies (deterministic) and "
                         "store them in a sidecar file; implies --plot-only for new tasks")
    args = ap.parse_args()
    if args.backfill_expected_energy:
        args.plot_only = True

    precs = [parse_precision(p) for p in args.precision]
    budgets = [b if b.endswith("n") else int(b) for b in args.budgets]
    if not args.plot_only:
        tasks = []
        # biggest n first so the slow tasks start early
        for n in sorted(args.n, reverse=True):
            common.get_instances(n, args.instances, args.seed, args.inst_dir)
            for i in range(args.instances):
                for p in precs:
                    for pm in args.phase_maps:
                        tasks.append(dict(
                            key=f"n{n}_i{i}_p{prec_label(p)}_pm{pm}_pd{args.pdiff}",
                            n=n, i=i, precision=p, count=args.instances, seed=args.seed,
                            inst_dir=args.inst_dir, budgets=budgets, pdiff=args.pdiff,
                            phase_map=pm))
        store = common.JsonlStore(args.out, meta=common.run_metadata(args))
        secs = common.run_tasks(run_task, tasks, store, workers=args.workers, log_every=10)
        print(f"done in {secs:.1f}s", flush=True)

    if args.backfill_expected_energy:
        _, recs = common.load_jsonl(args.out)
        need = [r for r in recs if r["results"] and "expected_energy" not in r["results"][0]]
        tasks = [dict(key=r["key"], n=r["n"], i=r["instance"],
                      precision=parse_precision(r["precision"]), count=args.instances,
                      seed=r["seed"], inst_dir=args.inst_dir,
                      budgets=[b if b.endswith("n") else int(b) for b in
                               dict.fromkeys(d["budget"] for d in r["results"])],
                      pdiff=r["pdiff"], phase_map=r["phase_map"]) for r in need]
        side = common.JsonlStore(sidecar_path(args.out), meta=common.run_metadata(args))
        common.run_tasks(backfill_task, tasks, side, workers=args.workers, log_every=50)

    _, recs = common.load_jsonl(args.out)
    rows = flatten(recs, load_sidecar(args.out), args.inst_dir)
    sdir = args.summary_dir or (os.path.dirname(os.path.dirname(args.out))
                                if os.path.basename(os.path.dirname(args.out)) == "raw"
                                else os.path.dirname(args.out))
    os.makedirs(sdir, exist_ok=True)
    base = os.path.join(sdir, "vqpm_grid")
    common.write_csv(base + "_runs.csv", rows)
    summ = plot(rows, base)
    common.write_csv(base + "_summary.csv", summ)
    common.write_csv(base + "_locks.csv", lock_margin_table(recs))
    timing = {}
    for r in recs:
        timing.setdefault((r["n"], r["precision"]), []).append(r["seconds"])
    common.write_csv(base + "_timing.csv", [
        {"n": n, "precision": p, "runs": len(v), "mean_seconds": np.mean(v), "max_seconds": np.max(v)}
        for (n, p), v in sorted(timing.items(), key=lambda kv: (kv[0][0], str(kv[0][1])))])
    print(f"wrote {base}_summary.csv and plots", flush=True)
    import reports

    reports.on_finish("vqpm_grid")


if __name__ == "__main__":
    main()
