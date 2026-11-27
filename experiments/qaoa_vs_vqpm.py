"""Experiment 5: QAOA vs VQPM (+ simulated annealing) on the same instances.

QAOA (comparison_to_qaoa.QAOA, statevector, on the paper's scaled QUBO
Q_adj) for each optimizer in {L-BFGS-B, COBYLA, gd, adam}, depths p,
`--restarts` random starts (best energy kept) and INTERP initialisation
(depth chain 1..max p, `--restarts` random depth-1 starts), all with the
same iteration cap; evaluations are recorded (nfev, adjoint-gradient
evaluations, and evals_fd_equiv = nfev + ngrad*(2p+1)). Random initial
angles are shared across optimizers (same seed per instance and depth).

VQPM: exact marginals, pdiff 0.01, precision 3, stop rules 'oracle' and
'max_prob', budgets 30 and 100 (one trajectory, read off at each budget).
SA: baselines.simulated_annealing with 10, 100, 1000 sweeps.

Metrics: found optimum (argmax of the output distribution / returned
state), P(optimum), normalised energy error of the returned state.

Optimizer specs: NAME[#GRADIENT][@LR], e.g. 'L-BFGS-B#fd', 'gd@0.3'.

Examples:
    # optimizer pilot at n = 8
    python experiments/qaoa_vs_vqpm.py --n 8 --instances 20 --p 4 8 \\
        --optimizers L-BFGS-B L-BFGS-B#fd COBYLA gd adam --out experiments/results/qaoa_pilot/raw/pilot.jsonl
    # main run
    python experiments/qaoa_vs_vqpm.py --n 4 6 8 10 12 14 16 --workers 14
"""

import argparse
import collections
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import numpy as np  # noqa: E402

import baselines  # noqa: E402
import comparison_to_qaoa as C  # noqa: E402
import vqpm  # noqa: E402

INIT_CODE = {"random": 0, "interp": 1}


def parse_opt(spec):
    name, lr, grad = spec, None, None
    if "@" in name:
        name, lr = name.split("@")
        lr = float(lr)
    if "#" in name:
        name, grad = name.split("#")
    if name not in C.QAOA_OPTIMIZERS:
        raise ValueError(f"unknown optimizer {name}")
    if grad is None and name != "COBYLA":
        grad = "adjoint"
    return name, grad, lr


def n_instances(n, args):
    return args.instances_small if n <= args.small_n else args.instances


def load(task):
    inst = common.get_instances(task["n"], task["count"], task["seed"], task["inst_dir"])
    i = task["i"]
    return (inst["Q"][i], int(inst["optimum"][i]), float(inst["e_min"][i]), float(inst["e_max"][i]))


def run_task(task):
    kind = task["kind"]
    n, i = task["n"], task["i"]
    Q, opt, e_min, e_max = load(task)
    base = {"key": task["key"], "kind": kind, "n": n, "instance": i, "seed": task["seed"]}
    t0 = time.time()
    E = vqpm.qubo_energies(n, Q)  # original (unscaled) units
    if kind == "vqpm":
        _, u, _, _ = vqpm.adjust_unitary_phase(n, Q)
        budgets = task["budgets"]
        r = vqpm.vqpm_for_qubo(u, n, expected_idx=opt, max_iter=max(budgets), pdiff=0.01,
                               precision=3, stop_rule="none", return_log=True,
                               record_trajectory=True, fast_forward_fixed_point=True, verbose=False,
                               energies=E)
        out = []
        for b in budgets:
            for rule in ("oracle", "max_prob"):
                d = common.derive_result(r[6]["trajectory"], r[6]["lock_log"], n, opt, rule, b)
                rec = dict(base, key=f"{task['key']}_{rule}_{b}", method=f"VQPM ({rule}, T={b})",
                           stop_rule=rule, budget=b, success=d["success"], p_opt=d["target_prob"],
                           iterations=d["iterations"], n_wrong_locks=d["n_wrong_locks"],
                           evals=d["iterations"], seconds=time.time() - t0,
                           found_state=d["found_state"],
                           expected_energy_is_state=d["expected_energy_is_state"])
                rec.update(common.outcome_metrics(n, Q, d["found_state"], opt, e_min, e_max,
                                                  d["expected_energy"]))
                out.append(rec)
        return out
    if kind == "sa":
        out = []
        for sweeps in task["sweeps"]:
            r = baselines.simulated_annealing(Q, n_sweeps=sweeps, n_restarts=1,
                                              rng=[task["seed"], n, i, sweeps])
            rec = dict(base, key=f"{task['key']}_{sweeps}", method=f"SA ({sweeps} sweeps)",
                       sweeps=sweeps, success=r["state"] == opt, p_opt=float(r["state"] == opt),
                       evals=r["proposals"], seconds=time.time() - t0, found_state=r["state"])
            rec.update(common.outcome_metrics(n, Q, r["state"], opt, e_min, e_max))
            out.append(rec)
        return out
    # QAOA
    name, grad, lr = parse_opt(task["optimizer"])
    Q_adj, _, _, _ = vqpm.adjust_unitary_phase(n, Q)
    qaoa = C.QAOA(Q_adj)
    results = {}
    if task["init"] == "interp":
        chain = qaoa.run_interp(max(task["p"]), maxiter=task["maxiter"], n_restarts=task["restarts"],
                                optimizer=name, gradient=grad, lr=lr,
                                rng=[task["seed"], n, i, INIT_CODE["interp"]])
        results = {p: chain[p] for p in task["p"]}
        cost_note = "cumulative over the INTERP chain 1..p and all restarts"
    else:
        for p in task["p"]:
            qaoa.run(p=p, maxiter=task["maxiter"], n_restarts=task["restarts"], optimizer=name,
                     gradient=grad, lr=lr, rng=[task["seed"], n, i, INIT_CODE["random"], p])
            results[p] = qaoa.last_run
        cost_note = "summed over restarts"
    out = []
    for p, res in results.items():
        probs = qaoa.final_probs(res["params"], p)
        state = int(np.argmax(probs))
        rec = dict(base, key=f"{task['key']}_p{p}", method=f"QAOA {task['optimizer']} {task['init']} p={p}",
                   optimizer=task["optimizer"], init=task["init"], p=p, restarts=task["restarts"],
                   maxiter=task["maxiter"], success=state == opt, p_opt=float(probs[opt]),
                   energy_expectation=res["energy"], restart_energies=res["restart_energies"],
                   nfev=res["nfev_total"], ngrad=res["ngrad_total"], nit=res["nit_total"],
                   evals=res["evals_fd_equiv_total"], cost_note=cost_note,
                   seconds=time.time() - t0, found_state=state)
        rec.update(common.outcome_metrics(n, Q, state, opt, e_min, e_max,
                                          float(np.dot(probs, E))))
        out.append(rec)
    return out


def summarise(recs):
    g = collections.defaultdict(list)
    for r in recs:
        if r["kind"] == "qaoa":
            key = ("qaoa", r["optimizer"], r["init"], r["p"], r["n"])
        elif r["kind"] == "vqpm":
            key = ("vqpm", r["stop_rule"], str(r["budget"]), 0, r["n"])
        else:
            key = ("sa", str(r["sweeps"]), "", 0, r["n"])
        g[key].append(r)
    rows = []
    for (kind, a, b, p, n), rs in sorted(g.items(), key=lambda kv: tuple(str(x).zfill(6) for x in kv[0])):
        rows.append({
            "kind": kind, "variant": a, "init_or_budget": b, "p": p, "n": n, "runs": len(rs),
            "method": rs[0]["method"],
            **common.metric_summary(rs),
            "found_optimum_rate": np.mean([r["success"] for r in rs]),
            "mean_p_optimum": np.mean([r["p_opt"] for r in rs]),
            "mean_norm_energy_error": np.mean([r["norm_energy_error"] for r in rs]),
            "mean_bit_diff": np.mean([r["bit_diff"] for r in rs]),
            "mean_evals": np.mean([r["evals"] for r in rs]),
            "mean_nfev": np.mean([r.get("nfev", np.nan) for r in rs]),
            "mean_ngrad": np.mean([r.get("ngrad", np.nan) for r in rs]),
            "mean_seconds": np.mean([r["seconds"] for r in rs]),
        })
    return rows


def plot(summ, out_prefix):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    opts = sorted({r["variant"] for r in summ if r["kind"] == "qaoa"})
    ps = sorted({r["p"] for r in summ if r["kind"] == "qaoa"})
    metrics = [("found_optimum_rate", "Found-optimum rate"), ("mean_p_optimum", "Mean P(optimum)"),
               ("mean_norm_energy_error", "Mean normalised energy error"), ("mean_evals", "Evaluations")]
    fig, axes = plt.subplots(len(opts), len(metrics), figsize=(4.3 * len(metrics), 3.3 * len(opts)),
                             squeeze=False, sharex=True)
    cmap = plt.get_cmap("viridis")
    for a, opt in enumerate(opts):
        for m, (key, title) in enumerate(metrics):
            ax = axes[a, m]
            for k, p in enumerate(ps):
                for init, ls in (("random", "-"), ("interp", "--")):
                    s = sorted([r for r in summ if r["kind"] == "qaoa" and r["variant"] == opt
                                and r["init_or_budget"] == init and r["p"] == p], key=lambda r: r["n"])
                    if s:
                        ax.plot([r["n"] for r in s], [r[key] for r in s], ls, marker="o", ms=3,
                                color=cmap(k / max(1, len(ps) - 1) * 0.85),
                                label=f"QAOA p={p} {init}")
            for rule, col in (("oracle", "k"), ("max_prob", "0.45")):
                s = sorted([r for r in summ if r["kind"] == "vqpm" and r["variant"] == rule
                            and r["init_or_budget"] == "100"], key=lambda r: r["n"])
                if s:
                    ax.plot([r["n"] for r in s], [r[key] for r in s], "-s", color=col, lw=2.2, ms=4,
                            label=f"VQPM {rule} T=100")
            s = sorted([r for r in summ if r["kind"] == "sa" and r["variant"] == "100"], key=lambda r: r["n"])
            if s:
                ax.plot([r["n"] for r in s], [r[key] for r in s], ":^", color="C3", lw=2, ms=4,
                        label="SA 100 sweeps")
            if key == "mean_evals":
                ax.set_yscale("log")
            ax.set_title(f"{title}\noptimizer: {opt}", fontsize=10)
            ax.grid(alpha=0.3)
            if a == len(opts) - 1:
                ax.set_xlabel("n")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(6, len(l)), fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_prefix}.{ext}", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, nargs="+", default=[4, 6, 8, 10, 12, 14, 16])
    ap.add_argument("--instances", type=int, default=50)
    ap.add_argument("--instances-small", type=int, default=100,
                    help="instances for n <= --small-n")
    ap.add_argument("--small-n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--p", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--optimizers", nargs="+", default=["L-BFGS-B", "COBYLA", "gd", "adam"])
    ap.add_argument("--inits", nargs="+", default=["random", "interp"], choices=["random", "interp"])
    ap.add_argument("--restarts", type=int, default=5)
    ap.add_argument("--maxiter", type=int, default=500)
    ap.add_argument("--vqpm-budgets", type=int, nargs="+", default=[30, 100])
    ap.add_argument("--sa-sweeps", type=int, nargs="+", default=[10, 100, 1000])
    ap.add_argument("--skip", nargs="*", default=[], choices=["vqpm", "sa", "qaoa"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--inst-dir", default=common.DEFAULT_INSTANCE_DIR)
    ap.add_argument("--out", default="experiments/results/qaoa_vs_vqpm/raw/qaoa_vs_vqpm.jsonl")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    if not args.plot_only:
        tasks = []
        for n in sorted(args.n, reverse=True):
            count = n_instances(n, args)
            common.get_instances(n, count, args.seed, args.inst_dir)
            common_kw = dict(n=n, count=count, seed=args.seed, inst_dir=args.inst_dir)
            for i in range(count):
                if "qaoa" not in args.skip:
                    for opt in args.optimizers:
                        for init in args.inits:
                            tasks.append(dict(common_kw, kind="qaoa", i=i, optimizer=opt, init=init,
                                              p=args.p, restarts=args.restarts, maxiter=args.maxiter,
                                              key=f"qaoa_n{n}_i{i}_{opt}_{init}_r{args.restarts}_m{args.maxiter}"))
                if "vqpm" not in args.skip:
                    tasks.append(dict(common_kw, kind="vqpm", i=i, budgets=args.vqpm_budgets,
                                      key=f"vqpm_n{n}_i{i}"))
                if "sa" not in args.skip:
                    tasks.append(dict(common_kw, kind="sa", i=i, sweeps=args.sa_sweeps,
                                      key=f"sa_n{n}_i{i}"))
        store = common.JsonlStore(args.out, meta=common.run_metadata(args))
        # records carry suffixed keys; a task is done if any record starts with its key + "_"
        done_tasks = set()
        for t in tasks:
            pre = t["key"] + "_"
            if any(k.startswith(pre) for k in store.done):
                done_tasks.add(t["key"])
        store.done |= done_tasks
        secs = common.run_tasks(run_task, tasks, store, workers=args.workers, log_every=10)
        print(f"done in {secs:.1f}s", flush=True)

    _, recs = common.load_jsonl(args.out)
    common.attach_energy_metrics(recs, args.seed, args.inst_dir)
    sdir = os.path.dirname(os.path.dirname(args.out)) if os.path.basename(
        os.path.dirname(args.out)) == "raw" else os.path.dirname(args.out)
    os.makedirs(sdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.out))[0]
    base = os.path.join(sdir, stem)
    summ = summarise(recs)
    common.write_csv(base + "_summary.csv", summ)
    common.write_csv(base + "_runs.csv", [{k: v for k, v in r.items() if k != "restart_energies"}
                                          for r in recs])
    plot(summ, base)
    vq = [r for r in recs if r["kind"] == "vqpm" and r["budget"] == max(args.vqpm_budgets)]
    sa = [r for r in recs if r["kind"] == "sa" and r["sweeps"] == 100]
    cmap_cols = ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"]
    for opt in sorted({r["optimizer"] for r in recs if r["kind"] == "qaoa"}):
        series = []
        ps = sorted({r["p"] for r in recs if r["kind"] == "qaoa"})
        for k, p in enumerate(ps):
            for init, ls in (("random", "-"), ("interp", "--")):
                byx = {}
                for r in recs:
                    if r["kind"] == "qaoa" and r["optimizer"] == opt and r["p"] == p and r["init"] == init:
                        byx.setdefault(r["n"], []).append(r)
                if byx:
                    series.append((f"QAOA p={p} {init}", {"ls": ls, "marker": "o", "ms": 3,
                                                          "color": cmap_cols[k % len(cmap_cols)]}, byx))
        for rule, col in (("oracle", "k"), ("max_prob", "0.5")):
            byx = {}
            for r in vq:
                if r["stop_rule"] == rule:
                    byx.setdefault(r["n"], []).append(r)
            if byx:
                series.append((f"VQPM {rule} T={max(args.vqpm_budgets)}",
                               {"marker": "s", "ms": 4, "lw": 2.2, "color": col}, byx))
        byx = {}
        for r in sa:
            byx.setdefault(r["n"], []).append(r)
        if byx:
            series.append(("SA 100 sweeps", {"marker": "^", "ms": 4, "ls": ":", "color": "C3"}, byx))
        common.plot_metric_panels(series, f"{base}_metrics_{opt}", f"QAOA optimizer {opt} vs VQPM and SA")
    # all optimizers in one figure per depth p: how much QAOA depends on the optimizer
    opts = sorted({r["optimizer"] for r in recs if r["kind"] == "qaoa"})
    opt_cols = {o: f"C{k}" for k, o in enumerate(opts)}
    for p in sorted({r["p"] for r in recs if r["kind"] == "qaoa"}):
        series = []
        for opt in opts:
            for init, ls in (("random", "-"), ("interp", "--")):
                byx = {}
                for r in recs:
                    if r["kind"] == "qaoa" and r["optimizer"] == opt and r["p"] == p and r["init"] == init:
                        byx.setdefault(r["n"], []).append(r)
                if byx:
                    series.append((f"QAOA {opt} {init}", {"ls": ls, "marker": "o", "ms": 3,
                                                          "color": opt_cols[opt]}, byx))
        for rule, col in (("oracle", "k"), ("max_prob", "0.5")):
            byx = {}
            for r in vq:
                if r["stop_rule"] == rule:
                    byx.setdefault(r["n"], []).append(r)
            if byx:
                series.append((f"VQPM {rule} T={max(args.vqpm_budgets)}",
                               {"marker": "s", "ms": 4, "lw": 2.2, "color": col}, byx))
        byx = {}
        for r in sa:
            byx.setdefault(r["n"], []).append(r)
        if byx:
            series.append(("SA 100 sweeps", {"marker": "^", "ms": 4, "ls": ":", "color": "C7"}, byx))
        common.plot_metric_panels(series, f"{base}_metrics_all_optimizers_p{p}",
                                  f"QAOA p={p}: all optimizers (solid: random restarts, dashed: INTERP) "
                                  f"vs VQPM and SA")
    print(f"wrote {base}_summary.csv and plots", flush=True)
    import reports

    reports.on_finish(stem)


if __name__ == "__main__":
    main()
