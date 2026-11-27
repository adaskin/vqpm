"""Experiment 6: noisy Qiskit Aer runs of VQPM (all-qubit measurement).

For each n, instance and simulator seed, the circuit VQPM
(vqpm_qiskit_noisy/vqpm_qiskit.py, measure_mode='all') runs under each
noise specification (e.g. 'fake:FakeTorino', 'depol:0.001,0.01,0.02',
'none'). For every instance the exact state-vector VQPM with the same pdiff,
precision and budget (stop_rule='none', so its trajectory covers the whole
budget) is stored as reference.

Plots (from the saved data):
* <out>_trajectory_*.png: Fig. 11 style per-qubit P(1) of the prepared
  state, exact vs noisy circuit, for a REPRESENTATIVE run chosen by rule:
  the run whose final normalised energy error is the median over all runs
  of the largest n (ties: lowest instance, then seed); not hand-picked.
* <out>_aggregate.png: over all runs, per n and noise model: found-optimum
  rate, mean normalised energy error, mean number of locked qubits vs
  iteration, and mean |P1_circuit - P1_exact| vs iteration.

Example:
    python experiments/noisy_runs.py --n 8 10 12 --instances 5 --sim-seeds 0 1 2 \\
        --noise fake:FakeTorino --shots 100000 --workers 14
"""

import argparse
import collections
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(common.ROOT, "vqpm_qiskit_noisy"))
import vqpm  # noqa: E402


def run_task(task):
    n, i = task["n"], task["i"]
    inst = common.get_instances(n, task["count"], task["seed"], task["inst_dir"])
    Q = inst["Q"][i]
    opt, e_min, e_max = int(inst["optimum"][i]), float(inst["e_min"][i]), float(inst["e_max"][i])
    t0 = time.time()
    if task["kind"] == "exact":
        _, u, _, _ = vqpm.adjust_unitary_phase(n, Q)
        r = vqpm.vqpm_for_qubo(u, n, expected_idx=opt, max_iter=task["max_iter"], pdiff=task["pdiff"],
                               precision=task["precision"], stop_rule="none", return_log=True,
                               record_trajectory="full", fast_forward_fixed_point=True, verbose=False)
        info = r[6]
        # outcome when stopping as the circuit does (all qubits locked or budget)
        found = info["found_state_marginals"]
        rec = {"key": task["key"], "kind": "exact", "n": n, "instance": i,
               "prepared_history": info["trajectory"]["marginals"],
               "n_locked_history": info["trajectory"]["n_locked"],
               "lock_log": info["lock_log"], "found_state": found, "success": found == opt,
               "iterations": next((j + 1 for j, k in enumerate(info["trajectory"]["n_locked"]) if k == n),
                                  task["max_iter"]),
               "seconds": time.time() - t0}
        rec.update(common.outcome_metrics(n, Q, found, opt, e_min, e_max))
        return rec
    import vqpm_qiskit as VQ

    qv = VQ.QuantumCircuitVQPM(n, max_iter=task["max_iter"], pdiff=task["pdiff"],
                               precision=task["precision"], rand_seed=task["sim_seed"],
                               noise=task["noise"], measure_mode="all",
                               locking_rule=task["locking_rule"], verbose=False,
                               threads=task["threads"], method=task.get("method", "density_matrix"))
    r = qv.run_vqpm(Q, shots=task["shots"], optimum=opt)
    rec = {"key": task["key"], "kind": "circuit", "noise": task["noise"], "n": n, "instance": i,
           "sim_seed": task["sim_seed"], "shots": task["shots"], "locking_rule": task["locking_rule"],
           "found_state": r["final_state"], "success": r["final_state"] == opt,
           "iterations": r["iterations"], "shots_used": r["shots_used"],
           "postselected_shots": r["postselected_shots"], "lock_log": r["lock_log"],
           "n_wrong_locks": sum(e["matches_optimum"] is False for e in r["lock_log"]),
           "prepared_history": r["prepared_history"],
           "n_locked_history": [sum(1 for v in h if v in (0.0, 1.0)) for h in r["prepared_history"]],
           "seconds": time.time() - t0}
    rec.update(common.outcome_metrics(n, Q, r["final_state"], opt, e_min, e_max))
    return rec


def _pad(seqs, L):
    out = np.full((len(seqs), L), np.nan)
    for k, s in enumerate(seqs):
        s = np.asarray(s, dtype=float)[:L]
        out[k, : len(s)] = s
        if len(s) and len(s) < L:
            out[k, len(s):] = s[-1]  # finished runs keep their final value
    return out


def summarise(recs):
    g = collections.defaultdict(list)
    for r in recs:
        g[(r.get("noise", "exact"), r["n"])].append(r)
    rows = []
    for (noise, n), rs in sorted(g.items()):
        rows.append({"noise": noise, "n": n, "runs": len(rs), **common.metric_summary(rs),
                     "found_optimum_rate": np.mean([r["success"] for r in rs]),
                     "mean_norm_energy_error": np.mean([r["norm_energy_error"] for r in rs]),
                     "mean_bit_diff": np.mean([r["bit_diff"] for r in rs]),
                     "mean_iterations": np.mean([r["iterations"] for r in rs]),
                     "mean_wrong_locks": np.mean([r.get("n_wrong_locks", np.nan) for r in rs]),
                     "mean_shots_used": np.mean([r.get("shots_used", np.nan) for r in rs]),
                     "mean_seconds": np.mean([r["seconds"] for r in rs])})
    return rows


def plot(recs, out_prefix):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    exact = {(r["n"], r["instance"]): r for r in recs if r["kind"] == "exact"}
    circ = [r for r in recs if r["kind"] == "circuit"]
    if not circ:
        return
    noises = sorted({r["noise"] for r in circ})
    ns = sorted({r["n"] for r in circ})
    # representative run: median final energy error at the largest n (first noise model)
    pool = sorted([r for r in circ if r["n"] == ns[-1] and r["noise"] == noises[0]],
                  key=lambda r: (r["norm_energy_error"], r["instance"], r["sim_seed"]))
    rep = pool[(len(pool) - 1) // 2]
    ex = exact.get((rep["n"], rep["instance"]))
    n = rep["n"]
    cols = (n + 1) // 2
    fig, axes = plt.subplots(2, cols, figsize=(2.6 * cols, 5.4), sharex=True, sharey=True)
    axes = np.ravel(axes)
    L = len(rep["prepared_history"])
    for q in range(n):
        ax = axes[q]
        if ex is not None:
            e = [h[q] for h in ex["prepared_history"]][:L]
            ax.plot(range(len(e)), e, "o-", ms=2.5, label="exact")
        c = [h[q] for h in rep["prepared_history"]]
        ax.plot(range(len(c)), c, "s-", ms=2.5, label=rep["noise"])
        ax.set_title(f"qubit {q + 1}", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
    for k in range(n, len(axes)):
        axes[k].set_visible(False)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"P(q=1) of the prepared state per iteration; n={n}, instance {rep['instance']}, "
                 f"seed {rep['sim_seed']} (median-error run of {len(pool)}); "
                 f"found optimum: {rep['success']}", fontsize=9)
    fig.supxlabel("iteration")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_prefix}_trajectory_representative.{ext}", dpi=150)
    plt.close(fig)

    # aggregate
    summ = summarise(recs)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    labels = ["exact"] + noises
    width = 0.8 / len(labels)
    for k, lab in enumerate(labels):
        s = sorted([r for r in summ if r["noise"] == lab and r["n"] in ns], key=lambda r: r["n"])
        x = np.array([r["n"] for r in s], dtype=float) + (k - (len(labels) - 1) / 2) * width * 2
        axes[0].bar(x, [r["found_optimum_rate"] for r in s], width=width * 2, label=lab)
        axes[1].bar(x, [r["mean_norm_energy_error"] for r in s], width=width * 2, label=lab)
    axes[0].set_title("Found-optimum rate")
    axes[1].set_title("Mean normalised energy error")
    for ax in axes[:2]:
        ax.set_xlabel("n")
        ax.set_xticks(ns)
    axes[0].legend(fontsize=8)
    for n_ in ns:
        for noise in noises:
            rs = [r for r in circ if r["n"] == n_ and r["noise"] == noise]
            if not rs:
                continue
            Lmax = max(len(r["n_locked_history"]) for r in rs)
            lk = _pad([r["n_locked_history"] for r in rs], Lmax)
            axes[2].plot(np.nanmean(lk, axis=0), label=f"n={n_} {noise}")
            diffs = []
            for r in rs:
                e = exact.get((r["n"], r["instance"]))
                if e is None:
                    continue
                A = _pad(r["prepared_history"], Lmax)
                B = _pad(e["prepared_history"], Lmax) if len(e["prepared_history"]) else None
                if B is None:
                    continue
                m = min(len(r["prepared_history"]), len(e["prepared_history"]))
                d = np.abs(np.asarray(r["prepared_history"][:m]) - np.asarray(e["prepared_history"][:m])).mean(axis=1)
                diffs.append(d)
            if diffs:
                D = _pad(diffs, Lmax)
                axes[3].plot(np.nanmean(D, axis=0), label=f"n={n_} {noise}")
        ex_rs = [e for (nn, _), e in exact.items() if nn == n_]
        if ex_rs:
            Lmax = max(len(e["n_locked_history"]) for e in ex_rs)
            axes[2].plot(np.nanmean(_pad([e["n_locked_history"] for e in ex_rs], Lmax), axis=0), "k--",
                         lw=1, label=f"n={n_} exact")
    axes[2].set_title("Mean locked qubits")
    axes[3].set_title("Mean |P1 circuit - P1 exact| over qubits")
    for ax in axes[2:]:
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_prefix}_aggregate.{ext}", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, nargs="+", default=[8, 10, 12])
    ap.add_argument("--instances", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--sim-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--noise", nargs="+", default=["fake:FakeTorino"])
    ap.add_argument("--shots", type=int, default=100000)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--pdiff", type=float, default=0.1)
    ap.add_argument("--precision", type=int, default=4)
    ap.add_argument("--locking-rule", default="pdiff", choices=["pdiff", "robust", "hoeffding_union"])
    ap.add_argument("--threads", type=int, default=1, help="Aer threads per worker")
    ap.add_argument("--method", default="density_matrix",
                    help="Aer method; density_matrix makes the shot count almost free")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--inst-dir", default=common.DEFAULT_INSTANCE_DIR)
    ap.add_argument("--out", default="experiments/results/noisy/raw/noisy.jsonl")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    if not args.plot_only:
        tasks = []
        tag = f"T{args.max_iter}_pd{args.pdiff}_pr{args.precision}"
        for n in sorted(args.n, reverse=True):
            common.get_instances(n, args.instances, args.seed, args.inst_dir)
            for i in range(args.instances):
                kw = dict(n=n, i=i, count=args.instances, seed=args.seed, inst_dir=args.inst_dir,
                          max_iter=args.max_iter, pdiff=args.pdiff, precision=args.precision)
                for noise in args.noise:
                    for s in args.sim_seeds:
                        tasks.append(dict(kw, kind="circuit", noise=noise, sim_seed=s, shots=args.shots,
                                          locking_rule=args.locking_rule, threads=args.threads,
                                          method=args.method,
                                          key=f"circ_{noise}_n{n}_i{i}_s{s}_sh{args.shots}_{args.locking_rule}_{tag}"))
                tasks.append(dict(kw, kind="exact", key=f"exact_n{n}_i{i}_{tag}"))
        store = common.JsonlStore(args.out, meta=common.run_metadata(args))
        secs = common.run_tasks(run_task, tasks, store, workers=args.workers, log_every=1)
        print(f"done in {secs:.1f}s", flush=True)

    _, recs = common.load_jsonl(args.out)
    common.attach_energy_metrics(recs, args.seed, args.inst_dir)
    sdir = os.path.dirname(os.path.dirname(args.out)) if os.path.basename(
        os.path.dirname(args.out)) == "raw" else os.path.dirname(args.out)
    os.makedirs(sdir, exist_ok=True)
    base = os.path.join(sdir, os.path.splitext(os.path.basename(args.out))[0])
    common.write_csv(base + "_summary.csv", summarise(recs))
    common.write_csv(base + "_runs.csv", [{k: v for k, v in r.items()
                                           if k not in ("prepared_history", "lock_log", "n_locked_history")}
                                          for r in recs])
    plot(recs, base)
    series = []
    for k, lab in enumerate(sorted({r.get("noise", "exact") for r in recs})):
        byx = {}
        for r in recs:
            if r.get("noise", "exact") == lab:
                byx.setdefault(r["n"], []).append(r)
        series.append((lab, {"marker": "o", "ms": 4}, byx))
    common.plot_metric_panels(series, f"{base}_metrics", "Noisy circuit VQPM vs exact")
    print(f"wrote {base}_summary.csv and plots", flush=True)
    import reports

    reports.on_finish(os.path.splitext(os.path.basename(args.out))[0])


if __name__ == "__main__":
    main()
