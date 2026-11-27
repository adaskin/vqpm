"""Shared helpers for the experiments.

* one BLAS thread per process (set before NumPy is imported)
* QUBO instances generated once per (n, seed) and stored on disk, so every
  method sees the same instances
* resumable JSON-lines output (one line per finished task)
* a process pool over tasks
* evaluation of one long VQPM trajectory under several stop rules and
  iteration budgets (the dynamics do not depend on the stop rule or on the
  budget, only where they are cut off)
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import vqpm  # noqa: E402

DEFAULT_INSTANCE_DIR = os.path.join(ROOT, "experiments", "instances")


# --------------------------------------------------------------------- instances
def generate_instance(n, seed, i):
    """Random QUBO with the same distribution as vqpm.random_qubo
    (Q = A + A^T, A_ij ~ N(0, 1)), from its own seeded stream so that
    instance i does not depend on how many others were drawn."""
    A = np.random.default_rng([int(seed), int(n), int(i)]).standard_normal((n, n))
    return A + A.T


def instance_path(n, seed, inst_dir=DEFAULT_INSTANCE_DIR):
    return os.path.join(inst_dir, f"qubo_n{n}_seed{seed}.npz")


def get_instances(n, count, seed, inst_dir=DEFAULT_INSTANCE_DIR):
    """Load (or create and save) `count` instances for (n, seed).

    The file stores Q (count, n, n), the optimum index, minimum / maximum
    energy and the gap to the second-lowest energy.
    """
    path = instance_path(n, seed, inst_dir)
    if os.path.exists(path):
        d = np.load(path)
        if d["Q"].shape[0] >= count:
            return {k: (d[k][:count] if d[k].ndim else d[k][()]) for k in d.files}
    os.makedirs(inst_dir, exist_ok=True)
    Qs = np.array([generate_instance(n, seed, i) for i in range(count)])
    opt, emin, emax, gap = [], [], [], []
    for Q in Qs:
        E = vqpm.qubo_energies(n, Q)
        k = int(np.argmin(E))
        opt.append(k)
        emin.append(E[k])
        emax.append(E.max())
        gap.append(np.partition(E, 1)[1] - E[k] if E.size > 1 else np.inf)
    data = dict(
        Q=Qs,
        optimum=np.array(opt),
        e_min=np.array(emin),
        e_max=np.array(emax),
        gap=np.array(gap),
        seed=np.array(seed),
        n=np.array(n),
    )
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **data)
    os.replace(tmp, path)
    return {k: v[:count] if np.ndim(v) else v for k, v in data.items()}


# --------------------------------------------------------------------- output
def run_metadata(args=None):
    def _git(*a):
        try:
            return subprocess.run(
                ["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except Exception:
            return None

    import scipy

    def _ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return None

    return {
        "matplotlib": _ver("matplotlib"),
        "qiskit": _ver("qiskit"),
        "qiskit_aer": _ver("qiskit_aer"),
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "host_cpus": os.cpu_count(),
        "argv": sys.argv,
        "args": vars(args) if args is not None else None,
    }


class JsonlStore:
    """Append-only JSON-lines file; the first line is metadata.

    Each record carries a string 'key'; existing keys are skipped on resume.
    """

    def __init__(self, path, meta=None):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.done = set()
        if os.path.exists(path):
            if meta:
                # record every invocation that resumes this file (provenance)
                with open(path, "a") as f:
                    f.write(json.dumps({"meta": dict(meta, resumed=True)}, default=_json_default) + "\n")
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # partially written last line
                    if "key" in rec:
                        self.done.add(rec["key"])
        else:
            with open(path, "w") as f:
                f.write(json.dumps({"meta": meta or {}}, default=_json_default) + "\n")

    def append(self, rec):
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=_json_default) + "\n")
            f.flush()
        self.done.add(rec["key"])


def load_jsonl(path):
    meta, recs = {}, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "meta" in r and "key" not in r:
                meta = r["meta"]
            else:
                recs.append(r)
    return meta, recs


def write_csv(path, rows, fields=None):
    if not rows:
        return
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def default_workers():
    return max(1, (os.cpu_count() or 2) - 2)


def run_tasks(fn, tasks, store, workers=1, log_every=1):
    """Run fn(task) -> record (or list of records) for tasks whose key is not
    yet in the store; append results as they finish."""
    todo = [t for t in tasks if t["key"] not in store.done]
    print(f"{len(tasks) - len(todo)} tasks already done, {len(todo)} to run "
          f"with {workers} worker(s)", flush=True)
    t0 = time.time()
    done = 0

    def _save(res):
        for r in res if isinstance(res, list) else [res]:
            store.append(r)

    if workers <= 1:
        for t in todo:
            _save(fn(t))
            done += 1
            if done % log_every == 0:
                print(f"[{done}/{len(todo)}] {t['key']} {time.time() - t0:.1f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fn, t): t for t in todo}
            for fut in as_completed(futs):
                t = futs[fut]
                try:
                    _save(fut.result())
                except Exception as e:  # keep going; report the failure
                    print(f"FAILED {t['key']}: {e!r}", flush=True)
                    continue
                done += 1
                if done % log_every == 0:
                    print(f"[{done}/{len(todo)}] {t['key']} {time.time() - t0:.1f}s",
                          flush=True)
    return time.time() - t0


# --------------------------------------------------------------------- VQPM
def bitdiff(a, b):
    return bin(int(a) ^ int(b)).count("1")


def energy_of(n, Q, state):
    x = np.array(vqpm.bits_of(state, n), dtype=float)
    return float(x @ np.triu(Q) @ x)


def locked_state_from_log(lock_log, n, upto_iter):
    """Basis state of all locks with iteration <= upto_iter (None if not all
    qubits are locked by then)."""
    vals = {e["qubit"]: e["value"] for e in lock_log if e["iteration"] <= upto_iter}
    if len(vals) < n:
        return None
    return sum(v << (n - q) for q, v in vals.items())


def derive_result(traj, lock_log, n, optimum, stop_rule, budget):
    """Outcome of a VQPM run with `stop_rule` and `max_iter=budget`, read
    off a trajectory recorded with stop_rule='none' (and max_iter >= budget).

    Valid for exact (shot-free) runs with a fixed pdiff, where the dynamics
    do not depend on the stop rule or on the budget. Returns found_state,
    final target probability, iterations used, stop reason and the locks
    made before stopping.
    """
    L = len(traj["target_prob"])
    if budget > L:
        raise ValueError(f"budget {budget} > recorded iterations {L}")
    tp, mp, am, nl = (traj["target_prob"], traj["max_prob"], traj["argmax"],
                      traj["n_locked"])
    ee = traj.get("expected_energy")
    found = None
    for j in range(budget):
        if stop_rule == "oracle":
            if tp[j] >= 0.5:
                found, prob, it, reason, lock_upto = am[j], tp[j], j + 1, "target_prob", j - 1
                break
            if tp[j] == 0:
                found, prob, it, reason, lock_upto = am[j], tp[j], j + 1, "target_lost", j - 1
                break
        elif stop_rule == "max_prob":
            if mp[j] >= 0.5:
                found, prob, it, reason, lock_upto = am[j], tp[j], j + 1, "max_prob", j - 1
                break
            if nl[j] == n:
                found = locked_state_from_log(lock_log, n, j)
                prob = 1.0 if found == optimum else 0.0
                it, reason, lock_upto = j + 1, "all_locked", j
                break
        else:
            raise ValueError(stop_rule)
    if found is None:
        j = budget - 1
        found, prob, it, reason, lock_upto = am[j], tp[j], budget, "max_iter", j
    locks = [e for e in lock_log if e["iteration"] <= lock_upto]
    wrong = [e for e in locks if e["matches_optimum"] is False]
    expected_energy = None
    if ee is not None:
        # all_locked: the output is the locked basis state (energy filled in
        # by outcome_metrics); otherwise the distribution of the last iteration
        expected_energy = None if reason == "all_locked" else float(ee[it - 1])
    ch = traj.get("changed")
    stall_len = None
    if ch is not None:
        k = lock_upto
        stall_len = 0
        while k >= 0 and not ch[k]:
            stall_len += 1
            k -= 1
    success = int(found) == int(optimum)
    if success:
        failure = None
    elif wrong:
        failure = "wrong_lock"
    elif reason == "max_iter":
        failure = "not_converged"
    else:
        failure = "wrong_stop"
    return {
        "found_state": int(found),
        "success": bool(success),
        "target_prob": float(prob),
        "iterations": int(it),
        "stop_reason": reason,
        "n_locks": len(locks),
        "n_wrong_locks": len(wrong),
        "first_wrong_lock_iter": min((e["iteration"] for e in wrong), default=None),
        "failure": failure,
        "stall_length": stall_len,
        "stalled": bool(stall_len is not None and stall_len >= 10 and reason == "max_iter"),
        "expected_energy": expected_energy,
        "expected_energy_is_state": reason == "all_locked",
    }


def outcome_metrics(n, Q, found, optimum, e_min, e_max, expected_energy=None):
    """Bit and energy metrics of a returned bitstring, in the units of the
    original (unscaled) Q: E_found, E_min, E_max, energy_gap = E_found - E_min,
    norm_energy_gap = (E_found - E_min) / (E_max - E_min) (also stored as
    norm_energy_error), rel_energy_error = gap / |E_min|, Hamming distance to
    the optimum, and, if given, the expected energy of the final output
    distribution with its normalised gap."""
    e = energy_of(n, Q, found)
    span = e_max - e_min
    norm = (e - e_min) / span if span > 0 else 0.0
    out = {
        "energy": e,
        "e_min": float(e_min),
        "e_max": float(e_max),
        "energy_gap": e - e_min,
        "norm_energy_gap": norm,
        "norm_energy_error": norm,
        "rel_energy_error": (e - e_min) / abs(e_min) if e_min != 0 else float("nan"),
        "bit_diff": bitdiff(found, optimum),
        "expected_energy": None,
        "norm_expected_energy_gap": None,
    }
    if expected_energy is not None:
        out["expected_energy"] = float(expected_energy)
        out["norm_expected_energy_gap"] = ((expected_energy - e_min) / span) if span > 0 else 0.0
    return out


def attach_energy_metrics(rows, seed, inst_dir=DEFAULT_INSTANCE_DIR, found_key="found_state",
                          expected_key="expected_energy"):
    """(Re)compute outcome_metrics for saved rows from the stored instances
    (rows need n, instance and the returned state). Rows that carry an
    expected energy keep it."""
    cache = {}
    for r in rows:
        n, i = r["n"], r["instance"]
        if n not in cache:
            path = instance_path(n, seed, inst_dir)
            if not os.path.exists(path):
                cache[n] = None
            else:
                d = np.load(path)
                cache[n] = {k: d[k] for k in ("Q", "optimum", "e_min", "e_max")}
        inst = cache[n]
        if inst is None or r.get(found_key) is None or i >= len(inst["Q"]):
            continue
        exp = r.get(expected_key)
        m = outcome_metrics(n, inst["Q"][i], int(r[found_key]), int(inst["optimum"][i]),
                            float(inst["e_min"][i]), float(inst["e_max"][i]), exp)
        if exp is None and r.get("expected_energy_is_state"):
            m["expected_energy"] = m["energy"]
            m["norm_expected_energy_gap"] = m["norm_energy_gap"]
        r.update(m)
    return rows


# --------------------------------------------------------------------- stats / plots
def mean_ci(vals):
    """Mean and 95% normal-approximation interval of the finite values."""
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return np.nan, np.nan, np.nan
    m = float(v.mean())
    if v.size < 2:
        return m, m, m
    h = 1.96 * float(v.std(ddof=1)) / np.sqrt(v.size)
    return m, m - h, m + h


def wilson_ci(successes):
    """Proportion and 95% Wilson interval of a list of booleans."""
    v = [bool(x) for x in successes if x is not None]
    k, n = sum(v), len(v)
    if n == 0:
        return np.nan, np.nan, np.nan
    z = 1.96
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


METRIC_PANELS = (
    ("success", "Found-optimum rate", "prop"),
    ("bit_diff", "Mean bit difference", "mean"),
    ("norm_energy_gap", "Mean normalised energy gap\n(E_found - E_min)/(E_max - E_min)", "mean"),
    ("norm_expected_energy_gap", "Mean normalised expected-energy gap\n(<E> - E_min)/(E_max - E_min)", "mean"),
)


def metric_summary(rs):
    """Summary columns (with 95% intervals) shared by all experiments."""
    out = {}
    p, lo, hi = wilson_ci([r.get("success") for r in rs])
    out.update(found_optimum_rate=p, found_optimum_lo=lo, found_optimum_hi=hi)
    for key in ("bit_diff", "energy_gap", "norm_energy_gap", "norm_expected_energy_gap"):
        vals = [r.get(key) for r in rs]
        if key == "norm_energy_gap":  # records written before the rename
            vals = [v if v is not None else r.get("norm_energy_error") for v, r in zip(vals, rs)]
        m, lo, hi = mean_ci(vals)
        out.update({f"mean_{key}": m, f"mean_{key}_lo": lo, f"mean_{key}_hi": hi})
    return out


def plot_metric_panels(series, out_prefix, title="", xlabel="n", logx=False, panels=METRIC_PANELS):
    """series: list of (label, plot kwargs, {x: rows}). One panel per metric
    (found-optimum rate with Wilson intervals, mean bit difference, mean
    normalised energy gap and expected-energy gap with 95% intervals);
    panels without data are dropped."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    have = [pnl for pnl in panels if any(r.get(pnl[0]) is not None
                                         for _, _, byx in series for rs in byx.values() for r in rs)]
    if not have:
        return
    fig, axes = plt.subplots(1, len(have), figsize=(4.6 * len(have), 3.9), squeeze=False)
    for label, kw, byx in series:
        xs = sorted(byx)
        for ax, (key, ttl, kind) in zip(axes[0], have):
            st = [wilson_ci([r.get(key) for r in byx[x]]) if kind == "prop"
                  else mean_ci([r.get(key) for r in byx[x]]) for x in xs]
            m = np.array([a for a, _, _ in st])
            if np.all(np.isnan(m)):
                continue
            lo = np.clip(np.nan_to_num(m - np.array([b for _, b, _ in st])), 0, None)
            hi = np.clip(np.nan_to_num(np.array([c for _, _, c in st]) - m), 0, None)
            ax.errorbar(xs, m, yerr=[lo, hi], capsize=2, label=label, **kw)
    for ax, (key, ttl, kind) in zip(axes[0], have):
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
        if logx:
            ax.set_xscale("log")
    h, l = axes[0, 0].get_legend_handles_labels()
    if len(l) > 6:
        fig.legend(h, l, loc="lower center", ncol=min(6, len(l)), fontsize=7)
        fig.tight_layout(rect=(0, 0.05 + 0.03 * (len(l) // 6), 1, 0.95))
    else:
        axes[0, 0].legend(fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    if title:
        fig.suptitle(title, fontsize=11)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_prefix}.{ext}", dpi=150)
    plt.close(fig)
