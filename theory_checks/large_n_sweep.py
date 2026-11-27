"""VQPM without rounding at n = 20 ... 100: sweep of the locking threshold p_diff.

Run from the repository root:
    python3 theory_checks/large_n_sweep.py > theory_checks/large_n_sweep.out
Writes theory_checks/large_n_sweep.json (committed gzipped). About an hour on 4 cores.

Same instances and reference strings x_ref as large_n.py (the annealer is called first with the
same random stream, so it returns the same x_ref). Marginals are not rounded. They are exact
(enumeration) at n = 20 and estimated with the control-variate estimator of large_n.py for n >= 30,
with 5e4 draws per round for p_diff >= 1e-3 and 2e5 draws for smaller p_diff.
Round budget: ceil(1.5 n^{3/2}). Stop rule of the code: stop when Pr(x_ref) >= 1/2 after the
filter, or when a lock contradicts x_ref.

Also records, per instance, the round-1 quantities needed to resolve the median |P1 - P0| at
three standard deviations: device shots 9 / (W D^2) (post-selected estimate) and classical draws
9 Var(w) / (W^2 D^2) (control-variate estimate).
"""

import json
import math
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from large_n import (SEED, anneal, cv_marginals, energy, exhaustive, instance, is_local_min,  # noqa: E402
                     setup, wfun)

N_INST = int(os.environ.get("N_INST", 100))
NS = [int(v) for v in os.environ.get("NS", "20,30,40,50,60,70,80,90,100").split(",")]
PDIFFS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]


class Exact:
    """Exact post-selected marginals by enumeration (n <= 20)."""

    def __init__(self, Q):
        T, S, _, _ = setup(Q)
        n = Q.shape[0]
        k = np.arange(1 << n)
        self.B = ((k[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1).astype(np.float32)
        self.w = wfun(energy(T, self.B.astype(float)), S)

    def __call__(self, pi, rng=None, N=None):
        logp = self.B @ np.log(np.maximum(pi, 1e-300)) + (1 - self.B) @ np.log(np.maximum(1 - pi, 1e-300))
        p = np.exp(logp)
        pw = p * self.w
        W = pw.sum()
        return (pw @ self.B) / W, W


def run_sweep(Q, xref, pdiff, max_iter, marg, rng):
    T, S, h, J = setup(Q)
    n = len(h)
    pi = np.full(n, 0.5)
    locked = {}
    wref = float(wfun(energy(T, xref[None]), S)[0])
    N = 50000 if pdiff >= 1e-3 else 200000
    rec = {"rounds": 0, "success": False, "wrong_lock": False, "first_lock": None, "locks_round1": 0}
    last = None
    for j in range(max_iter):
        post, W = marg(pi, rng, N)
        pref = float(np.prod(np.where(xref == 1, pi, 1 - pi)))
        rec["rounds"] = j + 1
        last = post
        if pref * wref / W >= 0.5:
            rec["success"] = True
            break
        if pref == 0:
            rec["wrong_lock"] = True
            break
        new_pi = pi.copy()
        for i in range(n):
            if i in locked:
                continue
            p1 = post[i]
            if (1 - p1) > p1 + pdiff:
                locked[i], new_pi[i] = 0, 0.0
            elif p1 > (1 - p1) + pdiff:
                locked[i], new_pi[i] = 1, 1.0
            else:
                new_pi[i] = p1
            if i in locked:
                if rec["first_lock"] is None:
                    rec["first_lock"] = j + 1
                if j == 0:
                    rec["locks_round1"] += 1
        pi = new_pi
    rec["hit_budget"] = not (rec["success"] or rec["wrong_lock"])
    xout = np.array([locked[i] if i in locked else float(last[i] > 0.5) for i in range(n)])
    rec["E_out"] = float(energy(T, xout[None])[0])
    rec["local_min"] = is_local_min(xout, h, J)
    rec["n_locked"] = len(locked)
    return rec


def work(args):
    n, k = args
    rng = np.random.default_rng([SEED, n, k])     # same stream as large_n.work: same annealer result
    Q = instance(n, k)
    t0 = time.time()
    xa, ea, hits = anneal(Q, rng)
    if n <= 20:
        xref, eref = exhaustive(Q)
    else:
        xref, eref = xa, ea
    T, S, h, J = setup(Q)
    out = {"n": n, "k": k, "E_ref": eref, "x_ref": xref.astype(int).tolist(), "sa_hits": hits}
    # round-1 resolution counts
    N1 = 200000
    X = (rng.random((N1, n)) < 0.5).astype(float)
    w = wfun(energy(T, X), S)
    W, vw = float(w.mean()), float(w.var())
    post, _ = cv_marginals(T, S, np.full(n, 0.5), N1, rng)
    Dmed = float(np.median(np.abs(2 * post - 1)))
    out["round1"] = {"W": W, "std_w": vw ** 0.5, "D_median": Dmed,
                     "device_shots_3sigma": 9 / (W * Dmed**2), "classical_draws_3sigma": 9 * vw / (W**2 * Dmed**2)}
    marg = Exact(Q) if n <= 20 else (lambda pi, r, N: cv_marginals(T, S, pi, N, r))
    budget = math.ceil(1.5 * n ** 1.5)
    out["budget"] = budget
    out["sweep"] = {"%g" % p: run_sweep(Q, xref, p, budget, marg, rng) for p in PDIFFS}
    out["seconds"] = time.time() - t0
    return out


if __name__ == "__main__":
    jobs = [(n, k) for n in NS for k in range(N_INST)]
    jobs.sort(key=lambda t: -t[0])               # largest first for better load balance
    recs = []
    with Pool(int(os.environ.get("PROCS", os.cpu_count() or 1))) as pool:
        for r in pool.imap_unordered(work, jobs, chunksize=1):
            recs.append(r)
            if len(recs) % 50 == 0:
                print("  done %d / %d" % (len(recs), len(jobs)), file=sys.stderr, flush=True)
    recs.sort(key=lambda r: (r["n"], r["k"]))
    path = os.path.join(HERE, "large_n_sweep.json")
    with open(path, "w") as f:
        json.dump(recs, f)
    print("records written to", os.path.relpath(path, os.path.dirname(HERE)), "| p_diff values:", PDIFFS)
