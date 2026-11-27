"""VQPM on larger QUBO instances (n = 20 ... 100) by classical sampling.

Run from the repository root:
    python3 theory_checks/large_n.py > theory_checks/large_n.out
Writes theory_checks/large_n.json (per-instance records; committed gzipped as large_n.json.gz) and, with matplotlib,
paper/figures/app_large_n.pdf via large_n_report.py. Runtime about 40 minutes on 4 cores.

No 2^n state vector is used. For a product input with marginals pi, the post-selected
marginals pi_i' = E_p[x_i w(x)] / E_p[w(x)], w = cos^2(lambda/2), are estimated from N draws
x ~ p with the control variate E_p[x_i] = pi_i (known exactly):
    pi_i' = pi_i + mean((x_i - pi_i)(w - mean w)) / mean w .
Checked against exact enumeration at n = 20 (max error printed below).

Instances: vqpm.random_qubo with seed 2026 + n (as in theory_checks.py), phase map of the code
(lambda = pi/4 + (pi/4) E / S). Reference solution x_ref: exhaustive search at n = 20, best of
simulated annealing (32 chains) plus 1-flip descent for n >= 30.

Three versions of the VQPM loop (p_diff = 0.01, at most 30 rounds, oracle stop as in the code:
stop when Pr(x_ref) >= 1/2 after the filter, or when a lock contradicts x_ref):
  paper     : marginals rounded to 3 decimals before the lock test and re-preparation (as in the
              main-text simulations; P1 is renormalised after rounding P0 and P1 separately),
              estimated with N_CV draws (error far below the rounding unit).
  norounding: same, without rounding.
  norounding_long: as norounding with at most 300 rounds.
  device    : M post-selected-shot experiment, ratio estimator over accepted shots, then rounding
              to 3 decimals (what a device with M shots per round would do).
"""

import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from vqpm import random_qubo  # noqa: E402

SEED = 2026
N_INST = int(os.environ.get("N_INST", 100))
NS = [int(v) for v in os.environ.get("NS", "20,30,40,50,60,70,80,90,100").split(",")]
N_CV = 20000          # draws per round, control-variate estimator
M_DEV = 200000        # shots per round for the device version (about the count of the derived Hoeffding threshold)
PDIFF, PREC, MAX_ITER = 0.01, 3, 30
MAX_LONG = 300       # longer budget for the version without rounding
CHUNK = 20000


def instance(n, k):
    np.random.seed(SEED + n)
    for _ in range(k + 1):
        Q = random_qubo(n)
    return Q


def setup(Q):
    T = np.triu(Q)
    S = np.abs(T).sum()
    h = np.diag(T).copy()
    J = T + T.T
    np.fill_diagonal(J, 0.0)
    return T, S, h, J


def energy(T, X):
    return np.einsum("ij,ij->i", X @ T, X)


def wfun(E, S):
    return np.cos((np.pi / 4 + np.pi / 4 * E / S) / 2) ** 2


def cv_marginals(T, S, pi, N, rng):
    n = len(pi)
    acc_w = 0.0
    acc_c = np.zeros(n)
    acc_x = np.zeros(n)
    acc_xw = np.zeros(n)
    for s in range(0, N, CHUNK):
        m = min(CHUNK, N - s)
        X = (rng.random((m, n)) < pi).astype(float)
        w = wfun(energy(T, X), S)
        acc_w += w.sum()
        acc_x += X.sum(axis=0)
        acc_xw += w @ X
    wb = acc_w / N
    # mean((x - pi)(w - wb)) = mean(x w) - wb mean(x) - pi mean(w) + pi wb = mean(xw) - wb mean(x)
    cov = acc_xw / N - wb * acc_x / N
    return pi + cov / wb, wb


def device_marginals(T, S, pi, M, rng):
    n = len(pi)
    acc = np.zeros(n)
    na = 0
    for s in range(0, M, CHUNK):
        m = min(CHUNK, M - s)
        X = (rng.random((m, n)) < pi).astype(float)
        c = rng.random(m) < wfun(energy(T, X), S)
        acc += X[c].sum(axis=0)
        na += int(c.sum())
    return acc / max(na, 1), na / M


def anneal(Q, rng, chains=32, sweeps=3000):
    T, S, h, J = setup(Q)
    n = len(h)
    x = rng.integers(0, 2, size=(chains, n)).astype(float)
    field = x @ J
    e = energy(T, x)
    best_e, best_x = e.copy(), x.copy()
    T0 = float(np.mean(np.abs(h) + np.abs(J).sum(axis=1))) / 2
    for temp in T0 * (1e-3) ** (np.arange(sweeps) / (sweeps - 1)):
        for k in rng.permutation(n):
            dE = (1 - 2 * x[:, k]) * (h[k] + field[:, k])
            a = (dE <= 0) | (rng.random(chains) < np.exp(-np.maximum(dE, 0) / temp))
            if a.any():
                sg = np.where(a, 1 - 2 * x[:, k], 0.0)
                x[:, k] += sg
                field += sg[:, None] * J[k][None, :]
                e += np.where(a, dE, 0.0)
                b = e < best_e - 1e-12
                best_e[b], best_x[b] = e[b], x[b]
    # 1-flip steepest descent from every chain's best
    out = []
    for xb in best_x:
        xb = descend(xb, h, J)
        out.append((float(energy(T, xb[None])[0]), xb))
    out.sort(key=lambda t: t[0])
    hits = sum(abs(o[0] - out[0][0]) < 1e-9 for o in out)
    return out[0][1], out[0][0], hits


def descend(x, h, J):
    x = x.copy()
    while True:
        d = (1 - 2 * x) * (h + J @ x)
        k = int(np.argmin(d))
        if d[k] >= -1e-12:
            return x
        x[k] = 1 - x[k]


def is_local_min(x, h, J):
    return bool(np.all((1 - 2 * x) * (h + J @ x) > 0))


def exhaustive(Q):
    n = Q.shape[0]
    T = np.triu(Q)
    best_e, best_i = np.inf, -1
    step = 1 << 16
    for s in range(0, 1 << n, step):
        k = np.arange(s, min(s + step, 1 << n))
        X = ((k[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1).astype(float)
        E = energy(T, X)
        j = int(np.argmin(E))
        if E[j] < best_e:
            best_e, best_i = E[j], k[j]
    x = ((best_i >> (n - 1 - np.arange(n))) & 1).astype(float)
    return x, float(best_e)


def run(Q, xref, mode, rng, max_iter=MAX_ITER):
    T, S, h, J = setup(Q)
    n = len(h)
    pi = np.full(n, 0.5)
    locked = {}
    wref = float(wfun(energy(T, xref[None]), S)[0])
    log = {"rounds": 0, "success": False, "wrong_lock": False, "stalled": False, "locks": [],
           "W": [], "maxD": []}
    last = None
    for j in range(max_iter):
        if mode == "device":
            post, W = device_marginals(T, S, pi, M_DEV, rng)
        else:
            post, W = cv_marginals(T, S, pi, N_CV, rng)
        log["W"].append(W)
        free = [i for i in range(n) if i not in locked]
        log["maxD"].append(float(np.max(np.abs(2 * post[free] - 1))) if free else 0.0)
        # oracle stop on x_ref, as in vqpm_for_qubo(expected_idx=...)
        pref = np.prod(np.where(xref == 1, pi, 1 - pi))
        qref = pref * wref / W
        log["rounds"] = j + 1
        last = post
        if qref >= 0.5:
            log["success"] = True
            break
        if pref == 0:
            log["wrong_lock"] = True
            break
        new_pi = pi.copy()
        for i in free:
            p1 = post[i]
            if mode in ("paper", "device"):
                r0, r1 = np.round(1 - p1, PREC), np.round(p1, PREC)
            else:
                r0, r1 = 1 - p1, p1
            if r0 > r1 + PDIFF:
                locked[i] = 0
                new_pi[i] = 0.0
                log["locks"].append((j, i, 0, int(xref[i] == 0)))
            elif r1 > r0 + PDIFF:
                locked[i] = 1
                new_pi[i] = 1.0
                log["locks"].append((j, i, 1, int(xref[i] == 1)))
            else:
                new_pi[i] = r1 / (r0 + r1)
        if mode == "paper" and np.array_equal(new_pi, pi):
            log["stalled"] = True       # deterministic from here on: the code would repeat this round
            break
        pi = new_pi
    xout = np.array([locked[i] if i in locked else float(last[i] > 0.5) for i in range(n)])
    log["E_out"] = float(energy(T, xout[None])[0])
    log["local_min"] = is_local_min(xout, h, J)
    log["n_locked"] = len(locked)
    log["W"] = [float(v) for v in log["W"]]
    return log


def round1(Q, xref, rng):
    """P1 - P0 after the first round from the uniform start: sampled and first-order formula."""
    T, S, h, J = setup(Q)
    n = len(h)
    post, W = cv_marginals(T, S, np.full(n, 0.5), 200000, rng)
    D = 2 * post - 1
    g = h + 0.5 * J.sum(axis=1)                      # d E_bar / d pi_i at pi = 1/2
    Ebar = 0.25 * np.trace(T) * 2 + 0.25 * (T.sum() - np.trace(T))   # E over uniform x
    lam = np.pi / 4 + np.pi / 4 * Ebar / S
    D1 = -np.sin(lam) * np.pi * g / (16 * S * np.cos(lam / 2) ** 2)
    sgn_ref = 2 * xref - 1
    top = int(np.argmax(np.abs(D)))
    return {"D": D.tolist(), "D_formula": D1.tolist(), "agree": float(np.mean(np.sign(D) == sgn_ref)),
            "agree_top": bool(np.sign(D[top]) == sgn_ref[top]),
            "agree_formula": float(np.mean(np.sign(D1) == sgn_ref)), "W": W}


def work(args):
    n, k = args
    rng = np.random.default_rng([SEED, n, k])
    Q = instance(n, k)
    t0 = time.time()
    xa, ea, hits = anneal(Q, rng)
    rec = {"n": n, "k": k, "E_sa": ea, "sa_hits": hits}
    if n <= 20:
        xe, ee = exhaustive(Q)
        rec["E_exact"] = ee
        rec["sa_is_exact"] = bool(abs(ea - ee) < 1e-9)
        xref, eref = xe, ee
    else:
        xref, eref = xa, ea
    rec["E_ref"] = eref
    T, S, h, J = setup(Q)
    rec["E_scale"] = float(S)
    rec["round1"] = round1(Q, xref, rng)
    for mode in ("paper", "norounding", "device"):
        rec[mode] = run(Q, xref, mode, rng)
    rec["norounding_long"] = run(Q, xref, "norounding", rng, max_iter=MAX_LONG)
    rec["seconds"] = time.time() - t0
    return rec


def check_cv():
    """Control-variate marginals versus exact enumeration at n = 20."""
    n = 20
    Q = instance(n, 0)
    T, S, h, J = setup(Q)
    k = np.arange(1 << n)
    B = ((k[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1).astype(float)
    w = wfun(energy(T, B), S)
    rng = np.random.default_rng(0)
    errs = []
    for pi in [np.full(n, 0.5), rng.uniform(0.1, 0.9, n)]:
        p = np.prod(np.where(B == 1, pi, 1 - pi), axis=1)
        exact = (p * w) @ B / (p @ w)
        est, _ = cv_marginals(T, S, pi, N_CV, rng)
        errs.append(np.max(np.abs(est - exact)))
    return errs


if __name__ == "__main__":
    print("control-variate estimator, N = %d draws, max error vs exact at n = 20: %s"
          % (N_CV, ", ".join("%.1e" % e for e in check_cv())))
    jobs = [(n, k) for n in NS for k in range(N_INST)]
    out_json = os.path.join(os.path.dirname(__file__), "large_n.json")
    recs = []
    with Pool(int(os.environ.get("PROCS", os.cpu_count() or 1))) as pool:
        for r in pool.imap_unordered(work, jobs, chunksize=1):
            recs.append(r)
            if len(recs) % 50 == 0:
                print("  done %d / %d" % (len(recs), len(jobs)), file=sys.stderr, flush=True)
    recs.sort(key=lambda r: (r["n"], r["k"]))
    with open(out_json, "w") as f:
        json.dump(recs, f)
    print("records written to", os.path.relpath(out_json, ROOT))
