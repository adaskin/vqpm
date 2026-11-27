"""The VQPM update without and with locking on the instances of the QAOA comparison, n = 4 ... 22.

Run from the repository root:
    python3 theory_checks/small_n_flow.py > theory_checks/small_n_flow.out
Writes theory_checks/small_n_flow.json.

Instances: experiments/common.generate_instance(n, 2026, i), i = 0..99 (the instances of the QAOA comparison and of the
iteration-budget grid), copied here so that the script needs only numpy. x_ref is the exact optimum (enumeration).

For every instance and p_diff in PDIFFS (None = no locking), from the uniform start:
  first-order update, meanfield.mf_run (Eq. mf), for every n;
  exact update, Eq. (vqpm-update) by enumeration (meanfield.exact_step_factory), for n <= EXACT_MAX.
Exact marginals, no rounding, no oracle stop: each run continues until all qubits are locked or the free marginals
settle at a vertex that is stable for them (as in large_n_flow.py). Recorded: t_half, the first iteration in which
x_ref has probability >= 1/2 after the filter; success = t_half exists and no lock contradicts x_ref before it
(the test of large_n_flow.py); found = the returned string is x_ref; rounds = iterations to the end.
"""

import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from meanfield import exact_step_factory, mf_run  # noqa: E402

NS = list(range(4, 23))
K = 100
SEED = 2026
PDIFFS = [None, 0.2, 0.01]
EXACT_MAX = 14
TOL = 1e-3


def generate_instance(n, seed, i):
    """= experiments/common.generate_instance."""
    A = np.random.default_rng([int(seed), int(n), int(i)]).standard_normal((n, n))
    return A + A.T


def optimum(Q):
    n = Q.shape[0]
    T = np.triu(Q)
    k = np.arange(1 << n)
    B = ((k[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1).astype(float)
    E = np.einsum("ij,ij->i", B @ T, B)
    j = int(np.argmin(E))
    return B[j], float(E[j])


def one(args):
    n, i = args
    Q = generate_instance(n, SEED, i)
    xref, eref = optimum(Q)
    out = {"n": n, "i": i, "E_ref": eref, "first_order": {}, "exact": {}}
    kinds = [("first_order", None)]
    if n <= EXACT_MAX:
        kinds.append(("exact", exact_step_factory(Q)))
    for kind, step in kinds:
        for p in PDIFFS:
            r = mf_run(Q, pdiff=p, xref=xref, tol=TOL, step=step, max_iter=10**6)
            wrong = [t for (t, j, v) in r["locks"] if v != xref[j]]
            th = r["t_half"]
            out[kind]["none" if p is None else "%g" % p] = {
                "rounds": r["rounds"], "t_half": th, "n_locks": len(r["locks"]),
                "success": th is not None and (not wrong or wrong[0] > th),
                "found": bool(np.array_equal(r["x"], xref)), "converged": r["converged"]}
    return out


if __name__ == "__main__":
    t0 = time.time()
    jobs = [(n, i) for n in NS for i in range(K)]
    with Pool(int(os.environ.get("PROCS", os.cpu_count() or 1))) as pool:
        R = pool.map(one, jobs, chunksize=4)
    json.dump(R, open(os.path.join(HERE, "small_n_flow.json"), "w"))
    keys = ["none", "0.2", "0.01"]
    print("VQPM update on the seed-2026 instances (100 per n), exact marginals, no rounding, no oracle stop")
    print("success = x_ref reaches probability 1/2 before a contradicting lock; found = returned string is x_ref")
    for kind in ["first_order", "exact"]:
        print("\n%s update" % kind.replace("_", "-"))
        print("   n | success none  0.2  0.01 | found none  0.2  0.01 | median t_half (passing) none / 0.2 / 0.01"
              " | median rounds to end none / 0.2 / 0.01")
        for n in NS:
            rs = [r for r in R if r["n"] == n and r[kind]]
            if not rs:
                continue
            s = [np.mean([r[kind][k]["success"] for r in rs]) for k in keys]
            f = [np.mean([r[kind][k]["found"] for r in rs]) for k in keys]
            th = [np.median([r[kind][k]["t_half"] for r in rs if r[kind][k]["success"]] or [np.nan]) for k in keys]
            rd = [np.median([r[kind][k]["rounds"] for r in rs]) for k in keys]
            print("%4d |      %s | %s | %s | %s" % (n, " ".join("%4.2f" % v for v in s), " ".join("%4.2f" % v for v in f),
                                                 " / ".join("%6.0f" % v for v in th), " / ".join("%6.0f" % v for v in rd)))
    print("\nexact vs first-order, same returned string (n <= %d):" % EXACT_MAX)
    for k in keys:
        same = [r["exact"][k]["found"] == r["first_order"][k]["found"] for r in R if r["exact"]]
        print("  %-5s found agrees in %.1f%% of %d runs" % (k, 100 * np.mean(same), len(same)))
    print("\ntime %.0f s" % (time.time() - t0), file=sys.stderr)
