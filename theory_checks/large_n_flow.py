"""The product-state map without and with locking, n = 20 ... 100.

Run from the repository root:
    python3 theory_checks/large_n_flow.py > theory_checks/large_n_flow.out
Writes theory_checks/large_n_flow.json (committed gzipped). CHECK=1 instead compares the batched runner
with meanfield.mf_run on a few instances and exits.

Uses the first-order (mean-field) form of the VQPM map from meanfield.py, which needs no samples and costs O(n^2)
per round, on the instances and reference strings x_ref of large_n_sweep.py (read from large_n_sweep.json[.gz]).
All runs of one n (100 instances x 10 thresholds) are iterated together; each run follows meanfield.mf_run.

For every instance: the unlocked map from the uniform start until every marginal is within 1e-3 of 0 or 1 at a
vertex that is stable (its end point x_inf), and the map with the code's locking rule for each p_diff in PDIFFS,
without rounding. No oracle stop: each run continues until all qubits are locked or the free marginals have
converged to a vertex that is stable for them (every free single flip raises the energy); t_half records the
first round in which x_ref has probability >= 1/2 after the filter (the code's success test).

Signs that locking reads. On the unlocked run, for every qubit and READ_PS threshold p, the round at which
|2 pi_i - 1| first exceeds p and the sign there, compared with the centre gradient (-sign g_i(1/2), the sign of the
round-1 D_i), with the linear prediction near the centre,
    pi = 1/2 + delta,  delta(t) = -(I - A^t) J^{-1} g0,  A = I - (eta0/4) J,
at the same round, with x_inf and with x_ref. tau = 4 / (eta0 |mu_min(J)|) is the time scale on which the
components along the eigenvectors of J with negative eigenvalues take over from the gradient.
"""

import gzip
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
from large_n import instance  # noqa: E402
from meanfield import _bits, mf_run, mf_setup  # noqa: E402

PDIFFS = [None, 0.2, 0.15, 0.1, 0.05, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
READ_PS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.15, 0.2]
GRID_C = np.logspace(-2, 2.5, 28)          # overlap logged at rounds ceil(c S)
TOL = 1e-3
MAX_ITER = 3 * 10**6
LOG_FOR = {None, 0.1, 1e-2, 1e-4}


def load_refs():
    p = os.path.join(HERE, "large_n_sweep.json")
    R = json.load(open(p)) if os.path.exists(p) else json.load(gzip.open(p + ".gz", "rt"))
    return {(r["n"], r["k"]): (np.array(r["x_ref"], float), r["E_ref"]) for r in R}


REFS = load_refs()


def batch_run(n, ks, pdiffs=PDIFFS, max_iter=MAX_ITER, tol=TOL, log_for=LOG_FOR, read_ps=READ_PS):
    """Iterate mf_run's loop for every (instance, p_diff) pair at once. Returns per-run results and, for the
    unlocked column (pdiffs[0] must be None), the first round and sign at which each qubit passes each read_ps."""
    K, P = len(ks), len(pdiffs)
    st = [mf_setup(instance(n, k)) for k in ks]
    T = np.stack([s[0] for s in st])
    S = np.array([s[1] for s in st])
    h = np.stack([s[2] for s in st])[:, :, None]
    J = np.stack([s[3] for s in st])
    xref = np.stack([REFS[(n, k)][0] for k in ks])
    sref = (2 * xref - 1)[:, :, None]
    lam_ref = np.pi / 4 + np.pi / 4 * np.einsum("ki,kij,kj->k", xref, T, xref) / S
    wref = np.cos(lam_ref / 2) ** 2
    log_at = [sorted({int(math.ceil(c * s)) for c in GRID_C}) for s in S]
    logcol = np.array([p in log_for for p in pdiffs])
    pd = np.array([np.inf if p is None else p for p in pdiffs])

    # results, indexed by the original (instance, p_diff)
    rounds = np.zeros((K, P), int)
    t_half = np.full((K, P), -1)
    xend = np.zeros((K, P, n))
    conv = np.zeros((K, P), bool)
    locks = [[[] for _ in range(P)] for _ in range(K)]
    logs = [[[] for _ in range(P)] for _ in range(K)]
    rfirst = np.full((len(read_ps), K, n), -1)
    rsign = np.zeros((len(read_ps), K, n))
    reads = list(range(len(read_ps)))
    logmap = {}
    for a_, la in enumerate(log_at):
        for t_ in la:
            logmap.setdefault(t_, []).append(a_)

    # working arrays for the runs still going: rows = instances ka, columns = thresholds pa
    ka, pa = np.arange(K), np.arange(P)
    Jw, hw, Sw, srw, wrw, pdw = J, h, S, sref, wref, pd
    piw = np.full((K, n, P), 0.5)
    frw = np.ones((K, n, P), bool)
    dnw = np.zeros((K, P), bool)
    drw = np.zeros((K, P), int)          # round at which the run finished
    thw = np.full((K, P), -1)

    def finish(mask, t):
        for a, b in zip(*np.nonzero(mask)):
            k, p = ka[a], pa[b]
            x = piw[a, :, b]
            rounds[k, p], xend[k, p], conv[k, p] = t, (x > 0.5), bool(np.all(np.minimum(x, 1 - x) < tol))
            if logcol[p]:
                logs[k][p].append((t, float(np.mean(srw[a, :, 0] * (2 * x - 1))), int((~frw[a, :, b]).sum()), _bits(x)))

    t = 0
    while True:
        t += 1
        Jpi = Jw @ piw
        Eb = (hw * piw).sum(1) + 0.5 * (piw * Jpi).sum(1)
        lam = np.pi / 4 + np.pi / 4 * Eb / Sw[:, None]
        c2 = np.cos(lam / 2) ** 2
        if t > 1:   # the code's success test for round t - 1 (needs Ebar of the state after that round)
            need = (thw < 0) & (~dnw | (drw == t - 1))
            if need.any():
                with np.errstate(divide="ignore"):
                    lp = np.log(np.where(srw > 0, piw, 1 - piw)).sum(1)
                hit = need & (np.exp(lp) * wrw[:, None] / c2 >= 0.5)
                for a, b in zip(*np.nonzero(hit)):
                    t_half[ka[a], pa[b]] = t - 1
                thw[hit] = t - 1
        if dnw.all() or t > max_iter:
            break
        if t % 256 == 0:   # drop finished instances and thresholds
            kr, kc = (~dnw).any(1), (~dnw).any(0)
            if not kr.all() or not kc.all():
                ka, pa = ka[kr], pa[kc]
                Jw, hw, Sw, srw, wrw, pdw = Jw[kr], hw[kr], Sw[kr], srw[kr], wrw[kr], pdw[kc]
                piw, frw = piw[kr][:, :, kc], frw[kr][:, :, kc]
                dnw, drw, thw = dnw[kr][:, kc], drw[kr][:, kc], thw[kr][:, kc]
                Jpi, Eb, lam, c2 = Jpi[kr][:, :, kc], Eb[kr][:, kc], lam[kr][:, kc], c2[kr][:, kc]
        eta = np.pi * np.sin(lam) / (8 * Sw[:, None] * c2)
        new = piw - eta[:, None, :] * piw * (1 - piw) * (hw + Jpi)
        act = frw & ~dnw[:, None, :]
        new = np.where(act, new, piw)
        d = 2 * new - 1
        hit = act & (np.abs(d) > pdw[None, None, :])
        if hit.any():
            for a, i, b in zip(*np.nonzero(hit)):
                locks[ka[a]][pa[b]].append((t, int(i), int(d[a, i, b] > 0)))
            new[hit] = (d[hit] > 0)
            frw &= ~hit
        piw = new
        if reads and pa[0] == 0:
            du = np.abs(2 * piw[:, :, 0] - 1)
            for j in list(reads):
                F = rfirst[j][ka]
                m = (F < 0) & (du > read_ps[j])
                if m.any():
                    F[m] = t
                    G = rsign[j][ka]
                    G[m] = np.sign(2 * piw[:, :, 0] - 1)[m]
                    rfirst[j][ka], rsign[j][ka] = F, G
                    if not (rfirst[j] < 0).any():
                        reads.remove(j)
        notdone = ~dnw
        if t in logmap:
            for k in logmap[t]:
                a = np.searchsorted(ka, k)
                if a == len(ka) or ka[a] != k:
                    continue
                for b in np.flatnonzero(logcol[pa] & notdone[a]):
                    x = piw[a, :, b]
                    logs[k][pa[b]].append((t, float(np.mean(srw[a, :, 0] * (2 * x - 1))),
                                           int((~frw[a, :, b]).sum()), _bits(x)))
        allfixed = ~frw.any(1)
        near = np.all((np.minimum(piw, 1 - piw) < tol) | ~frw, axis=1)
        newdone = notdone & allfixed
        cand = notdone & ~allfixed & near
        if cand.any():
            xv = (piw > 0.5).astype(float)
            if cand.sum() <= 16:
                for a, b in zip(*np.nonzero(cand)):
                    gx = hw[a, :, 0] + Jw[a] @ xv[a, :, b]
                    if np.all(((1 - 2 * xv[a, :, b]) * gx)[frw[a, :, b]] > 0):
                        newdone[a, b] = True
            else:
                ok = np.all(((1 - 2 * xv) * (hw + Jw @ xv) > 0) | ~frw, axis=1)
                newdone |= cand & ok
        if newdone.any():
            dnw |= newdone
            drw[newdone] = t
            finish(newdone, t)
    if not dnw.all():
        finish(~dnw, max_iter)
    return dict(ks=ks, T=T, S=S, h=h[:, :, 0], J=J, xref=xref, log_at=log_at, rounds=rounds, t_half=t_half,
                xend=xend, conv=conv, locks=locks, logs=logs, rfirst=rfirst, rsign=rsign)


def is_local_min(x, h, J):
    return bool(np.all((1 - 2 * x) * (h + J @ x) > 0))


def linear_sign(t, i, V, mu, c, eta0):
    """sign of delta_i(t) = -sum_k (1 - a_k^t)/mu_k c_k V_ik, a_k = 1 - eta0 mu_k / 4."""
    z = t * np.log1p(-eta0 * mu / 4)
    f = -np.expm1(np.minimum(z, 700.0)) / mu
    return float(np.sign(-(V[i] * f) @ c))


def work_n(n):
    t0 = time.time()
    ks = list(range(int(os.environ.get("N_INST", 100))))
    B = batch_run(n, ks)
    out = []
    for a, k in enumerate(ks):
        T, S, h, J, xref = B["T"][a], float(B["S"][a]), B["h"][a], B["J"][a], B["xref"][a]
        eref = REFS[(n, k)][1]
        rec = {"n": n, "k": k, "S": S, "E_ref": eref, "log_at": B["log_at"][a], "runs": {}}
        xinf = B["xend"][a, 0]
        for b, p in enumerate(PDIFFS):
            x = B["xend"][a, b]
            lk = B["locks"][a][b]
            wrong = [t for (t, i, v) in lk if v != xref[i]]
            th = int(B["t_half"][a, b]) if B["t_half"][a, b] > 0 else None
            r = {"rounds": int(B["rounds"][a, b]), "t_half": th, "converged": bool(B["conv"][a, b]),
                 "bits": _bits(x), "is_ref": bool(np.array_equal(x, xref)), "is_inf": bool(np.array_equal(x, xinf)),
                 "ham_ref": int(np.sum(x != xref)), "ham_inf": int(np.sum(x != xinf)),
                 "gap": (float(x @ T @ x) - eref) / abs(eref), "local_min": is_local_min(x, h, J),
                 "n_locks": len(lk), "free_at_end": n - len(lk),
                 "first_lock": lk[0][0] if lk else None, "last_lock": lk[-1][0] if lk else None,
                 "first_wrong_lock": wrong[0] if wrong else None,
                 "locks_agree_ref": float(np.mean([v == xref[i] for (t, i, v) in lk])) if lk else None,
                 "locks_agree_inf": float(np.mean([v == xinf[i] for (t, i, v) in lk])) if lk else None,
                 "locks_round1": sum(1 for (t, i, v) in lk if t == 1)}
            # the code's success test: x_ref reaches probability 1/2 before any lock contradicts it
            r["success"] = th is not None and (not wrong or wrong[0] > th)
            if p in LOG_FOR:
                r["log"] = B["logs"][a][b]
            rec["runs"]["none" if p is None else "%g" % p] = r
        # signs read on the unlocked run
        half = np.full(n, 0.5)
        g0 = h + J @ half
        mu, V = np.linalg.eigh(J)
        lam0 = np.pi / 4 + np.pi / 4 * float(h @ half + 0.5 * half @ J @ half) / S
        eta0 = np.pi * np.sin(lam0) / (8 * S * np.cos(lam0 / 2) ** 2)
        c = V.T @ g0
        sinf, sref, sg = 2 * xinf - 1, 2 * xref - 1, -np.sign(g0)
        rec["linear"] = {"eta0": eta0, "mu_min": float(mu[0]), "mu_max": float(mu[-1]),
                         "tau": 4 / (eta0 * abs(mu[0])), "inf_vs_grad": float(np.mean(sinf == sg)),
                         "ref_vs_grad": float(np.mean(sref == sg)), "read": {}}
        for j, p in enumerate(READ_PS):
            tt, s = B["rfirst"][j, a], B["rsign"][j, a]
            ok = tt > 0
            lin = np.array([linear_sign(tt[i], i, V, mu, c, eta0) if ok[i] else 0.0 for i in range(n)])
            flip = ok & (s != sg)
            rec["linear"]["read"]["%g" % p] = {
                "share_read": float(ok.mean()), "t": [int(v) for v in tt],
                "grad": float(np.mean((s == sg)[ok])), "lin": float(np.mean((s == lin)[ok])),
                "n_flips": int(flip.sum()), "lin_on_flips": float(np.mean((s == lin)[flip])) if flip.any() else None,
                "inf": float(np.mean((s == sinf)[ok])), "ref": float(np.mean((s == sref)[ok]))}
        out.append(rec)
    json.dump(out, open(os.path.join(HERE, "large_n_flow_n%d.json" % n), "w"))
    print("  n = %d done in %.0f s" % (n, time.time() - t0), file=sys.stderr, flush=True)
    return out


def check(n=20, ks=range(6)):
    """The batched runner against meanfield.mf_run, run by run (logged overlaps to 1e-12: summation order)."""
    B = batch_run(n, list(ks))
    bad = 0
    for a, k in enumerate(ks):
        Q = instance(n, k)
        for b, p in enumerate(PDIFFS):
            r = mf_run(Q, pdiff=p, xref=REFS[(n, k)][0], max_iter=MAX_ITER, tol=TOL,
                       log_at=set(B["log_at"][a]) if p in LOG_FOR else None)
            th = int(B["t_half"][a, b]) if B["t_half"][a, b] > 0 else None
            same = (r["rounds"] == B["rounds"][a, b] and r["t_half"] == th and r["locks"] == B["locks"][a][b]
                    and np.array_equal(r["x"], B["xend"][a, b]) and len(r["log"]) == len(B["logs"][a][b])
                    and all(u[0] == v[0] and u[2:] == v[2:] and abs(u[1] - v[1]) < 1e-12
                            for u, v in zip(r["log"], B["logs"][a][b])))
            if not same:
                bad += 1
                print("differs: n %d k %d p %s | rounds %d/%d t_half %s/%s locks %d/%d" % (
                    n, k, p, r["rounds"], B["rounds"][a, b], r["t_half"], th, len(r["locks"]), len(B["locks"][a][b])))
    print("batched vs mf_run: %d of %d runs differ" % (bad, len(ks) * len(PDIFFS)))


if __name__ == "__main__":
    if os.environ.get("CHECK"):
        check(int(os.environ.get("CHECK_N", 20)))
        sys.exit()
    NS = [int(v) for v in os.environ.get("NS", "20,30,40,50,60,70,80,90,100").split(",")]
    with Pool(int(os.environ.get("PROCS", os.cpu_count() or 1))) as pool:
        parts = pool.map(work_n, sorted(NS, reverse=True), chunksize=1)
    recs = sorted([r for part in parts for r in part], key=lambda r: (r["n"], r["k"]))
    path = os.path.join(HERE, "large_n_flow.json")
    json.dump(recs, open(path, "w"))
    for n in NS:
        os.remove(os.path.join(HERE, "large_n_flow_n%d.json" % n))
    print("records written to", os.path.relpath(path, os.path.dirname(HERE)), "| p_diff:", PDIFFS)
