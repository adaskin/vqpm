"""First-order (mean-field) form of the VQPM product-state map, and locking on top of it.

To first order in the spread of the phases, the VQPM round
    pi_i' = pi_i + pi_i (1 - pi_i) dW/dpi_i / W            (Eq. app-wright)
becomes
    pi_i' = pi_i - eta(pi) pi_i (1 - pi_i) g_i(pi),   g_i = dEbar/dpi_i = Q_ii + sum_{j != i} Q_ij pi_j,
    eta(pi) = pi sin(lam_bar) / (8 S W_bar),          lam_bar = pi/4 + (pi/4) Ebar(pi) / S,  W_bar = cos^2(lam_bar/2),
i.e. gradient descent on the mean-field energy Ebar(pi) = sum_i Q_ii pi_i + sum_{i<j} Q_ij pi_i pi_j in the logit
coordinates of pi (theta_i = ln(pi_i/(1-pi_i)) moves by about -eta g_i). One round costs O(n^2) and needs no samples.

This module provides the map (mf_run) with optional locking and the exact map (exact_run, by enumeration, n <= 20)
for comparison (run this file for the check at n = 8, 12). Used by large_n_flow.py.
"""

import numpy as np


def mf_setup(Q):
    T = np.triu(Q)
    S = float(np.abs(T).sum())
    h = np.diag(T).copy()
    J = T + T.T
    np.fill_diagonal(J, 0.0)
    return T, S, h, J


def ebar(h, J, pi):
    return float(h @ pi + 0.5 * pi @ J @ pi)


def _bits(pi):
    """Sign pattern of the marginals (bit i = 1 if pi_i > 1/2) as a hex string."""
    return format(int("".join("1" if v > 0.5 else "0" for v in pi), 2), "x")


def mf_step(pi, h, J, S):
    g = h + J @ pi
    lam = np.pi / 4 + np.pi / 4 * ebar(h, J, pi) / S
    eta = np.pi * np.sin(lam) / (8 * S * np.cos(lam / 2) ** 2)
    return pi - eta * pi * (1 - pi) * g


def overlap(pi, s):
    """(1/n) sum_i s_i (2 pi_i - 1), s_i = +-1 the target string: 1 at the target, 0 at the centre."""
    return float(np.mean(s * (2 * pi - 1)))


def mf_run(Q, pdiff=None, xref=None, max_iter=10**6, tol=1e-4, log_every=None, step=None, log_at=None):
    """Iterate the map from the uniform start.

    pdiff None: no locking. Otherwise lock qubit i when |pi_i' - (1 - pi_i')| > pdiff (the code's rule,
    applied to the new marginals, without rounding).
    Stops when every free marginal is within tol of 0 or 1 at a vertex that is stable for the free qubits
    (every free single flip raises the energy), or after max_iter rounds.
    Returns dict with the final string, rounds, the locks (round, qubit, value) and an overlap log.
    """
    T, S, h, J = mf_setup(Q)
    n = len(h)
    step = step or mf_step
    pi = np.full(n, 0.5)
    free = np.ones(n, bool)
    locks = []
    log = []
    s = None if xref is None else 2 * np.asarray(xref) - 1
    if xref is not None:
        xr = np.asarray(xref, float)
        lam_ref = np.pi / 4 + np.pi / 4 * float(xr @ T @ xr) / S
        w_ref = np.cos(lam_ref / 2) ** 2
    t_half = None
    t = 0
    for t in range(1, max_iter + 1):
        new = step(pi, h, J, S)
        new[~free] = pi[~free]
        if pdiff is not None:
            d = 2 * new - 1
            hit = free & (np.abs(d) > pdiff)
            for i in np.flatnonzero(hit):
                v = 1.0 if d[i] > 0 else 0.0
                locks.append((t, int(i), int(v)))
                new[i] = v
            free &= ~hit
        pi = new
        if s is not None and t_half is None:
            pref = float(np.prod(np.where(s > 0, pi, 1 - pi)))
            lam = np.pi / 4 + np.pi / 4 * ebar(h, J, pi) / S
            if pref * w_ref / np.cos(lam / 2) ** 2 >= 0.5:
                t_half = t          # first round in which x_ref has probability >= 1/2 after the filter
        if s is not None and ((log_every and (t % log_every == 0 or t == 1)) or (log_at and t in log_at)):
            log.append((t, overlap(pi, s), int((~free).sum()), _bits(pi)))
        if not free.any():
            break
        if np.all(np.minimum(pi[free], 1 - pi[free]) < tol):
            # stop only at a stable vertex: every free single flip must raise the energy
            # (Prop. stability); near an unstable vertex the escaping coordinate keeps growing
            xv = (pi > 0.5).astype(float)
            if np.all(((1 - 2 * xv) * (h + J @ xv))[free] > 0):
                break
    x = (pi > 0.5).astype(float)
    if s is not None and (log_every or log_at):
        log.append((t, overlap(pi, s), int((~free).sum()), _bits(pi)))
    return {"x": x, "rounds": t, "locks": locks, "log": log, "t_half": t_half,
            "converged": bool(np.all(np.minimum(pi, 1 - pi) < tol))}


def exact_step_factory(Q):
    """Exact Eq. (app-vqpm) by enumeration, as a drop-in replacement for mf_step (n <= 20)."""
    T, S, h, J = mf_setup(Q)
    n = Q.shape[0]
    k = np.arange(1 << n)
    B = ((k[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1).astype(float)
    E = np.einsum("ij,ij->i", B @ T, B)
    w = np.cos((np.pi / 4 + np.pi / 4 * E / S) / 2) ** 2

    def step(pi, h_, J_, S_):
        lp = B @ np.log(np.clip(pi, 1e-300, 1)) + (1 - B) @ np.log(np.clip(1 - pi, 1e-300, 1))
        pw = np.exp(lp) * w
        return (pw @ B) / pw.sum()

    return step


if __name__ == "__main__":
    # Check against the exact map, Eq. (app-vqpm), by enumeration: 30 instances each at n = 8 and 12 from the
    # paper's family (random_qubo, seed 2026 + n), iterated without locking from the uniform start until the
    # marginals settle at a stable vertex. Run from the repository root:
    #     python3 theory_checks/meanfield.py > theory_checks/meanfield.out
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from vqpm import random_qubo
    print("First-order map vs exact map, no locking, from the uniform start (tol 1e-4, stable vertex)")
    for n in (8, 12):
        np.random.seed(2026 + n)
        same = opt_m = opt_e = 0
        rr = []
        K = 30
        for _ in range(K):
            Q = random_qubo(n)
            T, S, h, J = mf_setup(Q)
            k = np.arange(1 << n)
            B = ((k[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1).astype(float)
            xs = B[np.argmin(np.einsum("ij,ij->i", B @ T, B))]
            a = mf_run(Q, tol=1e-4)
            b = mf_run(Q, tol=1e-4, step=exact_step_factory(Q))
            same += np.array_equal(a["x"], b["x"])
            opt_m += np.array_equal(a["x"], xs)
            opt_e += np.array_equal(b["x"], xs)
            rr.append((a["rounds"], b["rounds"]))
        rr = np.array(rr, float)
        print("n = %2d: same end point %d/%d; end point is the optimum: first-order %d, exact %d; rounds (median) "
              "first-order %d, exact %d, ratio %.2f" % (n, same, K, opt_m, opt_e, np.median(rr[:, 0]),
                                                        np.median(rr[:, 1]), np.median(rr[:, 0] / rr[:, 1])))
