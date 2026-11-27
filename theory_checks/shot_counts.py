"""Shot counts: quantum device versus classical sampler.

Run from the repository root:
    python3 theory_checks/shot_counts.py > theory_checks/shot_counts.out

Same instances (random_qubo, seed 2026 + n, 100 per n), phase map and VQPM loop
(p_diff = 0.01, precision 3, at most 30 rounds, locking) as theory_checks.py.

For every round of every run we take the product input distribution p of that round and
w(x) = cos^2(lambda(x)/2), and compute
  W     = sum_x p(x) w(x)                     acceptance Pr(c = 0) of one shot,
  ratio = max_i E_p[w^2 (x_i - pi_i')^2] / E_p[w (x_i - pi_i')^2]   over unlocked qubits i,
where pi' is the post-selected marginal.  For N draws, a marginal estimated from post-selected
shots (quantum device, or classical accept/reject) has asymptotic variance
E_p[w (x_i - pi_i')^2] / (N W^2); the classical sampler that weights every draw by w(x)
instead of drawing the control bit has E_p[w^2 (x_i - pi_i')^2] / (N W^2).  So the weighted
sampler needs `ratio` times as many draws for the same variance, and ratio lies in
[min w, max w].
"""

import contextlib
import io
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from vqpm import prepare_new_state, random_qubo  # noqa: E402

SEED = 2026
N_INST = int(os.environ.get("N_INST", 100))
quiet = contextlib.redirect_stdout(io.StringIO())


def bit_table(n):
    k = np.arange(2**n)
    return ((k[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1).astype(float)


def energies(Q, B):
    return np.sum((B @ np.triu(Q)) * B, axis=1)


def paper_phases(Q, B):
    S = np.sum(np.abs(np.triu(Q)))
    return np.pi / 4 + (np.pi / 4) * energies(Q, B) / S


def instances(n, count=N_INST, seed=SEED):
    np.random.seed(seed + n)
    return [random_qubo(n) for _ in range(count)]


def round_stats(p, w, B):
    W = p @ w
    pi_new = (p * w) @ B / W
    D2 = (B - pi_new[None, :]) ** 2
    v_rej = (p * w) @ D2
    v_wt = (p * w**2) @ D2
    live = v_rej > 1e-15
    ratio = np.max(v_wt[live] / v_rej[live]) if live.any() else np.nan
    # Four estimators of pi', asymptotic variance per draw (times N), worst unlocked qubit:
    #   dev_ratio : device, accepted shots only (as in the paper)
    #   dev_cv    : device, all shots, control variate E_p[x_i] = pi_i (pi known from the R_y angles)
    #   cl_ratio  : classical, every draw weighted by w(x)            (Eq. app-weighted)
    #   cl_cv     : classical, weighted draws with the same control variate
    pi_in = p @ B
    A = B - pi_in[None, :]
    dlt = pi_new - pi_in
    v_dc = (p * w) @ (A - dlt[None, :] - A * W) ** 2 + (p * (1 - w)) @ (A * W) ** 2
    v_cc = p @ (A * (w - W)[:, None] - dlt[None, :] * w[:, None]) ** 2
    worst = {}
    if live.any():
        for name, v in (("dev_ratio", v_rej), ("dev_cv", v_dc), ("cl_ratio", v_wt), ("cl_cv", v_cc)):
            worst[name] = np.max(v[live]) / W**2
    round_stats.worst = worst
    return W, ratio


def run(u, w, B, n, x_star, pdiff=0.01, precision=3, max_iter=30):
    """Same loop and stop rule as theory_checks.run_paper (oracle stop at Pr(x*) >= 1/2)."""
    N = 2**n
    in_vec = np.ones(N, dtype=complex) / np.sqrt(N)
    q_states = {}
    out = []
    run.worst = []
    for _ in range(max_iter):
        p = np.abs(in_vec) ** 2
        out.append(round_stats(p, w, B))
        run.worst.append(round_stats.worst)
        psi = (in_vec + u * in_vec) / 2
        psi /= np.linalg.norm(psi)
        pm = np.abs(psi[x_star]) ** 2
        if pm >= 0.5 or pm == 0:
            break
        with quiet:
            in_vec, q_states = prepare_new_state(psi, n, q_states, pdiff=pdiff, precision=precision)
    return out


T, DELTA, TAU = 30, 0.05, 0.01
print("Post-selected shots per round for the derived Hoeffding threshold: M = 2 ln(2nT/delta) / tau^2,"
      " T=%d, delta=%.2f, tau=%.2f" % (T, DELTA, TAU))
print(" n | QUBO terms n(n+1)/2 | M (post-sel.) | W round 1 med [min,max] | W all rounds med [min,max] |"
      " ratio med [max] | max |ratio - W| | quantum shots M/W (med W) | classical weighted draws ratio*M/W")
rows = []
for n in [8, 12, 16]:
    B = bit_table(n)
    W1, Wall, R, dif = [], [], [], []
    EST = {"dev_cv": [], "cl_ratio": [], "cl_cv": []}
    for Q in instances(n):
        lam = paper_phases(Q, B)
        u, w = np.exp(1j * lam), np.cos(lam / 2) ** 2
        xs = int(np.argmin(energies(Q, B)))
        st = run(u, w, B, n, xs)
        for wst in run.worst:
            if wst:
                for key in EST:
                    EST[key].append(wst[key] / wst["dev_ratio"])
        W1.append(st[0][0])
        Wall += [s[0] for s in st]
        R += [s[1] for s in st if not np.isnan(s[1])]
        dif += [abs(s[1] - s[0]) for s in st if not np.isnan(s[1])]
    W1, Wall, R = map(np.array, (W1, Wall, R))
    M = 2 * np.log(2 * n * T / DELTA) / TAU**2
    print("%2d | %4d | %.2e | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.3f [%.3f] | %.1e | %.2e | %.2e"
          % (n, n * (n + 1) // 2, M, np.median(W1), W1.min(), W1.max(), np.median(Wall), Wall.min(),
             Wall.max(), np.median(R), R.max(), max(dif), M / np.median(Wall), np.median(R) * M / np.median(Wall)))
    rows.append((n, M / np.median(Wall), {k: np.array(v) for k, v in EST.items()}))

print("\nDraws needed for the same variance as the device's post-selected (ratio) estimate, worst unlocked"
      " qubit per round; median [min, max] over all rounds; and the resulting draws per round for the derived threshold")
print(" n | device + control variate | classical weighted | classical weighted + control variate |"
      " device CV draws | classical CV draws")
for n, dev, est in rows:
    print("%2d | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.1e [%.1e, %.1e] | %.2e | %.2e"
          % (n, *(f(est["dev_cv"]) for f in (np.median, np.min, np.max)),
             *(f(est["cl_ratio"]) for f in (np.median, np.min, np.max)),
             *(f(est["cl_cv"]) for f in (np.median, np.min, np.max)),
             np.median(est["dev_cv"]) * dev, np.median(est["cl_cv"]) * dev))

print("\nClassical sampler: wall-clock time per round on one core (numpy, float64), weighted"
      " estimator, uniform product input, N draws = M / 0.85")
print(" n | draws per round | seconds per round | microseconds per draw")
rng = np.random.default_rng(SEED)
for n in [8, 12, 16, 20, 30, 50]:
    np.random.seed(SEED + n)
    Q = random_qubo(n)
    Tq = np.triu(Q)
    S = np.sum(np.abs(Tq))
    M = 2 * np.log(2 * n * T / DELTA) / TAU**2
    N = int(np.ceil(M / 0.85))
    pi = np.full(n, 0.5)
    t0 = time.perf_counter()
    reps = 3
    for _ in range(reps):
        est = np.zeros(n)
        tot = 0.0
        for start in range(0, N, 50000):
            m = min(50000, N - start)
            X = (rng.random((m, n)) < pi).astype(float)
            E = np.sum((X @ Tq) * X, axis=1)
            wv = np.cos((np.pi / 4 + (np.pi / 4) * E / S) / 2) ** 2
            est += wv @ X
            tot += wv.sum()
        est /= tot
    dt = (time.perf_counter() - t0) / reps
    print("%2d | %.2e | %.3f | %.3f" % (n, N, dt, 1e6 * dt / N))
