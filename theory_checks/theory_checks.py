"""Numerical checks for the theoretical analysis of the paper.

Run from the repository root:
    python3 theory_checks/theory_checks.py > theory_checks/theory_checks.out

Everything uses exact (statevector-level) probabilities and the repository's own
functions (random_qubo, prepare_new_state, calculate_prob, hoeffding_pdiff) from
vqpm.py, so the numbers refer to the algorithm as implemented.
Runtime: more than half an hour on one core.
"""

import contextlib
import io
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from vqpm import calculate_prob, hoeffding_pdiff, prepare_new_state, random_qubo  # noqa: E402

SEED = 2026
N_INST = int(os.environ.get("N_INST", 100))
quiet = contextlib.redirect_stdout(io.StringIO())


# ----------------------------------------------------------------------------
# helpers (qubit 1 = most significant bit, as in vqpm.calculate_prob)
# ----------------------------------------------------------------------------
def bit_table(n):
    k = np.arange(2**n)
    return ((k[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1).astype(float)


def energies(Q, B):
    """E(x) = sum_{i<=j} Q_ij x_i x_j  (upper triangle incl. diagonal), as in the code."""
    return np.sum((B @ np.triu(Q)) * B, axis=1)


def paper_phases(Q, B):
    """adjust_unitary_phase: lambda = pi/4 + (pi/4) E / S,  S = sum_{i<=j} |Q_ij|."""
    S = np.sum(np.abs(np.triu(Q)))
    return np.pi / 4 + (np.pi / 4) * energies(Q, B) / S


def tight_phases(Q, B, a=0.0, b=np.pi / 2):
    """lambda = a + (b-a)(E-L)/(U-L) with L,U = sums of negative/positive upper-triangle entries."""
    T = np.triu(Q)
    L, U = T[T < 0].sum(), T[T > 0].sum()
    return a + (b - a) * (energies(Q, B) - L) / (U - L)


def product_dist(pi, B):
    """p(x) = prod_i pi_i^{x_i} (1-pi_i)^{1-x_i}, pi_i = Pr(x_i = 1)."""
    return np.prod(np.where(B == 1, pi[None, :], 1 - pi[None, :]), axis=1)


def flip_index(n):
    idx = np.arange(2**n)
    return np.stack([idx ^ (1 << (n - 1 - i)) for i in range(n)], axis=1)


def local_minima_mask(E, F):
    """x is a strict 1-flip local minimum if every single bit flip raises the energy."""
    return np.all(E[F] > E[:, None], axis=1)


def instances(n, count=N_INST, seed=SEED):
    np.random.seed(seed + n)
    return [random_qubo(n) for _ in range(count)]


def section(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


# ----------------------------------------------------------------------------
# Two-qubit example: coherent iteration vs VQPM, and the iteration bound with the initial mass A
# ----------------------------------------------------------------------------
section("Two-qubit example  lambda(x) = pi/2 (x1 + x2 - x1 x2), order 00,01,10,11")
lam = np.array([0, np.pi / 2, np.pi / 2, np.pi / 2])
u = np.exp(1j * lam)
w = np.cos(lam / 2) ** 2


def filt(vec):
    out = (vec + u * vec) / 2
    return out / np.linalg.norm(out)


psi = np.ones(4) / 2
r1 = filt(psi)
coh2 = filt(r1)
with quiet:
    vin, _ = prepare_new_state(r1, 2, {}, pdiff=1, precision=15, locking="no")
v2 = filt(vin)
print("round 1          :", np.round(np.abs(r1) ** 2 * 5, 10), "/5")
print("coherent round 2 :", np.round(np.abs(coh2) ** 2 * 7, 10), "/7")
print("VQPM input rd 2  :", np.round(np.abs(vin) ** 2 * 25, 10), "/25")
print("VQPM round 2     :", np.round(np.abs(v2) ** 2 * 17, 10), "/17")
print("P(x1=0) coherent = %.6f (5/7 = %.6f); VQPM = %.6f (12/17 = %.6f)"
      % (calculate_prob(coh2, 1)[0], 5 / 7, calculate_prob(v2, 1)[0], 12 / 17))
B2 = bit_table(2)
for t, pi in enumerate([np.array([0.5, 0.5]), np.array([0.4, 0.4])]):
    print("Pr(control=0) at VQPM round %d: %.6f" % (t + 1, product_dist(pi, B2) @ w))
A, R2, eps = 0.25, 0.5, 0.25
print("bound without A: k >= ln(eps)/ln(R^2) = %.3f  -> k = 2" % (np.log(eps) / np.log(R2)))
kc = (np.log((1 - A) / A) + np.log((1 - eps) / eps)) / np.log(1 / R2)
print("corrected: k >= [ln((1-A)/A) + ln((1-eps)/eps)] / ln(1/R^2) = %.3f -> k = %d" % (kc, int(np.ceil(kc))))
for k in range(1, 6):
    print("  k=%d  exact coherent P(00) = %.4f   lower bound 1/(1+((1-A)/A)R^2k) = %.4f   "
          "Pr(all k controls = 0) = %.4f" % (k, 1 / (1 + 3 * 2.0**-k), 1 / (1 + 3 * R2**k),
                                           (1 + 3 * 2.0**-k) / 4))


# ----------------------------------------------------------------------------
# Phase map, gaps and coherent iteration counts
# ----------------------------------------------------------------------------
section("Paper phase map on random instances (median over %d instances, seed %d+n)" % (N_INST, SEED))
print(" n | lam_min lam_max | gap(rad) | 1-R^2    | 1-rho*   | k old13 | k corr | log10 Pr(k corr rounds succeed)")
for n in [4, 8, 12, 16]:
    B, F = bit_table(n), flip_index(n)
    rows = []
    for Q in instances(n):
        lam = paper_phases(Q, B)
        order = np.argsort(lam)
        ld, ls = lam[order[0]], lam[order[1]]
        wv = np.cos(lam / 2) ** 2
        wd = wv[order[0]]
        R2 = wv[order[1]] / wd
        rho = np.max(wv[F[order[0]]]) / wd       # local rate at the optimum
        A = 1 / 2**n
        k13 = np.log(0.5) / np.log(R2)
        kc = np.log((1 - A) / A) / np.log(1 / R2)  # eps = 1/2
        lw = np.ceil(kc) * np.log(wv)                # log of w^k, summed in log space
        surv = (np.max(lw) + np.log(np.sum(np.exp(lw - np.max(lw)))) - n * np.log(2)) / np.log(10)
        rows.append([lam.min(), lam.max(), ls - ld, 1 - R2, 1 - rho, k13, kc, surv])
    r = np.median(np.array(rows), axis=0)
    print("%2d | %.3f  %.3f  | %.2e | %.2e | %.2e | %7.0f | %6.0f | %8.1f"
          % (n, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]))
print("(window allowed by the map: [0, %.3f]; k for eps = 1/2; Pr = sum_x 2^-n cos^(2k)(lambda/2))" % (np.pi / 2))

section("Per-round contrast vs post-selection rate for other windows (n = 10)")
n = 10
B = bit_table(n)
maps = {
    "paper  +-S -> [0, pi/2]": lambda Q: paper_phases(Q, B),
    "tight [L,U] -> [0, pi/2]": lambda Q: tight_phases(Q, B, 0, np.pi / 2),
    "tight [L,U] -> [0, 0.9pi]": lambda Q: tight_phases(Q, B, 0, 0.9 * np.pi),
    "paper shifted -> [pi/2, pi]": lambda Q: paper_phases(Q, B) + np.pi / 2,
}
print("map                          | 1-R^2 (med) | tan(ld/2)*gap (med) | Pr(control=0) at uniform start (med)")
for name, f in maps.items():
    rows = []
    for Q in instances(n):
        lam = f(Q)
        s = np.sort(lam)
        R2 = (np.cos(s[1] / 2) / np.cos(s[0] / 2)) ** 2
        rows.append([1 - R2, np.tan(s[0] / 2) * (s[1] - s[0]), np.mean(np.cos(lam / 2) ** 2)])
    r = np.median(np.array(rows), axis=0)
    print("%-28s | %.2e    | %.2e            | %.3f" % (name, r[0], r[1], r[2]))


# ----------------------------------------------------------------------------
# Exact VQPM recursion (no rounding, no locking)
# ----------------------------------------------------------------------------
section("Exact VQPM map pi -> marginals of (p_pi * w): monotone W and limits")


def run_exact(wv, B, T=20000, tol=1e-6):
    pi = np.full(B.shape[1], 0.5)
    Ws = []
    for t in range(T):
        q = product_dist(pi, B) * wv
        W = q.sum()
        Ws.append(W)
        pi_new = (q @ B) / W
        if np.max(np.minimum(pi_new, 1 - pi_new)) < tol:
            pi = pi_new
            break
        pi = pi_new
    return pi, np.array(Ws), t + 1


print(" n | runs | min dW (>= 0?) | at vertex by T | rounded limit is 1-flip local min | rounded limit is optimum |"
      " max dist of unconverged | #strict 1-flip local minima (median, max) | rounds to vertex (median)")
for n in [6, 8, 10, 12]:
    B, F = bit_table(n), flip_index(n)
    stats = []
    for Q in instances(n):
        E = energies(Q, B)
        lam = paper_phases(Q, B)
        wv = np.cos(lam / 2) ** 2
        pi, Ws, iters = run_exact(wv, B)
        if np.max(np.minimum(pi, 1 - pi)) >= 1e-4:   # slow tail (rate close to 1): run longer
            pi, Ws, iters = run_exact(wv, B, T=300000)
        dW = np.diff(Ws).min() if len(Ws) > 1 else 0.0
        dist = np.max(np.minimum(pi, 1 - pi))
        vertex = dist < 1e-4
        x = int(np.round(pi) @ (1 << (n - 1 - np.arange(n))))
        lm = local_minima_mask(E, F)
        stats.append([dW, vertex, lm[x], x == np.argmin(E), lm.sum(), iters, 0.0 if vertex else dist])
    s = np.array(stats, dtype=float)
    print("%2d | %4d | %.1e        | %4.0f%%          | %4.0f%%                             | %4.0f%%                   | %.3f                   | %d, %d                                    | %d"
          % (n, len(s), s[:, 0].min(), 100 * s[:, 1].mean(), 100 * s[:, 2].mean(), 100 * s[:, 3].mean(),
             s[:, 6].max(), np.median(s[:, 4]), s[:, 4].max(), np.median(s[:, 5])))

print("\nStrict 1-flip local minima of E (median count over instances):")
for n in [4, 8, 12, 16]:
    B, F = bit_table(n), flip_index(n)
    c = [local_minima_mask(energies(Q, B), F).sum() for Q in instances(n)]
    print("  n=%2d  median %5.0f   mean %7.1f   max %5d" % (n, np.median(c), np.mean(c), np.max(c)))


# ----------------------------------------------------------------------------
# Paper settings (p_diff = 0.01, precision 3, 30 iterations): what locking does
# ----------------------------------------------------------------------------
section("Paper settings: when qubits lock, mean-field sign of each lock, wrong locks, rounding stalls")


def couplings(Q):
    T = np.triu(Q)
    return np.diag(T), T + T.T - 2 * np.diag(np.diag(T))   # linear terms, symmetric off-diagonal couplings


def run_paper(u, n, x_star, Q=None, pdiff=0.01, precision=3, max_iter=30, oracle=True):
    """Same loop and stop rule as vqpm.vqpm_for_qubo (locking='yes'), with a log of every lock."""
    N = 2**n
    in_vec = np.ones(N, dtype=complex) / np.sqrt(N)
    q_states = {}
    log = []          # (iteration, qubit index 0-based, value, mean-field energy change of that lock)
    stalled = False
    prev_in = None
    Ws = []           # acceptance Pr(control = 0) of each round
    j = 0
    for j in range(max_iter):
        psi = (in_vec + u * in_vec) / 2
        Ws.append(np.linalg.norm(psi) ** 2)
        psi /= np.linalg.norm(psi)
        probs = np.abs(psi) ** 2
        pm = probs[x_star] if oracle else probs.max()
        if pm >= 0.5 or (oracle and pm == 0):
            break
        if not oracle and len(q_states) == n:
            break
        before = set(q_states)
        pi_post = np.array([np.round(calculate_prob(psi, q), precision)[1] for q in range(1, n + 1)])
        for q, v in q_states.items():
            pi_post[q - 1] = np.argmax(v)
        with quiet:
            in_vec, q_states = prepare_new_state(psi, n, q_states, pdiff=pdiff, precision=precision)
        if Q is not None:
            h, J = couplings(Q)
            for q in set(q_states) - before:
                i, sv = q - 1, int(np.argmax(q_states[q]))
                g = h[i] + J[i] @ pi_post
                log.append((j, i, sv, (sv - pi_post[i]) * g))
        if prev_in is not None and np.allclose(prev_in, in_vec) and len(q_states) < n:
            stalled = True
        prev_in = in_vec.copy()
    run_paper.Ws = np.array(Ws)
    return int(np.argmax(np.abs(psi) ** 2)), log, stalled, q_states, j + 1


print(" n | rounds used | 1st lock (round) | locks lowering mean-field E | runs with wrong lock |"
      " failures with wrong lock | stalled by rounding | found x* | returned state 1-flip local min |"
      " rounds where Pr(c=0) fell")
for n in [8, 12, 16]:
    B, F = bit_table(n), flip_index(n)
    stats = []
    for Q in instances(n):
        E = energies(Q, B)
        u = np.exp(1j * paper_phases(Q, B))
        xs = int(np.argmin(E))
        found, log, stalled, qs, used = run_paper(u, n, xs, Q=Q)
        first = min([e[0] for e in log]) + 1 if log else np.nan
        lowering = [e[3] < 0 for e in log]
        wrong = any(B[xs][e[1]] != e[2] for e in log)
        lm = local_minima_mask(E, F)
        dW = np.diff(run_paper.Ws)
        stats.append([used, first, np.mean(lowering) if log else np.nan, wrong, stalled, found == xs, lm[found],
                      len(log), np.sum(dW < -1e-12), len(dW)])
    s = np.array(stats, dtype=float)
    fail = s[:, 5] == 0
    print("%2d | %5.1f       | %5.1f            | %5.1f%% (%d locks)          | %4.0f%%                |"
          " %4.0f%% of %3d             | %4.0f%%               | %4.0f%%    | %4.0f%%                            | %.1f%% of %d"
          % (n, np.mean(s[:, 0]), np.nanmedian(s[:, 1]), 100 * np.nansum(s[:, 2] * s[:, 7]) / np.sum(s[:, 7]),
             np.sum(s[:, 7]), 100 * s[:, 3].mean(), 100 * s[fail, 3].mean() if fail.any() else 0, fail.sum(),
             100 * s[:, 4].mean(), 100 * s[:, 5].mean(), 100 * s[:, 6].mean(),
             100 * s[:, 8].sum() / max(s[:, 9].sum(), 1), s[:, 9].sum()))


# ----------------------------------------------------------------------------
# Hoeffding threshold: the rule as coded vs a derived rule
# ----------------------------------------------------------------------------
section("Hoeffding threshold")
T = 30
vals = {n: [hoeffding_pdiff(j, num_qubits=n, total_iterations=T) for j in range(T)] for n in range(1, 19)}
allv = np.concatenate(list(vals.values()))
print("Hoeffding rule as coded (M=100, delta_total=0.5, T=30): p_diff in [%.4f, %.4f] over n=1..18" % (allv.min(), allv.max()))
for n in [4, 8, 12, 15, 18]:
    print("  n=%2d: [%.4f, %.4f]" % (n, min(vals[n]), max(vals[n])))
H = sum(1 / (T - j) for j in range(T))
print("sum_k delta_k with delta_k = delta_total/(T-k): %.3f x delta_total = %.2f for delta_total = 0.5" % (H, 0.5 * H))
print("\nDerived rule: lock iff |P0hat - P1hat| >= tau = 2 sqrt(ln(2nT/delta)/(2M)), M = post-selected shots")
print("post-selected shots M per iteration needed for tau = 0.01 (T = 30):")
for delta in [0.05, 0.5]:
    row = []
    for n in [8, 12, 16, 20]:
        M = 2 * np.log(2 * n * T / delta) / 0.01**2
        row.append("n=%d: %.2e" % (n, M))
    print("  delta=%.2f  " % delta + "   ".join(row))
print("\nStatistical wrong-lock bound exp(-M tau^2/2) for an undecided qubit (D = 0), tau = 0.01:")
for M in [1e3, 1e4, 1e5, 1e6]:
    print("  M=%.0e  bound %.3g" % (M, np.exp(-M * 0.01**2 / 2)))
print("\nShots that give a marginal standard error equal to the rounding error 0.5*10^-d at P = 1/2:")
for d in [3, 6]:
    print("  precision %d: M = 10^%d" % (d, 2 * d))


# ----------------------------------------------------------------------------
# Moment (degree-2 SoS) view
# ----------------------------------------------------------------------------
section("Moment-matrix (degree-2 SoS) view")


def moments(prob, B):
    """Y = E[(1,x)(1,x)^T] for a distribution over x in {0,1}^n."""
    X = np.hstack([np.ones((B.shape[0], 1)), B])
    return X.T @ (prob[:, None] * X)


# (a) two-qubit example: what the product re-preparation discards
q1 = np.abs(r1) ** 2
Y1 = moments(q1, B2)
print("two-qubit example, first-round filtered distribution q1 = (2,1,1,1)/5")
print("  E[x1] = E[x2] = %.4f,  E[x1 x2] = %.4f,  product value E[x1]E[x2] = %.4f,  covariance lost = %.4f"
      % (Y1[0, 1], Y1[1, 2], Y1[0, 1] * Y1[0, 2], Y1[1, 2] - Y1[0, 1] * Y1[0, 2]))


def mean_field_energy(Q, pi):
    T = np.triu(Q)
    return np.diag(T) @ pi + pi @ (T - np.diag(np.diag(T))) @ pi


# (b) exact locking identity on random product states
rng = np.random.default_rng(SEED)
worst = 0.0
for _ in range(1000):
    n = 8
    Q = random_qubo(n)
    T = np.triu(Q)
    Qs = T + T.T - 2 * np.diag(np.diag(T))
    pi = rng.random(n)
    i = rng.integers(n)
    s = float(rng.integers(2))
    g = np.diag(T)[i] + Qs[i] @ pi
    pi2 = pi.copy()
    pi2[i] = s
    worst = max(worst, abs(mean_field_energy(Q, pi2) - mean_field_energy(Q, pi) - (s - pi[i]) * g))
print("locking identity  E(pi with pi_i:=s) - E(pi) = (s - pi_i) g_i : max abs error over 1000 random cases = %.1e" % worst)

# (c) Shor (degree-2 SoS) lower bound vs optimum and vs VQPM output (paper settings)
try:
    import cvxpy as cp
except ImportError:  # pragma: no cover
    cp = None
if cp is None:
    print("cvxpy not installed: skipping the SDP lower bound")
else:
    print("\n n | instances | (E* - SDP)/(Emax - Emin) median [max] | SDP tight (gap < 1e-4) |"
          " VQPM (E_ret - E*)/(Emax - Emin) median | VQPM (E_ret - SDP)/(Emax - Emin) median")
    for n in [8, 12, 16]:
        B = bit_table(n)
        rows = []
        for Q in instances(n)[:30]:
            E = energies(Q, B)
            T = np.triu(Q)
            Y = cp.Variable((n + 1, n + 1), symmetric=True)
            cons = [Y >> 0, Y[0, 0] == 1] + [Y[i + 1, i + 1] == Y[0, i + 1] for i in range(n)]
            obj = sum(T[i, i] * Y[0, i + 1] for i in range(n)) + \
                sum(T[i, j] * Y[i + 1, j + 1] for i in range(n) for j in range(i + 1, n))
            prob = cp.Problem(cp.Minimize(obj), cons)
            prob.solve(solver=cp.CLARABEL)
            sdp = prob.value
            rngE = E.max() - E.min()
            u = np.exp(1j * paper_phases(Q, B))
            found, *_ = run_paper(u, n, int(np.argmin(E)))
            rows.append([(E.min() - sdp) / rngE, (E[found] - E.min()) / rngE, (E[found] - sdp) / rngE])
        r = np.array(rows)
        print("%2d | %9d | %.3f [%.3f]                         | %4.0f%%                  | %.3f                                  | %.3f"
              % (n, len(r), np.median(r[:, 0]), r[:, 0].max(), 100 * np.mean(r[:, 0] < 1e-4),
                 np.median(r[:, 1]), np.median(r[:, 2])))
