"""Classical baselines for the VQPM / QAOA comparison.

simulated_annealing: single-bit-flip Metropolis annealing over bitstrings
for E(x) = sum_{i<=j} Q[i, j] x_i x_j (same objective and bit convention as
vqpm.qubo_energies: bit 0 = most significant bit of the state index).
"""

import numpy as np


def qubo_energy(Q, x):
    """E(x) for a 0/1 vector x (upper triangle of Q, diagonal included)."""
    x = np.asarray(x, dtype=float)
    return float(x @ np.triu(Q) @ x)


def bits_to_state(x):
    n = len(x)
    return int(sum(int(b) << (n - 1 - i) for i, b in enumerate(x)))


def simulated_annealing(Q, n_sweeps=1000, n_restarts=1, T0=None, T1=None, rng=None):
    """Simulated annealing with a geometric temperature schedule.

    Each sweep proposes one flip per bit (random order); a flip of bit k
    changes the energy by dE = (1 - 2 x_k) (Q_kk + sum_{j != k} J_kj x_j)
    with J = triu(Q) + triu(Q)^T - 2 diag(Q) (O(n) per proposal, local fields
    kept up to date). n_restarts independent chains run side by side.

    Defaults: T0 = mean_k (|Q_kk| + sum_j |J_kj|) / 2 (typical size of a
    flip), T1 = T0 / 1000.

    Returns dict(state, energy, bits, proposals (= n_restarts * n_sweeps * n),
    chain_energies).
    """
    rng = np.random.default_rng(rng)
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    Qt = np.triu(Q)
    h = np.diag(Qt).copy()
    J = Qt + Qt.T
    np.fill_diagonal(J, 0.0)
    if T0 is None:
        T0 = float(np.mean(np.abs(h) + np.abs(J).sum(axis=1))) / 2 or 1.0
    if T1 is None:
        T1 = T0 / 1000.0
    R = int(n_restarts)
    x = rng.integers(0, 2, size=(R, n)).astype(float)
    field = x @ J  # (R, n): sum_j J_kj x_j
    energy = np.einsum("ri,ij,rj->r", x, Qt, x)
    best_e = energy.copy()
    best_x = x.copy()
    temps = T0 * (T1 / T0) ** (np.arange(n_sweeps) / max(n_sweeps - 1, 1))
    rows = np.arange(R)
    for T in temps:
        for k in rng.permutation(n):
            xk = x[:, k]
            dE = (1.0 - 2.0 * xk) * (h[k] + field[:, k])
            accept = (dE <= 0) | (rng.random(R) < np.exp(-np.maximum(dE, 0) / T))
            if not accept.any():
                continue
            sgn = np.where(accept, 1.0 - 2.0 * xk, 0.0)  # +1: 0->1, -1: 1->0
            x[:, k] = xk + sgn
            field += sgn[:, None] * J[k][None, :]
            energy += np.where(accept, dE, 0.0)
            better = energy < best_e - 1e-12
            if better.any():
                best_e[better] = energy[better]
                best_x[better] = x[better]
    r = int(np.argmin(best_e))
    bits = best_x[r].astype(int)
    return {
        "state": bits_to_state(bits),
        "energy": float(qubo_energy(Q, bits)),
        "bits": bits.tolist(),
        "proposals": int(R * n_sweeps * n),
        "chain_energies": best_e[rows].tolist(),
    }
