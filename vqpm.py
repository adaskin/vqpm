"""
QPM converges the eigenvector of I+U with the maximum magnitude
that is max |1+e^{iλ}|, where λ is the minimum eigenvalue of H.
The file includes the implementations of variational quantum power method (vqpm) applied to random QUBO,
vqpmForQUBO: it finds the minimum eigenphase of U.

@author: adaskin, updated and modified with the help of DeepSeek/copilot assistant
"""

import numpy as np
import matplotlib.pyplot as plt


def bit_matrix(n, start=0, stop=None):
    """Bit table of the basis states ``start .. stop-1`` of an n-qubit register.

    Row k holds the bits of the integer k with bit 0 = most significant bit,
    i.e. ``B[k, i] = (k >> (n - 1 - i)) & 1`` (same convention as
    ``bin(k)[2:].zfill(n)`` and ``calculate_prob``).
    """
    if stop is None:
        stop = 2**n
    k = np.arange(start, stop, dtype=np.int64)
    shifts = n - 1 - np.arange(n, dtype=np.int64)
    return ((k[:, None] >> shifts[None, :]) & 1).astype(float)


def qubo_energies(n, Q, method="sequential", chunk_bits=18):
    """QUBO values E(x) = sum_{i<=j} Q[i, j] x_i x_j for all 2^n bitstrings.

    Vectorised replacement of the former Python double loop over 2^n states.
    Only the upper triangle (including the diagonal) of Q is used, exactly as
    the loop did; bit 0 is the most significant bit of the state index.

    method:
        'sequential' (default): adds Q[i, j] * x_i * x_j over the pairs
            (i, j), j >= i, in the same order as the old loop. Adding 0.0 is
            exact, so the result is bit-for-bit identical to the old loop and
            all published numbers reproduce exactly.
        'matmul': E = sum((B @ triu(Q)) * B, axis=1). Same values up to
            floating-point rounding (~1e-15), slightly faster for large n.
    States are processed in blocks of 2^chunk_bits to bound memory.
    """
    Q = np.asarray(Q, dtype=float)
    N = 2**n
    block = 2 ** min(n, chunk_bits)
    E = np.empty(N)
    for start in range(0, N, block):
        B = bit_matrix(n, start, start + block)
        if method == "matmul":
            E[start : start + block] = np.sum((B @ np.triu(Q[:n, :n])) * B, axis=1)
        elif method == "sequential":
            e = np.zeros(B.shape[0])
            for i in range(n):
                bi = B[:, i]
                for j in range(i, n):
                    e += Q[i][j] * (bi * B[:, j])
            E[start : start + block] = e
        else:
            raise ValueError(f"unknown method {method!r}")
    return E


def unitary_for_qubo(n, Q):
    """Creates diagonal unitary matrix for QUBO.
    Args:
        n (int): Number of qubits.
        Q (matrix): QUBO matrix.
    Returns:
        array: Diagonal of unitary matrix, exp(1j * E(x)).
    """
    return np.exp(1j * qubo_energies(n, Q))


def test_u_for_qubo(n, Q):
    """QUBO values of all basis states (used to validate the unitary).
    Args:
        n (int): Number of qubits.
        Q (matrix): QUBO matrix.
    Returns:
        array: E(x) for x = 0 .. 2^n - 1.
    """
    return qubo_energies(n, Q)


def random_qubo(n):
    """Generates random QUBO matrix.
    Args:
        n (int): Matrix dimension.
    Returns:
        matrix: Symmetric Q matrix.
    """
    Q = np.random.randn(n, n)
    return Q + Q.T


PHASE_MAPS = ("paper", "tight", "tight_wide", "shifted")


def phase_map_coefficients(Q, phase_map="paper"):
    """Affine map from QUBO value to eigenphase: phase(x) = a * E(x) + b.

    With S = sum |triu(Q)| and L, U = sum of the negative / positive entries
    of triu(Q) (so L <= E(x) <= U):

    'paper'      : a = pi/(4S),       b = pi/4    -> phase in [0, pi/2]
                   (the map used in the paper; E in [-S, S])
    'tight'      : a = (pi/2)/(U-L),  b = -a*L    -> phase in [0, pi/2],
                   (E-L)/(U-L) * pi/2, uses the full interval
    'tight_wide' : a = 0.9*pi/(U-L),  b = -a*L    -> phase in [0, 0.9*pi]
    'shifted'    : a = pi/(4S),       b = 3*pi/4  -> phase in [pi/2, pi]
    All maps are increasing in E, so the ground state has the smallest phase.
    """
    Qt = np.triu(np.asarray(Q, dtype=float))
    if phase_map in ("paper", "shifted"):
        S = np.sum(np.abs(Qt))
        if S == 0:
            raise ValueError("Q is zero")
        a = np.pi / (4 * S)
        b = np.pi / 4 if phase_map == "paper" else 3 * np.pi / 4
        return a, b
    if phase_map in ("tight", "tight_wide"):
        L = np.sum(Qt[Qt < 0])
        U = np.sum(Qt[Qt > 0])
        if U == L:
            raise ValueError("Q is zero")
        width = np.pi / 2 if phase_map == "tight" else 0.9 * np.pi
        a = width / (U - L)
        return a, -a * L
    raise ValueError(f"unknown phase_map {phase_map!r}; choose from {PHASE_MAPS}")


def adjust_unitary_phase(n, Q, phase_map="paper"):
    """Adjusts QUBO matrix for phase calculations.
    Args:
        n (int): Number of qubits.
        Q (matrix): QUBO matrix.
        phase_map (str): one of PHASE_MAPS, see phase_map_coefficients.
            'paper' (default) is the published map E*pi/(4S) + pi/4.
    Returns:
        tuple: Scaled Q (a*Q), unitary diagonal exp(i*phase), phases, and
        expected state (index of the minimum phase = QUBO ground state).
    """
    if phase_map == "paper":
        # unchanged published code path (bit-for-bit)
        max_q = np.sum(np.abs(np.triu(Q)))
        Q_scaled = Q / max_q * np.pi / 4  # in [-pi/4, pi/4]
        u = unitary_for_qubo(n, Q_scaled)
        u *= np.exp(1j * np.pi / 4)  # Shift phases to positive
        phases = np.angle(u)  # Get eigenphases
        expected_state = np.argmin(phases)
        return Q_scaled, u, phases, expected_state
    a, b = phase_map_coefficients(Q, phase_map)
    Q_scaled = Q * a
    phases = qubo_energies(n, Q_scaled) + b  # direct (no wrap-around near pi)
    u = np.exp(1j * phases)
    expected_state = np.argmin(phases)
    return Q_scaled, u, phases, expected_state


def resolve_max_iter(max_iter, n):
    """Iteration budget: an int, or a string 'c*n' / 'cn' (e.g. '10n', '2.5n')
    meaning ceil(c * n)."""
    if isinstance(max_iter, str):
        s = max_iter.replace("*", "").strip()
        if not s.endswith("n"):
            raise ValueError(f"bad iteration budget {max_iter!r}")
        c = float(s[:-1]) if s[:-1] else 1.0
        return int(np.ceil(c * n))
    return int(max_iter)


def calculate_prob(psi, qubit):
    """Computes probabilities for the states of a given qubit.
    Args:
        psi (array): Quantum state vector.
        qubit (int): Index of the qubit (1-based).
    Returns:
        array: Probabilities [prob_0, prob_1].
    """
    n = len(psi)
    logn = int(np.log2(n))
    mask = 1 << (logn - qubit)  # Mask to check the specific qubit
    # Indices where qubit is 1
    indices_1 = (np.arange(n) & mask) != 0
    prob_1 = np.sum(np.abs(psi[indices_1]) ** 2)
    prob_0 = 1.0 - prob_1
    return np.array([prob_0, prob_1])


def prepare_new_state(
    out_vec,
    n,
    q_states,
    pdiff=0.01,
    precision=3,
    qubit_weights=None,
    locking="yes",
    marginals=None,
    lock_log=None,
    iteration=None,
    optimum=None,
    verbose=True,
):
    """Generates a new state based on qubit probabilities.
    Args:
        out_vec (array): Quantum state vector.
        n (int): Number of qubits.
        q_states (dict): Precomputed qubit states.
        pdiff (float): Threshold for collapsing probabilities.
        precision (int or None): Rounding precision (None = no rounding).
        qubit_weights (array): Weights for each qubit.
        marginals (array, optional): estimated P(qubit q = 1) for q = 1..n
            (e.g. from finite shots). If None, exact marginals of out_vec.
        lock_log (list, optional): lock events are appended here as dicts
            {iteration, qubit (1-based), value, p0, p1, margin, threshold,
            matches_optimum}.
        optimum (int, optional): index of the true optimum, used only to
            fill matches_optimum in lock_log.
    Returns:
        tuple: New state vector and updated q_states.
    """
    state = [complex(1)]

    # No locking
    if locking == "no":
        pdiff = 1

    for q in range(1, n + 1):
        if q in q_states:
            state_q = q_states[q]
        else:
            if marginals is None:
                state_q = calculate_prob(out_vec, q)
            else:
                state_q = np.array([1.0 - marginals[q - 1], marginals[q - 1]])
            if precision is not None:
                state_q = np.round(state_q, precision)

            # Scale pdiff by qubit influence (higher weight = stricter threshold)
            if qubit_weights is not None:
                pdiff_scaled = pdiff * qubit_weights[q - 1]  # weights are 0-indexed
                if verbose:
                    print(f"qubit{q} pdiff_scaled {pdiff_scaled}", state_q)
            else:
                pdiff_scaled = pdiff  # weights are 0-indexed

            # Lock this qubit if one outcome dominates by more than pdiff.
            # Any number of qubits (including all of them) can lock in the
            # same iteration; there is no one-per-iteration limit.
            value = None
            if state_q[0] > state_q[1] + pdiff_scaled:
                value = 0
            elif state_q[1] > state_q[0] + pdiff_scaled:
                value = 1
            if value is not None:
                if lock_log is not None:
                    match = None
                    if optimum is not None:
                        match = bool(((int(optimum) >> (n - q)) & 1) == value)
                    lock_log.append(
                        {
                            "iteration": iteration,
                            "qubit": q,
                            "value": value,
                            "p0": float(state_q[0]),
                            "p1": float(state_q[1]),
                            "margin": float(abs(state_q[0] - state_q[1])),
                            "threshold": float(pdiff_scaled),
                            "matches_optimum": match,
                        }
                    )
                state_q = np.array([1.0, 0.0]) if value == 0 else np.array([0.0, 1.0])
                q_states[q] = state_q
            else:
                state_q = np.sqrt(state_q)  # Take amplitude

        state = np.kron(state, state_q)

    return state, q_states


def hoeffding_pdiff(
    iteration: int,
    num_qubits: int = 10,
    total_iterations: int = 20,
    shots_per_iter: int = 100,
    delta_total: float = 0.5,
    lowest_threshold=0.001,
) -> float:
    """
    Dynamically adjusts pdiff using Hoeffding's inequality (published policy
    'hoeffding', kept unchanged so that Figs. 7-8 reproduce).

    Note: the per-iteration split delta_i = delta_total / (T - i) is not a
    valid union bound: summed over i = 0..T-1 it gives delta_total * H_T
    (H_T the T-th harmonic number), e.g. 0.5 * H_30 = 2.0 for T = 30,
    delta_total = 0.5, i.e. no guarantee at all. The formula also uses a
    fixed nominal shot count. Use the 'hoeffding_union' policy of
    vqpm_for_qubo (with shots=...) for a valid bound.

    Parameters:
        iteration: Current iteration number (0-indexed)
        total_iterations: Total planned iterations
        shots_per_iter: Measurement shots per iteration
        delta_total: Total allowed failure probability over all iterations

    Returns:
        pdiff: Adaptive probability difference threshold
    """
    # Distribute error budget across remaining iterations
    remaining_iters = total_iterations - iteration
    delta_i = delta_total / remaining_iters  # (not a valid union bound, see docstring)

    # Hoeffding bound: ε = sqrt(ln(2/δ)/(2M))
    pdiff = np.sqrt(np.log(2 / delta_i) / (20 * num_qubits * shots_per_iter))

    return max(pdiff, lowest_threshold)


def hoeffding_union_threshold(n, total_iterations, delta, m_post, precision=None):
    """Locking threshold of the 'hoeffding_union' policy.

    With M post-selected shots, each marginal estimate satisfies
    |P1_hat - P1| <= eps with probability >= 1 - delta/(n*T), where
    eps = sqrt(ln(2 n T / delta) / (2 M)) (Hoeffding). A union bound over the
    n qubits and T iterations makes all estimates eps-accurate with
    probability >= 1 - delta. Since P0 - P1 = 1 - 2 P1, the sign of P0 - P1 is
    then certain whenever |P0_hat - P1_hat| > 2 eps, so that is the locking
    threshold. If the estimates are rounded to `precision` decimals, the
    rounding error (at most 10**-precision on the difference) is added.
    Returns inf if M = 0 (no lock possible).
    """
    if m_post <= 0:
        return np.inf
    eps = np.sqrt(np.log(2 * n * total_iterations / delta) / (2 * m_post))
    slack = 10.0 ** (-precision) if precision is not None else 0.0
    return 2 * eps + slack


def compute_qubit_weights(Q):
    """Computes influence weights for each qubit.
    Args:
        Q (matrix): QUBO matrix.
    Returns:
        array: Weights.
    """
    n = Q.shape[0]
    influence = np.array([np.sum(np.abs(Q[i, :])) for i in range(n)])
    return influence / np.max(influence)


def dynamically_update_pdiff(
    pdiff,
    current_iter=1,
    num_qubits=10,
    dynamic_pdiff_policy="hoeffding",
    max_iter=20,
    lowest_threshold=0.001,
    verbose=True,
):
    """Dynamically updates pdiff"""
    if dynamic_pdiff_policy == "hoeffding":
        pdiff = hoeffding_pdiff(
            current_iter,
            num_qubits=num_qubits,
            total_iterations=max_iter,
        )
        if verbose:
            print(f"hoeffding pdiff: {pdiff}")

    elif dynamic_pdiff_policy == "scaledby2":
        pdiff = pdiff / 2**current_iter

    return max(pdiff, lowest_threshold)  # Ensure pdiff doesn't go below the threshold


def bits_of(state, n):
    """Bits of a basis-state index, bit 0 = most significant."""
    return [(int(state) >> (n - 1 - i)) & 1 for i in range(n)]


##########################################################################
#########   Main simulatino function #######################
##########################################################################
STOP_RULES = ("oracle", "max_prob", "none")
_NO_POLICY = {None, "none", "None"}


def vqpm_for_qubo(
    u,  # diagonal of unitary matrix for QUBO
    n,  # number of parameters
    expected_idx=None,  # target_state indx (if known) in the vec to follow its probs
    max_iter=20,
    pdiff=0.01,
    precision=3,
    dynamic_pdiff_policy=None,  #'none' | 'hoeffding' | 'scaledby2' | 'hoeffding_union'
    ###############################################################################
    qubit_weights=None,
    locking="yes",  #'no' | 'only_when_p0_increases'
    stop_rule="oracle",  # 'oracle' | 'max_prob' | 'none'
    shots=None,
    rng=None,
    delta=0.05,
    return_log=False,
    record_trajectory=False,
    fast_forward_fixed_point=False,
    verbose=True,
    energies=None,
):
    """VQPM implementation for QUBO problems.
    Args:
        u (array): Diagonal of unitary matrix U.
        n (int): Number of qubits.
        max_iter (int or str): Maximum iterations, or a budget scaling with n
            such as '10n' (see resolve_max_iter).
        expected_idx (int): Index of expected eigenstate.
        pdiff (float): Probability difference threshold.
        precision (int or None): Rounding precision of the marginals
            (None = no rounding).
        qubit_weights (array): Weights for each qubit.
        dynamic_pdiff_policy: None/'none' (fixed pdiff), 'hoeffding' and
            'scaledby2' (published policies, unchanged), or
            'hoeffding_union' (requires shots): lock only if
            |P0 - P1| > 2 eps with eps = sqrt(ln(2 n T / delta) / (2 M)),
            M = post-selected shots of the iteration, T = max_iter
            (see hoeffding_union_threshold).
        stop_rule:
            'oracle' (default, published): uses expected_idx; stop when the
                target probability >= 0.5, abort when it is 0.
            'max_prob': no oracle; stop when the largest basis-state
                probability >= 0.5 (estimated from the samples if shots is
                set) or when all qubits are locked. expected_idx is then only
                used for reporting.
            'none': never stop early (run max_iter iterations); useful with
                record_trajectory to evaluate several budgets/stop rules from
                one run.
        shots (int or None): if set, the qubit marginals are estimated from
            `shots` circuit runs per iteration: the control qubit is
            post-selected on 0 (probability P0 = sum_x p(x) cos^2(phase_x/2)),
            the M ~ Binomial(shots, P0) surviving samples are drawn from
            p(x) cos^2(phase_x/2) / P0, and the marginal of qubit q is the
            fraction of them with bit q = 1. Rounding and locking then work as
            usual. Default None = exact marginals (published behaviour).
        rng: seed or np.random.Generator for the sampling (never touches the
            global np.random state).
        delta (float): total failure probability for 'hoeffding_union'.
        return_log (bool): if True, return a 7th element `info` (dict):
            lock_log (list of {iteration, qubit, value, p0, p1, margin,
            threshold, matches_optimum}), iterations (iterations actually
            executed), stop_reason, max_iter, shots_used,
            postselected_shots, n_locked, last_change_iter, stall_length,
            stalled (no change of marginals/locks over >= 10 final
            iterations), fixed_point_iter, found_state_exact,
            found_state_marginals, and the trajectory if requested.
        record_trajectory (bool or 'full'): store per-iteration target_prob,
            max_prob, argmax (of the exact output distribution), n_locked and
            changed (whether the prepared input state changed) after the
            iteration, and the prepared marginals P(q=1) if 'full', in
            info['trajectory'].
        fast_forward_fixed_point (bool): in exact mode with fixed pdiff and
            locking='yes', stop iterating once the prepared state no longer
            changes (a fixed point) and fill the remaining iterations with the
            fixed-point values. The returned results are identical to running
            all iterations; only time is saved.
        verbose (bool): print progress (published behaviour).
        energies (array, optional): QUBO values E(x) of all basis states (in
            any units, e.g. the original Q). If given, the expected energy of
            the exact output distribution is recorded per iteration
            (trajectory 'expected_energy') and at the end
            (info['final_expected_energy']).
    Returns:
        tuple: (found_state, max_prob, q_states, p0, num_iter, probs) as in
        the published code, plus `info` if return_log=True.
        Note (published behaviour kept): when the oracle rule aborts because
        the target probability hit 0, num_iter stays max_iter and probs has
        trailing zeros; info['iterations'] holds the true count.
    """
    max_iter = resolve_max_iter(max_iter, n)
    if stop_rule not in STOP_RULES:
        raise ValueError(f"unknown stop_rule {stop_rule!r}; choose from {STOP_RULES}")
    sampled = shots is not None
    if dynamic_pdiff_policy == "hoeffding_union" and not sampled:
        raise ValueError("dynamic_pdiff_policy='hoeffding_union' requires shots")
    if sampled:
        shots = int(shots)
        rng = np.random.default_rng(rng)

    p_min = np.zeros(max_iter)
    N = 2**n

    # the gate is [s1 s2;s2 -s1]
    s2 = 1 / np.sqrt(2)  # ampltidue for hadamard
    # s1 = np.sqrt(1-s2**2)

    in_vec = np.ones(N, dtype=complex) / np.sqrt(N)
    psi1 = np.zeros(N, dtype=complex)
    q_states = {}
    num_iter = max_iter
    p0_prev = 0.0
    psi0_prev = in_vec

    # --- bookkeeping for the optional features (no effect on the published path)
    lock_log = [] if return_log else None
    info = {
        "max_iter": max_iter,
        "stop_reason": "max_iter",
        "iterations": max_iter,
        "shots_used": 0,
        "postselected_shots": 0,
        "fast_forwarded_from": None,
    }
    traj = None
    if record_trajectory:
        traj = {
            "target_prob": [],
            "max_prob": [],
            "argmax": [],
            "n_locked": [],
            "changed": [],
        }
        if record_trajectory == "full":
            traj["marginals"] = []
        if energies is not None:
            traj["expected_energy"] = []
    can_fast_forward = (
        fast_forward_fixed_point
        and not sampled
        and dynamic_pdiff_policy in _NO_POLICY
        and locking == "yes"
    )
    last_change_iter = -1
    last_prepare_iter = -1
    last_marginals = None
    if sampled:
        w_accept = np.abs((1 + u) / 2) ** 2  # cos^2(phase/2): P(control = 0 | x)
        idx = np.arange(N)
        ones_masks = [(idx & (1 << (n - q))) != 0 for q in range(1, n + 1)]
        marg_prev = None
        last_counts = None
        emp_max_prob = None

    for j in range(max_iter):

        # apply first hadamard (we only simulate ket{0} part:
        # the first half of state vector)
        psi0 = in_vec * s2  # the other half zero: \ket0\otimes invec

        psi1[:] = u * psi0  # Apply CU

        # Apply second Hadamard
        psi0_final = (psi0 + psi1) * s2  # I+U
        p0 = np.linalg.norm(psi0_final)
        psi0_final /= p0

        marginals = None
        if sampled:
            joint = np.abs(in_vec) ** 2 * w_accept
            P0 = float(joint.sum())
            m_post = int(rng.binomial(shots, min(max(P0, 0.0), 1.0)))
            info["shots_used"] += shots
            info["postselected_shots"] += m_post
            if m_post > 0:
                counts = rng.multinomial(m_post, joint / joint.sum())
                marginals = np.array([counts[m].sum() / m_post for m in ones_masks])
                last_marginals = marginals
                last_counts = counts
                emp_max_prob = counts.max() / m_post
            else:
                emp_max_prob = None
            p0_used = np.sqrt(m_post / shots)  # estimated amplitude norm
        else:
            p0_used = p0

        ##########################################################
        # Calculate probabilities for the expected state
        if expected_idx is not None:
            p_min[j] = np.abs(psi0_final[expected_idx]) ** 2
        else:
            p_min[j] = np.max(np.abs(psi0_final) ** 2)
        if traj is not None:
            probs_j = np.abs(psi0_final) ** 2
            traj["target_prob"].append(float(p_min[j]))
            traj["max_prob"].append(float(probs_j.max()))
            traj["argmax"].append(int(np.argmax(probs_j)))
            if energies is not None:
                traj["expected_energy"].append(float(np.dot(probs_j, energies)))
        # print(f"p_min[{j}]: {p_min[j]}")
        stop = False
        if stop_rule == "oracle":
            if p_min[j] >= 0.5:
                num_iter = j + 1
                info["stop_reason"] = "target_prob"
                stop = True
            elif p_min[j] == 0:
                if verbose:
                    print(
                        "\n\nUnsuccesfull run: Optimization converged to a suboptimal value!\n\n"
                    )
                info["stop_reason"] = "target_lost"
                stop = True
        elif stop_rule == "max_prob":
            if sampled:
                maxp = emp_max_prob if emp_max_prob is not None else 0.0
            else:
                maxp = np.max(np.abs(psi0_final) ** 2)
            if maxp >= 0.5:
                num_iter = j + 1
                info["stop_reason"] = "max_prob"
                stop = True
        if stop:
            info["iterations"] = j + 1
            if traj is not None:
                traj["n_locked"].append(len(q_states))
                traj["changed"].append(False)
            break

        ##########################################################
        ## NEW pdiff from Hoeffding
        if dynamic_pdiff_policy == "hoeffding_union":
            pdiff = hoeffding_union_threshold(
                n, max_iter, delta, m_post, precision=precision
            )
        elif dynamic_pdiff_policy not in _NO_POLICY:
            pdiff = dynamically_update_pdiff(
                pdiff,
                current_iter=j,
                num_qubits=n,
                dynamic_pdiff_policy=dynamic_pdiff_policy,
                max_iter=max_iter,
                verbose=verbose,
            )

        if verbose:
            print(f"\niteration-{j} pdiff: {pdiff}")

        prev_in_vec = in_vec
        common = dict(
            qubit_weights=qubit_weights,
            lock_log=lock_log,
            iteration=j,
            optimum=expected_idx,
            verbose=verbose,
        )
        no_samples = sampled and marginals is None  # M = 0: nothing learned

        ## LOCKING with DIFFERENT POLICIES
        if no_samples:
            pass  # keep the current input state
        elif locking == "no":  # No locking
            in_vec, q_states = prepare_new_state(
                psi0_final,
                n,
                q_states,
                pdiff=pdiff,
                precision=precision,
                locking="no",  # THIS time DO NOT LOCK
                marginals=marginals,
                **common,
            )
        elif locking == "yes":
            in_vec, q_states = prepare_new_state(
                psi0_final,
                n,
                q_states,
                pdiff=pdiff,
                precision=precision,
                locking="yes",
                marginals=marginals,
                **common,
            )
            # update previous state
            # previous state and the probability of the previous state
            psi0_prev = psi0_final
            p0_prev = p0_used
        # Check if the probability of 0 in the1st qubit is increasing
        elif locking == "only_when_p0_increases":
            if (precision is None and p0_used >= p0_prev) or (
                precision is not None
                and np.round(p0_used, precision) >= np.round(p0_prev, precision)
            ):
                if verbose:
                    print(f"iteration: {j} p0_prev: {p0_prev} p0: {p0_used}")
                # Restore previous state and the probability
                psi0_prev = psi0_final
                p0_prev = p0_used
                if sampled:
                    marg_prev = marginals

                in_vec, q_states = prepare_new_state(
                    psi0_final,
                    n,
                    q_states,
                    pdiff,
                    precision,
                    locking="yes",
                    marginals=marginals,
                    **common,
                )
            else:
                if verbose:
                    print(f"no locking p0 remains the same p0: {p0_used}")
                psi0_final = psi0_prev
                p0 = p0_prev
                # If the probability of 0 in the 1st qubit is not increasing,
                #  do not lock
                in_vec, q_states = prepare_new_state(
                    psi0_prev,  # GO back to the previous state
                    n,
                    q_states,
                    pdiff,
                    precision,
                    locking="no",  # THIS time DO NOT LOCK
                    marginals=marg_prev if sampled else None,
                    **common,
                )

        else:  # No policy found
            if verbose:
                print(f"Unknown locking policy: {locking}. Using no locking.")
            in_vec, q_states = prepare_new_state(
                psi0_final,  # current output state (was the undefined `psi_final`)
                n,
                q_states,
                pdiff,
                precision,
                locking="no",
                marginals=marginals,
                **common,
            )

        last_prepare_iter = j
        changed = not np.array_equal(in_vec, prev_in_vec)
        if changed:
            last_change_iter = j
        if traj is not None:
            traj["n_locked"].append(len(q_states))
            traj["changed"].append(bool(changed))
            if record_trajectory == "full":
                traj["marginals"].append(
                    [float(calculate_prob(in_vec, q)[1]) for q in range(1, n + 1)]
                )

        if stop_rule == "max_prob" and len(q_states) == n:
            # all qubits locked: the output is the locked basis state
            num_iter = j + 1
            info["stop_reason"] = "all_locked"
            info["iterations"] = j + 1
            psi0_final = np.asarray(in_vec, dtype=complex).copy()
            p_min[j] = (
                np.abs(psi0_final[expected_idx]) ** 2
                if expected_idx is not None
                else 1.0
            )
            break

        if can_fast_forward and not changed and j + 1 < max_iter:
            # fixed point: every later iteration repeats this one exactly
            p_min[j + 1 :] = p_min[j]
            info["fast_forwarded_from"] = j + 1
            last_prepare_iter = max_iter - 1
            if traj is not None:
                rest = max_iter - j - 1
                for key in traj:
                    traj[key].extend([traj[key][-1]] * rest)
            break

    final_probs = np.abs(psi0_final) ** 2
    max_prob = np.max(final_probs)
    found_state = np.argmax(final_probs)

    if not return_log:
        return found_state, max_prob, q_states, p0, num_iter, p_min[:num_iter]

    # ------------------------------------------------------------------ info
    bits = []
    for q in range(1, n + 1):
        if q in q_states:
            bits.append(int(q_states[q][1] == 1.0))
        elif sampled and last_marginals is not None:
            bits.append(int(last_marginals[q - 1] > 0.5))
        else:
            bits.append(int(calculate_prob(psi0_final, q)[1] > 0.5))
    info["found_state_exact"] = int(found_state)
    info["found_state_marginals"] = int("".join(map(str, bits)), 2)
    if sampled:
        info["found_state_sampled"] = (
            int(np.argmax(last_counts)) if last_counts is not None else None
        )
        if info["stop_reason"] != "all_locked" and last_counts is not None:
            # report what an experiment would see: the most frequent sample
            found_state = int(np.argmax(last_counts))
    info["lock_log"] = lock_log
    info["n_locked"] = len(q_states)
    info["last_change_iter"] = int(last_change_iter)
    # number of consecutive final state-preparation steps that changed nothing
    stall_length = last_prepare_iter - last_change_iter
    info["stall_length"] = int(stall_length)
    info["stalled"] = bool(stall_length >= 10)
    info["fixed_point_iter"] = (
        int(last_change_iter + 1) if stall_length >= 1 else None
    )
    if expected_idx is not None:
        info["final_target_prob"] = float(np.abs(psi0_final[expected_idx]) ** 2)
    if energies is not None:
        info["final_expected_energy"] = float(np.dot(final_probs, energies))
    if traj is not None:
        info["trajectory"] = traj
    return found_state, max_prob, q_states, p0, num_iter, p_min[:num_iter], info


if __name__ == "__main__":
    # Make sure to set seed for reproducibility.
    # np.random.seed(42)  # For reproducibility
    n = 10
    # Generate QUBO problem
    Q = random_qubo(n)

    utest = test_u_for_qubo(n, Q)  # Get actual QUBO values

    qubit_weights = compute_qubit_weights(Q)
    # Prepare phase-adjusted unitary
    Q_adj, u, phases, target_state = adjust_unitary_phase(n, Q)

    # Run VQPM algorithm
    result_state, max_prob, q_states, _, iters, probs = vqpm_for_qubo(
        u,
        n,
        expected_idx=target_state,  # target_state indx (if known) in the vec to follow its probs
        max_iter=30,  # best 30
        pdiff=0.01,  # best 0.01
        precision=3,  # best 3
        dynamic_pdiff_policy=None,  #'none' | 'hoeffding' | 'scaledby2',
        ###############################################################################
        qubit_weights=None,
        locking="yes",  # "yes",  #'no' | 'only_when_p0_increases'
    )

    # Convert states to binary format
    def state_to_bin(state, n):
        return bin(state)[2:].zfill(n)

    target_bin = state_to_bin(target_state, n)
    result_bin = state_to_bin(result_state, n)

    state_diff = target_state ^ result_state  # xor
    diff_bits = state_to_bin(state_diff, n)
    num_diff_bits = diff_bits.count("1")

    # Calculate success metrics
    success_prob = np.abs(u[target_state]) ** 2 / (2**n)  # Theoretical maximum

    # Print detailed comparison
    print("================= Problem Summary =================")
    print(f"QUBO Matrix (first 5x5):")
    print(Q[:5, :5])
    print("\n=============== Algorithm Results =================")
    print("Expected state: %10ld %30s" % (target_state, target_bin))
    print("Found state   : %10ld %30s" % (result_state, result_bin))
    print("Diff bits     : %10d %30s\n" % (num_diff_bits, diff_bits))

    print(f"\nQUBO Values:")
    print(f"Expected: {utest[target_state]:.4f} | Found: {utest[result_state]:.4f}")
    print(f"\nProbabilities in the final state:")
    print(f"Max probability:       {max_prob:.4f}")
    print(f"Expected state prob:   {probs[-1]:.4f}")
    print(f"for comparison: 1/2^n:  {1/(2**n):.4f}  1/n:{1/(n):.4f}")

    # -------------------------------------------------------------------------
    # Matplotlib configuration for consistent styling.
    # -------------------------------------------------------------------------
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "figure.titlesize": 18,
            "figure.figsize": (20, 5),
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "lines.linewidth": 2,
            "lines.markersize": 6,
        }
    )

    # Energy landscape visualization
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(utest, "o", markersize=3)
    plt.plot(target_state, utest[target_state], "ro", label="Target")
    plt.plot(result_state, utest[result_state], "kx", label="Found")
    plt.xlabel("State Index")
    plt.ylabel("QUBO Value")
    plt.title("Energy Landscape")
    plt.legend()

    # Convergence plot
    plt.subplot(1, 2, 2)
    plt.plot(range(iters), probs, "o-", label="Algorithm")
    plt.axhline(1 / (2**n), color="r", linestyle="--", label="Random guess (1/2ⁿ)")
    plt.axhline(1 / (n), color="b", linestyle="-.", label="Polynomial line (1/n)")

    plt.xlabel("Iteration")
    plt.ylabel("Success Probability")
    plt.title("Convergence Progress")
    plt.legend()

    plt.tight_layout()
    plt.savefig("vqpm_convergence.png", dpi=100)
    plt.savefig("vqpm_convergence.pdf", dpi=100)
    plt.show()
