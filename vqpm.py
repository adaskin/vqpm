"""
QPM converges the eigenvector of I+U with the maximum magnitude
that is max |1+e^{iλ}|, where λ is the minimum eigenvalue of H.
The file includes the implementations of variational quantum power method (vqpm) applied to random QUBO,
vqpmForQUBO: it finds the minimum eigenphase of U.

@author: adaskin, updated and modified with the help of DeepSeek/copilot assistant
"""

import numpy as np
import matplotlib.pyplot as plt


def unitary_for_qubo(n, Q):
    """Creates diagonal unitary matrix for QUBO.
    Args:
        n (int): Number of qubits.
        Q (matrix): QUBO matrix.
    Returns:
        array: Diagonal of unitary matrix.
    """
    num_terms = 2**n
    u = np.ones(num_terms, dtype=complex)
    for k in range(num_terms):
        b = bin(k)[2:].zfill(n)
        phase = 0.0
        for i in range(n):
            for j in range(i, n):
                if b[i] == "1" and b[j] == "1":
                    phase += Q[i][j]
        u[k] = np.exp(1j * phase)
    return u


def test_u_for_qubo(n, Q):
    """Validates unitary construction.
    Args:
        n (int): Number of qubits.
        Q (matrix): QUBO matrix.
    Returns:
        array: Test values.
    """
    num_terms = 2**n
    test_vals = np.zeros(num_terms)
    for k in range(num_terms):
        b = bin(k)[2:].zfill(n)
        total = 0.0
        for i in range(n):
            for j in range(i, n):
                if b[i] == "1" and b[j] == "1":
                    total += Q[i][j]
        test_vals[k] = total
    return test_vals


def random_qubo(n):
    """Generates random QUBO matrix.
    Args:
        n (int): Matrix dimension.
    Returns:
        matrix: Symmetric Q matrix.
    """
    Q = np.random.randn(n, n)
    return Q + Q.T


def adjust_unitary_phase(n, Q):
    """Adjusts QUBO matrix for phase calculations.
    Args:
        n (int): Number of qubits.
        Q (matrix): QUBO matrix.
    Returns:
        tuple: Scaled Q, unitary, phases, and expected state.
    """
    max_q = np.sum(np.abs(np.triu(Q)))
    Q_scaled = Q / max_q * np.pi / 4  # in [-pi/4, pi/4]
    u = unitary_for_qubo(n, Q_scaled)
    u *= np.exp(1j * np.pi / 4)  # Shift phases to positive
    phases = np.angle(u)  # Get eigenphases
    expected_state = np.argmin(phases)
    return Q_scaled, u, phases, expected_state


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
    locking="yes"
):
    """Generates a new state based on qubit probabilities.
    Args:
        out_vec (array): Quantum state vector.
        n (int): Number of qubits.
        q_states (dict): Precomputed qubit states.
        pdiff (float): Threshold for collapsing probabilities.
        precision (int): Rounding precision.
        qubit_weights (array): Weights for each qubit.
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
            state_q = calculate_prob(out_vec, q)
            state_q = np.round(state_q, precision)

            # Scale pdiff by qubit influence (higher weight = stricter threshold)
            if qubit_weights is not None:
                pdiff_scaled = pdiff * qubit_weights[q - 1]  # weights are 0-indexed
                print(f"qubit{q} pdiff_scaled {pdiff_scaled}", state_q)
            else:
                pdiff_scaled = pdiff  # weights are 0-indexed

            # locking qubits at Most 1 for each iteration
            if state_q[0] > state_q[1] + pdiff_scaled:
                state_q = np.array([1.0, 0.0])
                q_states[q] = state_q
            elif state_q[1] > state_q[0] + pdiff_scaled:
                state_q = np.array([0.0, 1.0])
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
    Dynamically adjusts pdiff using Hoeffding's inequality.

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
    delta_i = delta_total / remaining_iters  # Union bound

    # Hoeffding bound: ε = sqrt(ln(2/δ)/(2M))
    pdiff =  np.sqrt(np.log(2 / delta_i) / (20*num_qubits*shots_per_iter))

    return max(pdiff, lowest_threshold)


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
):
    """Dynamically updates pdiff"""
    if dynamic_pdiff_policy == "hoeffding":
        pdiff = hoeffding_pdiff(
            current_iter,
            num_qubits=num_qubits,
            total_iterations=max_iter,
        )
        print(f"hoeffding pdiff: {pdiff}")

    elif dynamic_pdiff_policy == "scaledby2":
        pdiff = pdiff / 2**current_iter
    

    return max(pdiff,lowest_threshold)  # Ensure pdiff doesn't go below the threshold




##########################################################################
#########   Main simulatino function #######################
##########################################################################
def vqpm_for_qubo(
    u, # diagonal of unitary matrix for QUBO
    n,  #number of parameters
    expected_idx=None,  # target_state indx (if known) in the vec to follow its probs
    max_iter=20,
    pdiff=0.01,
    precision=3,
    dynamic_pdiff_policy=None,  #'none' | 'hoeffding' | 'scaledby2',
    ###############################################################################
    qubit_weights=None,
    locking="yes",  #'no' | 'only_when_p0_increases'
):
    """VQPM implementation for QUBO problems.
    Args:
        u (array): Diagonal of unitary matrix U.
        n (int): Number of qubits.
        max_iter (int): Maximum iterations.
        expected_idx (int): Index of expected eigenstate.
        pdiff (float): Probability difference threshold.
        precision (int): Rounding precision.
        qubit_weights (array): Weights for each qubit.
    Returns:
        tuple: Results including found state, probabilities, and iterations.
    """

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

    for j in range(max_iter):

        # apply first hadamard (we only simulate ket{0} part:
        # the first half of state vector)
        psi0 = in_vec * s2  # the other half zero: \ket0\otimes invec

        psi1[:] = u * psi0  # Apply CU

        # Apply second Hadamard
        psi0_final = (psi0 + psi1) * s2  # I+U
        p0 = np.linalg.norm(psi0_final)
        psi0_final /= p0

        ##########################################################
        # Calculate probabilities for the expected state
        if expected_idx is not None:
            p_min[j] = np.abs(psi0_final[expected_idx]) ** 2
        else:
            p_min[j] = np.max(np.abs(psi0_final) ** 2)
        # print(f"p_min[{j}]: {p_min[j]}")
        if p_min[j] >= 0.5:
            num_iter = j + 1
            break
        elif p_min[j] == 0:
            print(
                "\n\nUnsuccesfull run: Optimization converged to a suboptimal value!\n\n"
            )
            break

        ##########################################################
        ## NEW pdiff from Hoeffding
        if dynamic_pdiff_policy not in {None, "none", "None"}:
            pdiff = dynamically_update_pdiff(
                        pdiff,
                        current_iter=j,
                        num_qubits=n,
                        dynamic_pdiff_policy=dynamic_pdiff_policy,
                        max_iter=max_iter,
                    )


        print(f"\niteration-{j} pdiff: {pdiff}")

        ## LOCKING with DIFFERENT POLICIES
        if locking == "no":  # No locking
            in_vec, q_states = prepare_new_state(
                psi0_final,
                n,
                q_states,
                pdiff=pdiff,
                precision=precision,
                qubit_weights=qubit_weights,
                locking="no",  # THIS time DO NOT LOCK
            )
        elif locking == "yes":
            in_vec, q_states = prepare_new_state(
                psi0_final,
                n,
                q_states,
                pdiff=pdiff,
                precision=precision,
                qubit_weights=qubit_weights,
                locking="yes",
            )
            # update previous state
            # previous state and the probability of the previous state
            psi0_prev = psi0_final
            p0_prev = p0
        # Check if the probability of 0 in the1st qubit is increasing
        elif locking == "only_when_p0_increases":
            if np.round(p0, precision) >= np.round(p0_prev, precision):
                print(f"iteration: {j} p0_prev: {p0_prev} p0: {p0}")
                # Restore previous state and the probability
                psi0_prev = psi0_final
                p0_prev = p0

                in_vec, q_states = prepare_new_state(
                    psi0_final,
                    n,
                    q_states,
                    pdiff,
                    precision,
                    qubit_weights=qubit_weights,
                    locking="yes",
                )
            else:
                print(f"no locking p0 remains the same p0: {p0}")
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
                    qubit_weights,
                    locking="no",  # THIS time DO NOT LOCK
                )

        else:  # No policy found
            print(f"Unknown locking policy: {locking}. Using no locking.")
            in_vec, q_states = prepare_new_state(
                psi_final,  # GO back to the previous state
                n,
                q_states,
                pdiff,
                precision,
                qubit_weights,
                locking="no",
            )

    final_probs = np.abs(psi0_final) ** 2
    max_prob = np.max(final_probs)
    found_state = np.argmax(final_probs)

    return found_state, max_prob, q_states, p0, num_iter, p_min[:num_iter]


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
        max_iter=30, #best 30
        pdiff=0.01, #best 0.01
        precision=3, #best 3
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
