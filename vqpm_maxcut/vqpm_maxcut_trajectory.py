"""
QPM converges the eigenvector of I-U with the maximum magnitude
that is max |1-e^{iλ}|, where λ is the minimum eigenvalue of H.
The file includes the implementations of variational quantum power method (vqpm) applied to random QUBO,
vqpmForQUBO: it finds the minimum eigenphase of U.

@author: adaskin, updated and modified with the help of DeepSeek assistant
"""

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

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
                if b[i] != b[j] :
                    total += Q[i][j]
        test_vals[k] = total/2 # to match the QUBO definition for Max-Cut
    return test_vals

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
   
        u[k] = np.exp(1j * (phase))  

    return u

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
    target_state = np.amax(phases)
    all_targets = np.argwhere(np.isclose(phases, target_state, atol=1e-15)).flatten()
    if len(all_targets) > 1:
        print(f"Warning: Multiple states with maximum phase found: {all_targets}. Choosing the first one.")
    return Q_scaled, u, phases, all_targets




def random_qubo(n):
    """Generates random QUBO matrix.
    Args:
        n (int): Matrix dimension.
    Returns:
        matrix: Symmetric Q matrix.
    """
    Q = np.random.randn(n, n)
    return Q + Q.T




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
    out_vec, n, q_states, pdiff=0.01, 
    precision=3, qubit_weights=None, locking="yes",
    break_symmetry_by_locking_qubit1=True
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
        elif q == 1 and break_symmetry_by_locking_qubit1:  # For the first qubit, break symmetry to avoid uniform superposition
            state_q = np.array([0.0, 1.0]) #break the symmetry for the first qubit to avoid uniform superposition
            q_states[q] = state_q
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




def compute_sos_metrics(state, edges, n):
    """Computes moment matrix Y and tracks SoS metrics from the live state vector."""
    N = 2**n
    Y = np.zeros((n + 1, n + 1))
    Y[0, 0] = 1.0

    spins = np.zeros((N, n))
    for k in range(N):
        b = bin(k)[2:].zfill(n)
        spins[k] = [1 if bit == "0" else -1 for bit in b]

    probs = np.abs(state) ** 2

    # Fill single-qubit expectations <Z_i>
    for i in range(n):
        Y[0, i + 1] = np.sum(probs * spins[:, i])
        Y[i + 1, 0] = Y[0, i + 1]

    # Fill two-qubit expectations <Z_i Z_j>
    for i in range(n):
        for j in range(n):
            if i == j:
                Y[i + 1, j + 1] = 1.0
            else:
                Y[i + 1, j + 1] = np.sum(probs * spins[:, i] * spins[:, j])

    E_rho = 0.0
    F_GW = 0.0
    for i, j in edges:
        y_ij = Y[i + 1, j + 1]
        # Invert minimization mapping to match expected positive Cut orientation
        E_rho += (1.0 - y_ij) / 2.0
        F_GW += np.arccos(np.clip(y_ij, -1.0, 1.0)) / np.pi

    return E_rho, F_GW



def gw_sdp_upper_bound(n, edges, weights):
    """
    Returns the optimal value of the GW SDP relaxation for Max-Cut.
    edges: list of (i,j) tuples
    weights: list of corresponding edge weights (or scalar if uniform)
    """
    Y = cp.Variable((n, n), symmetric=True)
    constraints = [Y >> 0, cp.diag(Y) == 1]
    objective = cp.Maximize(
        sum(weights[k] * (1 - Y[edges[k][0], edges[k][1]]) / 2
            for k in range(len(edges)))
    )
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)   # SCS is robust for small/medium n
    return prob.value
##########################################################################
#########   Main simulation function #######################
### Finds the maximum phase, for negative phases it converges to zero
# change  psi0_final = (psi0 - psi1) * s2 into  
# psi0_final = (psi0 +psi1) * s2 for minumum phase convergence
##########################################################################


def vqpm_for_qubo(
    u,  # diagonal of unitary matrix for QUBO
    n,  # number of parameters
    edges,  # graph edges added for live SoS monitoring
    expected_idx=None,  # target_state indx (if known) in the vec to follow its probs
    max_iter=20,
    pdiff=0.01,
    precision=3,
    dynamic_pdiff_policy=None,  #'none' | 'hoeffding' | 'scaledby2',
    qubit_weights=None,
    locking="yes",  #'no' | 'only_when_p0_increases'
    break_symmetry_by_locking_qubit1=False
):
    if expected_idx is not None and isinstance(expected_idx, int):
        p_expected = np.zeros(max_iter)
    elif isinstance(expected_idx, list) or isinstance(expected_idx, np.ndarray):
        p_expected = np.zeros((max_iter, len(expected_idx)))  # track all target states
    p_max = np.zeros(max_iter)
    N = 2**n
    s2 = 1 / np.sqrt(2)

    in_vec = np.ones(N, dtype=complex) / np.sqrt(N)
    psi1 = np.zeros(N, dtype=complex)
    q_states = {}
    num_iter = max_iter
    p0_prev = 0.0
    psi0_prev = in_vec

    # Dynamic arrays to follow continuous semidefinite relaxations
    history_E_rho = []
    history_F_GW = []
    history_floor = []
    alpha_GW = 0.87856

    for j in range(max_iter):
        psi0 = in_vec * s2
        psi1[:] = -u * psi0 #change -u to u for minimum phase convergence
        psi0_final = (psi0 + psi1) * s2
        p0 = np.linalg.norm(psi0_final)
        psi0_final /= p0
   
        # Continuous tracking of real-time geometric parameters
        E_rho, F_GW = compute_sos_metrics(psi0_final, edges, n)
        history_E_rho.append(E_rho)
        history_F_GW.append(F_GW)
        history_floor.append(alpha_GW * E_rho)

        if expected_idx is not None:
            if isinstance(expected_idx, int):
                p_expected[j] = np.abs(psi0_final[expected_idx]) ** 2
            else:
                for k, idx in enumerate(expected_idx):
                    p_expected[j, k] = np.abs(psi0_final[idx]) ** 2

        p_max[j] = np.max(np.abs(psi0_final) ** 2)

        if np.max(p_expected[j]) >= 0.99:  # Threshold adjusted to collect a complete trail
            num_iter = j + 1
            break


        if locking == "no":
            in_vec, q_states = prepare_new_state(
                psi0_final,
                n,
                q_states,
                pdiff=pdiff,
                precision=precision,
                qubit_weights=qubit_weights,
                locking="no",
                break_symmetry_by_locking_qubit1=break_symmetry_by_locking_qubit1
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
                break_symmetry_by_locking_qubit1=break_symmetry_by_locking_qubit1
            )
 

    final_probs = np.abs(psi0_final) ** 2
    max_prob = np.max(final_probs)
    found_state = np.argmax(final_probs)
    results = {
        "found_state": found_state,
        "max_prob": max_prob,
        "q_states": q_states,
        "p0": p0,
        "num_iter": num_iter,
        "p_expected": p_expected[:num_iter],
        "p_max": p_max[:num_iter],
        "history_E_rho": history_E_rho,
        "history_F_GW": history_F_GW,
        "history_floor": history_floor,
    }
    return results
# ------------------------------------------------------------
#  Random Max‑Cut instance generator
# ------------------------------------------------------------

def random_maxcut_qubo(n, p=0.5, weight_range=(1,1), seed=None):
    """
    Generate QUBO matrix for Max‑Cut on an Erdős–Rényi graph G(n,p).
    Edge weights are drawn uniformly from weight_range (default 1).
    Returns: Q (n×n upper‑triangular), edges list, maxcut value (brute‑force).
    """
    if seed is not None:
        np.random.seed(seed)
    edges = []
    Q = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            if np.random.rand() < p:
                w = np.random.uniform(weight_range[0], weight_range[1])
                edges.append((i, j))
                # Max‑Cut QUBO: Q[i][i] += w , Q[j][j] += w , Q[i][j] -= 2*w
                Q[i, i] += w
                Q[j, j] += w
                Q[i, j] -=  2*w
    

    # brute‑force MaxCut (only for n <= 20)
    maxcut = 0
    if n <= 20:
        N = 2**n
        for mask in range(N):
            cut = 0
            for i, j in edges:
                b_i = (mask >> i) & 1
                b_j = (mask >> j) & 1
                if b_i != b_j:
                    cut += 1   # assumes unit weight, adapt if needed
                
            maxcut = max(maxcut, cut)
    else:
        maxcut = None   # not computed
    return Q, edges, maxcut



# if __name__ == "__main__":
# n = 4
# edges = [(0,1), (1,2), (2,3), (3,0)]
# Q = np.zeros((n, n))
# for i, j in edges:
#     Q[i, j] -= 2.0   # off‑diagonal
#     Q[j, j] += 1.0
#     Q[i, i] += 1.0   # diagonal for linear term

n = 14

Q, edges, exact_maxcut = random_maxcut_qubo(n, p=0.5, weight_range=(1,1), 
                                            seed=None)
# print("QUBO Matrix:\n", Q[:5,:5])
print("Edges:\n", edges)
print("Exact MaxCut:\n", exact_maxcut)
# After building Q and edges
weights = [1.0] * len(edges)   # or extract from graph if weighted
sdp_opt = gw_sdp_upper_bound(n, edges, weights)
print(f"GW SDP upper bound: {sdp_opt:.4f}")

utest = test_u_for_qubo(n, Q)
Q_adj, u, phases, all_targets = adjust_unitary_phase(n, Q)
target_state = all_targets[-1]  # Choose the first target state
# Run modified VQPM tracking pipeline
# After obtaining all_targets
target_states = all_targets   # list of degenerate ground states
results = vqpm_for_qubo(
    u, n, edges=edges,
    expected_idx=target_states,   # pass the whole list
    max_iter=300, pdiff=0.2, precision=3,
    qubit_weights=None, locking="yes",
    break_symmetry_by_locking_qubit1=True
)

def state_to_bin(state, n):
    return bin(state)[2:].zfill(n)
found_state = results["found_state"]
target_states = all_targets   # list of degenerate ground states

# Convert target states to binary strings
target_bins = [state_to_bin(s, n) for s in target_states]
result_bin = state_to_bin(found_state, n)

# Check if found state is one of the targets
is_optimal = found_state in target_states

# Compute minimum Hamming distance to any target (optional)
min_hamming = min(bin(found_state ^ s).count('1') for s in target_states)

print("================= Problem Summary =================")
print(f"QUBO Matrix (first 5x5):")
print(Q[:5, :5])
print("\n=============== Algorithm Results =================")
print("Expected states (degenerate ground states):")
for s, b in zip(target_states, target_bins):
    print(f"  {s:10d} {b}")
print(f"Found state   : {found_state:10d} {result_bin}")
print(f"Optimal?      : {is_optimal}")
print(f"Min Hamming distance to any target: {min_hamming}")
print(f"\nQUBO Values:")
print(f"Expected (max cut): {utest[target_states[0]]:.4f} (all targets have same value)")
print(f"Found state value: {utest[found_state]:.4f}")
print(f"\nProbabilities in the final state:")
print(f"Max probability (any state): {results['max_prob']:.4f}")
print(f"Total target probability (sum over all ground states):", results['p_expected'][-1])

print(f"Uniform random baseline 1/2^n: {1/(2**n):.4f}")
print(f"Polynomial baseline 1/n: {1/n:.4f}")

plt.rcParams.update(
    {
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "figure.figsize": (18, 5),
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "lines.linewidth": 2,
        "lines.markersize": 6
    }
)

# 3-Panel Unified Figure Layout
plt.figure(figsize=(20, 10))

# Panel 1: Original Cost Value Landscape
plt.subplot(2, 2, 1)
plt.plot(utest, "o", color="#1f77b4", markersize=5, label="Basis Configurations")
plt.plot(all_targets, utest[all_targets], "ro", markersize=10, label="Target")
plt.plot(found_state, utest[found_state], "kx", markersize=12, markeredgewidth=2, label="Found")

plt.xlabel("State Index", labelpad=10)
plt.ylabel("Energy Value ", labelpad=10)
energy_diff= np.abs(utest[target_state] - utest[found_state])
relative_error = energy_diff / np.abs(utest[target_state]) if utest[target_state] != 0 else np.nan
plt.title(f"QUBO Energy Landscape ($\delta$: {relative_error:.4f})", 
          pad=15)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()

# # Panel 2: Classical Target Success Convergence
plt.subplot(2, 2, 2)
plt.plot(range(1, len(results["p_expected"]) + 1), 
         np.sum(results["p_expected"], axis=1), 
"o-", color="#e61919", linewidth=2.5, label="Total Target Probability")
plt.plot(range(1, len(results["p_max"]) + 1), results["p_max"], "o-", 
         color="#043D11", linewidth=2.5, label="Max Probability")

plt.axhline(1 / (2**n), color="b", linestyle="-", label="Uniform Random $1/2^n$")
plt.axhline(1 / n, color="b", linestyle="-.", label="Polynomial Floor $1/n$")
plt.xlabel("Iteration", labelpad=10)
plt.ylabel("Probability", labelpad=10)
plt.title("Convergence Progress", pad=15)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper left")


# Panel 3: Live Geometric Moment Relaxations (SoS / GW-Bound)
plt.subplot(2, 2, 3)
steps = range(1, len(results["history_E_rho"]) + 1)
plt.plot(steps, results["history_F_GW"], "o-", color="#e377c2", linewidth=2.5, label=r"Rounded Bound $F_{\text{vqpm}}$")
plt.plot(steps, results["history_E_rho"], "s--", color="#bcbd22", linewidth=2.0, label=r"Quantum Energy $E_{\text{vqpm}}$")
plt.plot(steps, results["history_floor"], "k:", linewidth=2.0, label=r"Analytical Floor $\alpha_{\text{GW}} \cdot E_{\text{vqpm}}$")
plt.xlabel("Iteration", labelpad=10)
plt.ylabel("Cut Size Value", labelpad=10)
plt.title("Live Moment Relaxation Trajectory", pad=15)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="lower right")

alpha_GW = 0.87856

weights = [1.0] * len(edges)   # or extract from graph if weighted
sdp_opt = gw_sdp_upper_bound(n, edges, weights)
print(f"GW SDP upper bound: {sdp_opt:.4f}")
ratio_history = [f / sdp_opt for f in results["history_F_GW"]]
plt.subplot(2, 2, 4)
plt.plot(range(1, len(ratio_history)+1), ratio_history, 'o-', color='purple', linewidth=2.5)
plt.axhline(alpha_GW, color='gray', linestyle='--', label=r'$\alpha_{\text{GW}} = 0.87856$')
plt.xlabel('Iteration', labelpad=10)
plt.ylabel(r'Certified Ratio $F_{\text{GW}} / \text{SDP}^*$', labelpad=10)
plt.title('Lower bound on approximation ratio', pad=15)
plt.grid(alpha=0.3)
plt.legend()


plt.tight_layout()
plt.savefig("vqpm_extended_sos_metrics.png", dpi=300)
plt.savefig("vqpm_extended_sos_metrics.pdf", dpi=300)
plt.show()