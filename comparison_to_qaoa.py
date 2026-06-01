import numpy as np
import matplotlib.pyplot as plt

from vqpm import (
    random_qubo,
    adjust_unitary_phase,
    unitary_for_qubo,
    test_u_for_qubo,
    compute_qubit_weights,
    vqpm_for_qubo,
    compute_qubit_weights,
)
from run_multiple import run_multiple_trials, run_vqpm_trial

from scipy.linalg import expm
from scipy.optimize import minimize

import numpy as np
import matplotlib.pyplot as plt

# class QAOA:
#     def __init__(self, Q):
#         self.n = Q.shape[0]  # Number of qubits (assumed Q is n x n for an n-qubit QUBO)
#         self.N = 2**self.n  # Hilbert space dimension
#         self.Hp = self.create_problem_hamiltonian(Q)
#         # self.Hm = self.create_mixer_hamiltonian()
#     def create_problem_hamiltonian(self, Q):
#         """Create a 1D diagonal Hamiltonian array from the QUBO matrix."""
#         H = np.zeros(self.N)
#         for k in range(self.N):
#             b = bin(k)[2:].zfill(self.n)
#             energy = 0.0
#             for i in range(self.n):
#                 for j in range(i, self.n):
#                     if b[i] == "1" and b[j] == "1":
#                         energy += Q[i, j]
#             H[k] = energy
#         return H  # REMOVE np.diag() here! Keep it as a 1D vector.

#     def ansatz_layer(self, state, beta, gamma):
#         # Apply problem Hamiltonian using clean, fast element-wise vector math
#         state = state * np.exp(-1j * gamma * self.Hp)
#         # Apply mixer Hamiltonian via single-qubit rotations
#         state = self.apply_mixer(state, beta)
#         return state

#     def _expectation(self, params):
#         """Return the expectation value using element-wise vector multiplication."""
#         state = np.ones(self.N, dtype=complex) / np.sqrt(self.N)
#         betas = params[: self.p]
#         gammas = params[self.p :]
#         for beta, gamma in zip(betas, gammas):
#             state = self.ansatz_layer(state, beta, gamma)
#         # Element-wise multiplication (*) instead of matrix multiplication (@)
#         return np.real(np.vdot(state, self.Hp * state))
#     # def create_problem_hamiltonian(self, Q):
#     #     """Create diagonal Hamiltonian from the QUBO matrix."""
#     #     H = np.zeros(self.N)
#     #     for k in range(self.N):
#     #         b = bin(k)[2:].zfill(self.n)
#     #         energy = 0.0
#     #         for i in range(self.n):
#     #             for j in range(i, self.n):
#     #                 if b[i] == "1" and b[j] == "1":
#     #                     energy += Q[i, j]
#     #         H[k] = energy
#     #     return np.diag(H)

#     def create_mixer_hamiltonian(self):
#         """Construct the transverse field mixer Hamiltonian: sum of Pauli-X on each qubit."""
#         X = np.array([[0, 1], [1, 0]])
#         H = np.zeros((self.N, self.N))
#         for i in range(self.n):
#             ops = [np.eye(2) for _ in range(self.n)]
#             ops[i] = X
#             term = ops[0]
#             for j in range(1, self.n):
#                 term = np.kron(term, ops[j])
#             H += term
#         return H

#     # def ansatz_layer(self, state, beta, gamma):
#     #     # Apply problem Hamiltonian (diagonal phase)
#     #     diag_Hp = np.diag(self.Hp)
#     #     state = state * np.exp(-1j * gamma * diag_Hp)
#     #     # Apply mixer Hamiltonian via single-qubit rotations
#     #     state = self.apply_mixer(state, beta)
#     #     return state

#     def apply_mixer(self, state, beta):
#         n_qubits = self.n
#         gate = np.array(
#             [[np.cos(beta), -1j * np.sin(beta)], [-1j * np.sin(beta), np.cos(beta)]]
#         )
#         for qubit in range(n_qubits):
#             stride = 2**qubit
#             for i in range(0, len(state), 2 ** (qubit + 1)):
#                 for j in range(i, i + stride):
#                     v0 = state[j]
#                     v1 = state[j + stride]
#                     state[j], state[j + stride] = (
#                         gate[0, 0] * v0 + gate[0, 1] * v1,
#                         gate[1, 0] * v0 + gate[1, 1] * v1,
#                     )
#         return state

#     def run(self, p=1, maxiter=50):
#         """Optimize QAOA parameters and return final state data."""
#         self.p = p
#         initial_params = np.random.rand(2 * p) * np.pi
#         res = minimize(
#             self._expectation,
#             initial_params,
#             method="L-BFGS-B",#"COBYLA", #"L-BFGS-B",
#             options={"maxiter": maxiter},
#         )
#         self.optimal_params = res.x
#         return self._final_state()

#     # def _expectation(self, params):
#     #     """Return the expectation value of the problem Hamiltonian."""
#     #     state = np.ones(self.N, dtype=complex) / np.sqrt(
#     #         self.N
#     #     )  # Uniform superposition
#     #     betas = params[: self.p]
#     #     gammas = params[self.p :]
#     #     for beta, gamma in zip(betas, gammas):
#     #         state = self.ansatz_layer(state, beta, gamma)
#     #     return np.real(np.vdot(state, self.Hp @ state))


#     def _final_state(self):
#         """Generate the final state from the optimized parameters."""
#         state = np.ones(self.N, dtype=complex) / np.sqrt(self.N)
#         betas = self.optimal_params[: self.p]
#         gammas = self.optimal_params[self.p :]
#         for beta, gamma in zip(betas, gammas):
#             state = self.ansatz_layer(state, beta, gamma)
#         self.probs = np.abs(state) ** 2
#         self.optimal_state = int(np.argmax(self.probs))
#         return self.optimal_state, self.probs
class QAOA:
    def __init__(self, Q):
        self.n = Q.shape[0]  # Number of qubits
        self.N = 2**self.n  # Hilbert space dimension
        self.Hp = self.create_problem_hamiltonian(Q)

    def create_problem_hamiltonian(self, Q):
        """Create a 1D diagonal array representing the Hamiltonian (No dense matrices)."""
        H = np.zeros(self.N)
        for k in range(self.N):
            b = bin(k)[2:].zfill(self.n)
            energy = 0.0
            for i in range(self.n):
                for j in range(i, self.n):
                    if b[i] == "1" and b[j] == "1":
                        energy += Q[i, j]
            H[k] = energy
        return H  # REMOVED np.diag() -> Keeps it as a fast 1D vector

    def ansatz_layer(self, state, beta, gamma):
        # Fully vectorized element-wise multiplication for Phase separation
        state = state * np.exp(-1j * gamma * self.Hp)
        # Apply the completely vectorized mixer
        state = self.apply_mixer(state, beta)
        return state

    def apply_mixer(self, state, beta):
        """
        Applies mixer Hamiltonian via completely vectorized axis manipulation.
        Eliminates all internal Python loops.
        """
        gate = np.array(
            [[np.cos(beta), -1j * np.sin(beta)], [-1j * np.sin(beta), np.cos(beta)]]
        )

        for qubit in range(self.n):
            stride = 2**qubit
            # Reshape array to isolate the target qubit dimension
            reshaped = state.reshape(-1, 2, stride)
            v0 = reshaped[:, 0, :]
            v1 = reshaped[:, 1, :]

            # Compute the linear combination using fast vector arithmetic
            new_v0 = gate[0, 0] * v0 + gate[0, 1] * v1
            new_v1 = gate[1, 0] * v0 + gate[1, 1] * v1

            # Stack and flatten back to the state vector shape instantly
            state = np.stack([new_v0, new_v1], axis=1).ravel()

        return state

    def run(self, p=1, maxiter=50):
        """Optimize QAOA parameters and return final state data."""
        self.p = p
        initial_params = np.random.rand(2 * p) * np.pi
        res = minimize(
            self._expectation,
            initial_params,
            method="L-BFGS-B",  # "COBYLA", #"L-BFGS-B",#LBFGS gives better for shorter iterations
            options={"maxiter": maxiter},
        )
        self.optimal_params = res.x
        return self._final_state()

    def _expectation(self, params):
        """Return the expectation value using element-wise vector operations O(2^n)."""
        state = np.ones(self.N, dtype=complex) / np.sqrt(self.N)
        betas = params[: self.p]
        gammas = params[self.p :]
        for beta, gamma in zip(betas, gammas):
            state = self.ansatz_layer(state, beta, gamma)
        # Element-wise array multiplication (*) instead of dense matrix multiplication (@)
        return np.real(np.vdot(state, self.Hp * state))

    def _final_state(self):
        """Generate the final state from the optimized parameters."""
        state = np.ones(self.N, dtype=complex) / np.sqrt(self.N)
        betas = self.optimal_params[: self.p]
        gammas = self.optimal_params[self.p :]
        for beta, gamma in zip(betas, gammas):
            state = self.ansatz_layer(state, beta, gamma)
        self.probs = np.abs(state) ** 2
        self.optimal_state = int(np.argmax(self.probs))
        return self.optimal_state, self.probs


def state_to_bin(state, n):
    """Convert an integer state to a binary string with n bits."""
    return bin(state)[2:].zfill(n)


def multiple_random_qubos(n, num_of_qubos):
    """Generates num_of_qubos random QUBO matrix of size n x n."""
    Qubos = {}
    for i in range(num_of_qubos):
        Q = random_qubo(n)
        Qubos[i] = Q
    return Qubos


# --- Main comparison functions ---
def compare_algorithms(trials=100, n=10, p=4, trialQs=None):
    """
    Run 100 trials of the VQPM vs QAOA comparison for n qubits.
    For each trial the following are computed:
      - Hamming distance (number of different bits) between the target state and the found state.
      - The final probability (or max probability) achieved for the found state.

    Returns a dictionary containing the results for VQPM and QAOA.
    """
    bit_diff_vqpm = []
    bit_diff_qaoa = []
    final_prob_vqpm = []
    final_prob_qaoa = []
    final_prob_qaoa_max = []
    for trial in range(trials):
        if trialQs is not None:
            # Generate a random QUB O instance (function defined elsewhere)
            Q = random_qubo(n)
        else:
            Q = trialQs[trial]

        # Adjust the unitary phase to create the proper settings
        # adjust_unitary_phase returns: Q_adj, u, phases, target_state
        Q_adj, u, phases, target_state = adjust_unitary_phase(n, Q)

        # --- Run VQPM ---
        # vqpm_for_qubo returns:
        # (vqpm_state, vqpm_max_prob, q_states, _, vqpm_iters, vqpm_probs)
        vqpm_state, vqpm_max_prob, q_states, _, vqpm_iters, vqpm_probs = vqpm_for_qubo(
            u,
            n,
            expected_idx=target_state,
            max_iter=30,
            pdiff=0.01,
            precision=3,
            dynamic_pdiff_policy=None,  # or "none" if preferred
            qubit_weights=None,
            locking="yes",
        )

        # --- Run QAOA ---
        # Here we use the phase-adjusted QUBO (Q_adj).
        qaoa = QAOA(Q_adj)
        # qaoa.run returns (qaoa_state, qaoa_probs)
        qaoa_state, qaoa_probs = qaoa.run(p=p, maxiter=500)

        # --- Compute Bit Differences ---
        # Compute the Hamming distance between the target state and the found states.
        diff_v = target_state ^ vqpm_state
        diff_q = target_state ^ qaoa_state
        bit_diff_v = state_to_bin(diff_v, n).count("1")
        bit_diff_q = state_to_bin(diff_q, n).count("1")

        bit_diff_vqpm.append(bit_diff_v)
        bit_diff_qaoa.append(bit_diff_q)

        # --- Record Final Probabilities ---
        # For VQPM we record the final maximum probability found.
        final_prob_vqpm.append(vqpm_probs[-1])  # Last probability in the list
        # For QAOA, assuming qaoa_probs is a state-probability map, we access the probability for qaoa_state.
        final_prob_qaoa.append(qaoa_probs[target_state])
        final_prob_qaoa_max.append(qaoa_probs[qaoa_state])

    return {
        "bit_diff_vqpm": bit_diff_vqpm,
        "bit_diff_qaoa": bit_diff_qaoa,
        "final_prob_vqpm": final_prob_vqpm,
        "final_prob_qaoa": final_prob_qaoa,
        "final_prob_qaoa_max": final_prob_qaoa_max,
    }


def plot_bit_differences(results, p=4):
    """Generate a plot that compares the bit differences over trials."""
    trials = range(len(results["bit_diff_vqpm"]))
    plt.figure(figsize=(8, 6))
    plt.plot(trials, results["bit_diff_vqpm"], "o-", label="VQPM Bit Differences")
    plt.plot(
        trials, results["bit_diff_qaoa"], "s-", label=f"QAOA (p={p}) Bit Differences"
    )
    plt.xlabel("Trial")
    plt.ylabel("Number of Different Bits")
    plt.title("Bit Differences Between Target and Found States")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig = plt.gcf()  # Get the current figure object.
    plt.show()
    return fig  # Return the figure object for further manipulation if needed.


# =============================================================================


def plot_bit_differences_as_bars(results, p=4, ax=None):
    """Generate a bar plot that compares the bit differences over trials."""
    if ax is None:
        ax = plt.gca()
    trials = range(len(results["bit_diff_vqpm"]))

    width = 0.35  # Width of the bars

    ax.bar(
        np.array(trials) - width / 2,
        results["bit_diff_vqpm"],
        width,
        label="VQPM Bit Differences",
        color="skyblue",
        alpha=0.7,
    )
    ax.bar(
        np.array(trials) + width / 2,
        results["bit_diff_qaoa"],
        width,
        label=f"QAOA (p={p}) Bit Differences",
        color="salmon",
        alpha=0.7,
    )
    mean_diff_v = np.mean(results["bit_diff_vqpm"])
    ax.axhline(
        mean_diff_v, color="b", linestyle="-.", label=f"Mean VQPM= {mean_diff_v:.2f}"
    )
    mean_diff_q = np.mean(results["bit_diff_qaoa"])
    ax.axhline(
        mean_diff_q, color="r", linestyle="--", label=f"Mean QAOA= {mean_diff_q:.2f}"
    )

    fig = plt.gcf()  # Get the current figure object.
    # plt.show()
    return fig  # Return the figure object for further manipulation if needed.


# =============================================================================


def plot_final_probs(results, p=4, ax=None):
    """Generate a plot that compares the final probabilities over trials."""
    if ax is None:
        ax = plt.gca()

    trials = range(len(results["final_prob_vqpm"]))
    ax.plot(trials, results["final_prob_vqpm"], "o", label="VQPM Final Probability")
    ax.plot(
        trials,
        results["final_prob_qaoa"],
        "^",
        label=f"QAOA (p={p}) Final Probability",
    )

    mean_diff_v = np.mean(results["final_prob_vqpm"])
    ax.axhline(
        mean_diff_v, color="b", linestyle="-.", label=f"Mean VQPM= {mean_diff_v:.2f}"
    )
    mean_diff_q = np.mean(results["final_prob_qaoa"])
    ax.axhline(
        mean_diff_q, color="r", linestyle="--", label=f"Mean QAOA= {mean_diff_q:.2f}"
    )

    fig = plt.gcf()  # Get the current figure object.
    # plt.show()
    return fig  # Return the figure object for further manipulation if needed.


# =============================================================================
# %%
def compare_scaling_across_n(n_values, trials=10, p_values=[2, 4, 6, 8]):
    """
    Compares VQPM and QAOA across different qubit counts (n) and QAOA layer depths (p).
    Computes the average Hamming distance (different bits) and target state probabilities.

    Parameters:
    -----------
    n_values : list of int
        The qubit numbers to evaluate (e.g., [4, 6, 8, 10]).
    trials : int
        The number of random QUBO instances to average over for each n.
    p_values : list of int
        The QAOA depths (p) to evaluate.
    """
    # Structures to accumulate results across different values of n
    scaling_results = {
        "vqpm_bit_diff": [],
        "vqpm_target_prob": [],
        "qaoa_bit_diff": {p: [] for p in p_values},
        "qaoa_target_prob": {p: [] for p in p_values},
    }

    for n in n_values:
        print(f"Evaluating n = {n} qubits over {trials} trials...")

        # Temporary lists to hold data for the current n
        trial_vqpm_bits = []
        trial_vqpm_probs = []

        trial_qaoa_bits = {p: [] for p in p_values}
        trial_qaoa_target_probs = {p: [] for p in p_values}

        for trial in range(trials):
            # Generate a fresh random QUBO instance for this trial
            Q = random_qubo(n)
            Q_adj, u, phases, target_state = adjust_unitary_phase(n, Q)

            # --- Run VQPM ---
            vqpm_state, vqpm_max_prob, q_states, _, vqpm_iters, vqpm_probs = (
                vqpm_for_qubo(
                    u,
                    n,
                    expected_idx=target_state,
                    max_iter=30,
                    pdiff=0.01,
                    precision=3,
                    dynamic_pdiff_policy=None,
                    qubit_weights=None,
                    locking="yes",
                )
            )

            diff_v = target_state ^ vqpm_state
            bit_diff_v = bin(diff_v)[2:].zfill(n).count("1")
            trial_vqpm_bits.append(bit_diff_v)
            trial_vqpm_probs.append(vqpm_probs[-1])  # Last probability in the sequence

            # --- Run QAOA for each p value ---
            qaoa = QAOA(Q_adj)
            for p in p_values:
                qaoa_state, qaoa_probs = qaoa.run(p=p, maxiter=500)

                diff_q = target_state ^ qaoa_state
                bit_diff_q = bin(diff_q)[2:].zfill(n).count("1")

                trial_qaoa_bits[p].append(bit_diff_q)
                trial_qaoa_target_probs[p].append(qaoa_probs[target_state])

        # Compute and record the averages over all trials for this n
        scaling_results["vqpm_bit_diff"].append(np.mean(trial_vqpm_bits))
        scaling_results["vqpm_target_prob"].append(np.mean(trial_vqpm_probs))

        for p in p_values:
            scaling_results["qaoa_bit_diff"][p].append(np.mean(trial_qaoa_bits[p]))
            scaling_results["qaoa_target_prob"][p].append(
                np.mean(trial_qaoa_target_probs[p])
            )

    # -------------------------------------------------------------------------
    # Matplotlib configuration for consistent styling matching your paper.
    # -------------------------------------------------------------------------
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 14,  # Marginally smaller to accommodate more labels gracefully
            "figure.titlesize": 18,
            "figure.figsize": (14, 5),
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "lines.linewidth": 2,
            "lines.markersize": 6,
        }
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Unique marker and color configurations for each QAOA p depth
    markers = {2: "s-", 4: "^-", 6: "v-", 8: "d-"}
    colors = {2: "#ff7f0e", 4: "#2ca02c", 6: "#d62728", 8: "#9467bd"}

    # --- Plot 1: Average Bit Differences vs n ---
    ax1.plot(
        n_values,
        scaling_results["vqpm_bit_diff"],
        "o-",
        label="VQPM",
        color="#1f77b4",
        linewidth=3,
    )
    for p in p_values:
        ax1.plot(
            n_values,
            scaling_results["qaoa_bit_diff"][p],
            markers[p],
            label=f"QAOA ($p={p}$)",
            color=colors[p],
        )
    ax1.set_xlabel("Number of Qubits ($n$)")
    ax1.set_ylabel("Avg. Different Bits")
    ax1.set_title("Hamming Distance Scaling")
    ax1.set_xticks(n_values)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # --- Plot 2: Final Target State Probabilities vs n ---
    ax2.plot(
        n_values,
        scaling_results["vqpm_target_prob"],
        "o-",
        label="VQPM",
        color="#1f77b4",
        linewidth=3,
    )
    for p in p_values:
        ax2.plot(
            n_values,
            scaling_results["qaoa_target_prob"][p],
            markers[p],
            label=f"QAOA ($p={p}$)",
            color=colors[p],
        )
    ax2.set_xlabel("Number of Qubits ($n$)")
    ax2.set_ylabel("Avg. Target State Probability")
    ax2.set_title("Target State Probability Scaling")
    ax2.set_xticks(n_values)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("figures/vqpm_vs_qaoa_scaling_n.png", dpi=300)
    plt.savefig("figures/vqpm_vs_qaoa_scaling_n.pdf", dpi=300)
    plt.show()

    return scaling_results


# ---- Main Routine ----
def main1():
    # For reproducibility (optional)
    np.random.seed(42)
    # Set parameters
    n = 8  # Number of qubits
    # p for qubo
    p = 8  # Number of layers in QAOA

    trials = 100  # Number of trials
    trialQs = multiple_random_qubos(n, trials)
    # Generate random QUBO matrices

    # Run the comparisons
    results = compare_algorithms(trials=trials, n=n, p=p, trialQs=trialQs)

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
            "figure.figsize": (14, 5),
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "lines.linewidth": 2,
            "lines.markersize": 6,
        }
    )
    # Plot the results:
    # 1. Bit differences over 100 trials
    # plot_bit_differences(results)
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    # 2. Final probabilities over 100 trials
    fig1 = plot_final_probs(results, p)
    plt.xlabel("Trial")
    plt.ylabel("Final Probability")
    plt.title("Final Probabilities of the Target States")
    plt.legend()

    plt.grid(True)
    plt.tight_layout()

    plt.subplot(1, 2, 2)
    fig2 = plot_bit_differences_as_bars(results, p)
    plt.xlabel("Trial")
    plt.ylabel("Number of Different Bits")
    plt.title("Different Bits for Each Trial")
    plt.legend()
    plt.tight_layout()

    # plt.figure(fig1)
    # plt.figure(fig2)
    plt.tight_layout()
    plt.savefig(f"figures/vqpm_vs_qaoa_p{p}_n{n}.png", dpi=300)
    plt.savefig(f"figures/vqpm_vs_qaoa_p{p}_n{n}.pdf", dpi=300)
    plt.show()

    # fig2.savefig(f"figures/vqpm_vs_qaoa_p{p}_bits.png")


def main2():
    np.random.seed(42)

    # Define a range of qubit counts to test
    n_values = np.arange(4, 21, 2)  # Extend to larger n for scaling analysis
    trials = 20  # Number of trials per qubit size

    # Run scaling analysis across different values of n and p
    results = compare_scaling_across_n(
        n_values=n_values, trials=trials, p_values=[2, 4, 6, 8]
    )


if __name__ == "__main__":
    # main1()
    main2()
