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


# =============================================================================
# Helper: Convert state to binary string for display
# =============================================================================
def state_to_bin(state, n):
    return bin(state)[2:].zfill(n)


# =============================================================================
# Single Trial Execution: Run the VQPM algorithm for one instance of QUBO.
# =============================================================================
def run_vqpm_trial(
    n=15,
    max_iter=20,
    pdiff=0.01,
    precision=3,
    dynamic_pdiff_policy="none",
    compute_qubit_weights_in_Q=False,
    locking="no",  # 'yes' | 'no' | 'only_when_p0_increases'
    Q=None,
):
    if Q is None:
        # Generate QUBO problem and compute parameters
        Q = random_qubo(n)
    utest = test_u_for_qubo(n, Q)  # Get actual QUBO values
    if compute_qubit_weights_in_Q:
        qubit_weights = compute_qubit_weights(Q)
    else:
        qubit_weights = None

    # Prepare phase-adjusted unitary
    Q_adj, u, phases, target_state = adjust_unitary_phase(n, Q)

    # Run VQPM algorithm
    result_state, max_prob, q_states, p0, iters, probs = vqpm_for_qubo(
        u,
        n,
        expected_idx=target_state,
        max_iter=max_iter,
        pdiff=pdiff,
        precision=precision,
        dynamic_pdiff_policy=dynamic_pdiff_policy,
        qubit_weights=qubit_weights,
        locking=locking,
    )

    # Convert the states into binary strings for readability
    target_bin = state_to_bin(target_state, n)
    result_bin = state_to_bin(result_state, n)

    # Compute bit difference between target and found states
    state_diff = target_state ^ result_state  # XOR between states
    diff_bits = state_to_bin(state_diff, n)
    num_diff_bits = diff_bits.count("1")

    # Calculate theoretical maximum success probability
    success_prob = np.abs(u[target_state]) ** 2 / (2**n)

    # Package all the results together
    result_dict = {
        "Q": Q,
        "utest": utest,
        "target_state": target_state,
        "target_bin": target_bin,
        "result_state": result_state,
        "result_bin": result_bin,
        "num_diff_bits": num_diff_bits,
        "success_prob": success_prob,
        "max_prob": max_prob,
        "probs": probs,  # Convergence curve (list/array of probabilities per iteration)
        "iters": iters,
    }
    return result_dict


# =============================================================================
# Multiple Trials: Run multiple [given] experiments and collect convergence data.
# =============================================================================
def run_multiple_trials(
    num_trials,
    n=15,
    max_iter=20,
    pdiff=0.01,
    precision=3,
    dynamic_pdiff_policy="none",  # 'none' | 'hoeffding' | 'scaledby2'
    compute_qubit_weights_in_Q=False,
    locking="no",  # 'yes' | 'no' | 'only_when_p0_increases'
    trials=None # Optional: Provide a list of qubo matrices to run
):
    trials_results = []
    convergence_curves = []

    for trial in range(num_trials):
        if trials is not None:
            Q = trials[trial]
        else:
            Q = None
        
        result = run_vqpm_trial(
            n=n,
            max_iter=max_iter,
            pdiff=pdiff,
            precision=precision,
            dynamic_pdiff_policy=dynamic_pdiff_policy,
            compute_qubit_weights_in_Q=compute_qubit_weights_in_Q,
            locking=locking,
            Q=Q,
        )
        trials_results.append(result)
        convergence_curves.append(result["probs"])

    return trials_results, convergence_curves

# =============================================================================
# Main entry point to run experiments and plot results.
# =============================================================================
if __name__ == "__main__":
    # Make sure to set seed for reproducibility.
    # np.random.seed(42)
    # Configuration parameters.
    options = {
        "num_trials": 100,   # Total number of trials.
        "n": 5,            # QUBO problem size.
        "max_iter": 30,  # Maximum number of iterations per trial.
        "pdiff": 0.01,
        "precision": 3, # np.rounding precision. for 3 e.g. 0.51
        "dynamic_pdiff_policy": "none",  # 'none' | 'hoeffding' | 'scaledby2'
        "compute_qubit_weights_in_Q": False,
        "locking": "no",  # 'yes' | 'no' | 'only_when_p0_increases'-use only with better precision
        "file_name": "figures/fixed-pdiff-no-locking",  # File name for saving figures.
    }
    # First, you might want to test your original modular experiment.
    trials_results, convergence_curves = run_multiple_trials(
        options["num_trials"],
        options["n"],
        max_iter=options["max_iter"],
        pdiff=options["pdiff"],
        precision=options["precision"],
        dynamic_pdiff_policy=options["dynamic_pdiff_policy"],
        compute_qubit_weights_in_Q=options["compute_qubit_weights_in_Q"],
        locking=options["locking"],
    )


    from run_multiple import plot_results

    # Plot the energy landscape (from one trial) and convergence curves with mean.
    plot_results(trials_results, convergence_curves, n)
