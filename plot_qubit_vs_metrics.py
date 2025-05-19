import numpy as np
import matplotlib.pyplot as plt
from run_multiple import run_multiple_trials


# =============================================================================
# Plotting: Visualize the energy landscape and convergence progress (with mean).
# =============================================================================
def plot_results(trials_results, convergence_curves, n):
    # -------------------------------------------------------------------------
    # Plot: Number of Different Bits for Each Trial
    # -------------------------------------------------------------------------
    # Extract "number of diff bits" from each trial result.
    diff_bits_all = [trial["num_diff_bits"] for trial in trials_results]

    # -------------------------------------------------------------------------
    # Prepare Convergence Curves: Align curves by truncating to the minimum curve length.
    # -------------------------------------------------------------------------
    min_length = min(len(curve) for curve in convergence_curves)
    aligned_curves = np.array([curve[:min_length] for curve in convergence_curves])
    # mean_convergence = np.mean(aligned_curves, axis=0)

    max_length = max(len(curve) for curve in convergence_curves)
    padded_curves = np.array(
        [
            np.pad(curve, (0, max_length - len(curve)), constant_values=np.nan)
            for curve in convergence_curves
        ]
    )
    # Compute mean convergence over aligned curves.
    mean_convergence = np.nanmean(padded_curves, axis=0)

    # Create a figure with two subplots (one for diff bits, one for convergence progress).
    plt.figure(figsize=(14, 5))

    # -------------------------------------------------------------------------
    # Subplot 1: Bar Chart for "Number of Different Bits" per trial.
    # -------------------------------------------------------------------------
    plt.subplot(1, 2, 1)
    trials = list(range(1, len(diff_bits_all) + 1))
    plt.bar(trials, diff_bits_all, color="skyblue", alpha=0.7)
    mean_diff = np.mean(diff_bits_all)
    plt.axhline(mean_diff, color="r", linestyle="--", label=f"Mean = {mean_diff:.2f}")
    plt.xlabel("Trial")
    plt.ylabel("Number of Different Bits")
    plt.title("Different Bits for Each Trial")
    plt.legend()

    # -------------------------------------------------------------------------
    # Subplot 2: Convergence Plot (with mean convergence over aligned curves).
    # -------------------------------------------------------------------------
    plt.subplot(1, 2, 2)
    for idx, curve in enumerate(padded_curves):
        plt.plot(
            range(len(curve)), curve, "-", alpha=0.5
        )  # label=f'Trial {idx+1}' if idx == 0 else None)
    plt.plot(
        range(mean_convergence.shape[0]),
        mean_convergence,
        "k-o",
        markersize=6,
        linewidth=3,
        label="Mean Convergence",
    )
    plt.axhline(1 / (2**n), color="r", linestyle="--", label="Random guess (1/2ⁿ)")
    plt.axhline(1 / (n), color="b", linestyle="-.", label="Polynomial line (1/n)")
    plt.xlabel("Iteration")
    plt.ylabel("Success Probability")
    plt.title("Convergence Progress")
    plt.legend()

    plt.tight_layout()
    fig = plt.gcf()  # Get the current figure object.
    plt.show()
    return fig  # Return the figure object for further manipulation if needed.


# =============================================================================
# Qubit-Dependent Analysis: Run multiple trials for different qubit counts.
# =============================================================================
def plot_qubits_vs_metrics(
    n_min=1,
    n_max=10,
    num_trials=10,
    max_iter=20,
    pdiff=0.01,
    precision=3,
    dynamic_pdiff_policy="none",
    compute_qubit_weights_in_Q=False,
    locking="yes",
):
    """
    For each number of qubits in [n_min, n_max], run multiple VQPM trials,
    compute the mean number of bit differences (num_diff_bits) and
    the mean final success probability (from the last element of probs).

    Plots:
       1. Mean number of different bits vs. number of qubits.
       2. Mean final success probability vs. number of qubits, with a reference curve for random guessing (1/2^n).
    """
    qubit_counts = []
    mean_diff_bits = []
    mean_final_probs = []

    # Iterate through different problem sizes.
    for n in range(n_min, n_max + 1):
        print(f"Running {num_trials} trials for {n} qubit(s)...")

        # Run multiple trials for the current number of qubits.
        trials_results, convergence_curves = run_multiple_trials(
            num_trials,
            n,
            max_iter=max_iter,
            pdiff=pdiff,
            precision=precision,
            dynamic_pdiff_policy=dynamic_pdiff_policy,
            compute_qubit_weights_in_Q=compute_qubit_weights_in_Q,
            locking=locking,  #
        )

        # Compute mean number of different bits.
        diff_bits_all = [trial["num_diff_bits"] for trial in trials_results]
        mean_diff = np.mean(diff_bits_all)

        # Extract the final success probability for each trial.
        final_probs = [
            trial["probs"][-1] for trial in trials_results if len(trial["probs"]) > 0
        ]
        mean_prob = np.mean(final_probs) if final_probs else np.nan

        qubit_counts.append(n)
        mean_diff_bits.append(mean_diff)
        mean_final_probs.append(mean_prob)

    # Plot the metrics versus the number of qubits.
    plt.figure(figsize=(14, 5))

    # Subplot 1: Mean number of different bits.
    plt.subplot(1, 2, 1)
    plt.plot(qubit_counts, mean_diff_bits, "bo-", label="Mean Different Bits")
    plt.xlabel("Number of Qubits")
    plt.ylabel("Mean Different Bits")
    plt.title("Mean Number of Different Bits vs. Qubit Count")
    plt.legend()

    # Subplot 2: Mean final success probability.
    plt.subplot(1, 2, 2)
    plt.plot(
        qubit_counts, mean_final_probs, "go-", label="Mean Final Success Probability"
    )
    # For reference, compute the success probability of a random guess: 1/(2^n)
    random_guess_probs = [1 / (2**n) for n in qubit_counts]
    poly_line = [1 / (n) for n in qubit_counts]
    plt.plot(qubit_counts, random_guess_probs, "r--", label="Random guess (1/2ⁿ)")
    plt.plot(qubit_counts, poly_line, "b-.", label="Polynomial line (1/n)")
    plt.xlabel("Number of Qubits")
    plt.ylabel("Mean Final Success Probability")
    plt.title("Mean Final Success Probability vs. Qubit Count")
    plt.legend()

    plt.tight_layout()
    fig = plt.gcf()  # Get the current figure object.
    plt.show()
    return fig  # Return the figure object for further manipulation if needed.


# Example usage (this line can go inside your main or testing routine):
if __name__ == "__main__":
    # Make sure to set seed for reproducibility.
    np.random.seed(42)
    # Configuration parameters.
    n_min = 1
    n_max = 18
    options = {
        "num_trials": 100,  # Total number of trials.
        "n": 15,  # QUBO problem size.
        "max_iter": 500,  # Maximum number of iterations per trial.
        "pdiff": 0.01,
        "precision": 3,  # np.rounding precision. for 3 e.g. 0.521
        "dynamic_pdiff_policy": "none",  # 'none' | 'hoeffding' | 'scaledby2'
        "compute_qubit_weights_in_Q": False,  # True | False
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

    fig1 = plot_results(trials_results, convergence_curves, options["n"])

    fig1.savefig(f"{options['file_name']}1.png", dpi=300)
    fig1.savefig(f"{options['file_name']}1.pdf", dpi=300)

    # Now, run the qubit-dependent analysis.
    fig2 = plot_qubits_vs_metrics(
        n_min=n_min,
        n_max=n_max,
        num_trials=options["num_trials"],
        max_iter=options["max_iter"],
        pdiff=options["pdiff"],
        precision=options["precision"],
        dynamic_pdiff_policy=options["dynamic_pdiff_policy"],
        compute_qubit_weights_in_Q=options["compute_qubit_weights_in_Q"],
        locking=options["locking"],
    )
    fig2.savefig(f"{options['file_name']}2.png", dpi=300)
    fig2.savefig(f"{options['file_name']}2.pdf", dpi=300)

    # Save the dictionary to a text file in JSON format
    import json

    with open(f"{options['file_name']}-options.txt", "w") as file:
        json.dump(options, file, indent=4)
