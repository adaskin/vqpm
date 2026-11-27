"""Qiskit Aer (optionally noisy) circuit implementation of VQPM for QUBO.

Each iteration prepares the product state prod_q Ry(theta_q)|0> on the n
data qubits, applies H on a control qubit, the controlled diagonal unitary
U = diag(exp(i (a E(x) + b))) (phase_map coefficients a, b; the paper's map
is a = pi/(4S), b = pi/4), H on the control again, and measures.

measure_mode='all' (default): one circuit per iteration measures the control
and ALL data qubits; the marginals of every data qubit come from the same
shots, post-selected on control = 0. measure_mode='per_qubit' is the
original scheme (one circuit per data qubit, `shots` each).

Data qubit i holds QUBO variable i (bit i of the state index, bit 0 = most
significant, as in vqpm.py).

Usage (from the repository root or this folder):
    python vqpm_qiskit_noisy/vqpm_qiskit.py --n 12 --noise fake:FakeTorino
    python vqpm_qiskit_noisy/vqpm_qiskit.py --gate-counts 8 12 16
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile  # noqa: E402
from qiskit.circuit import ParameterVector  # noqa: E402
from qiskit_aer import AerSimulator  # noqa: E402
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error  # noqa: E402

import vqpm as vqpm_root  # noqa: E402  (root vqpm.py: energies, phase maps, bounds)

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------- simulators
def make_simulator(noise=None, seed=None, method="automatic", threads=None):
    """AerSimulator for a noise specification.

    noise:
        None / 'none'          ideal simulator
        'fake:<BackendName>'   AerSimulator.from_backend(<fake backend>)
                               (calibrated noise, native gates and coupling map
                               of e.g. FakeTorino, FakeSherbrooke, FakePeekskill)
        'depol:p1,p2,ro'       depolarising error p1 on sx/x, p2 on CX, symmetric
                               readout error ro, all-to-all coupling (no routing).
                               Median FakeTorino values: 2.7e-4, 4.1e-3, 2.3e-2.
    Note: with a fake backend the routed circuit uses extra physical qubits
    (SWAP paths on heavy-hex), so density-matrix simulation becomes
    infeasible beyond n ~ 8.
    """
    sim = _make_simulator(noise, seed, method)
    if threads is not None:
        sim.set_options(max_parallel_threads=int(threads))
    return sim


def _make_simulator(noise, seed, method):
    if noise in (None, "none", "ideal"):
        return AerSimulator(seed_simulator=seed, method=method)
    if noise.startswith("fake:"):
        from qiskit_ibm_runtime import fake_provider

        backend = getattr(fake_provider, noise.split(":", 1)[1])()
        return AerSimulator.from_backend(backend, seed_simulator=seed, method=method)
    if noise.startswith("depol:"):
        p1, p2, ro = (float(x) for x in noise.split(":", 1)[1].split(","))
        nm = NoiseModel(basis_gates=["cx", "rz", "sx", "x", "id"])
        # rz is a virtual (error-free) gate on IBM hardware
        nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["sx", "x"])
        nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
        nm.add_all_qubit_readout_error(ReadoutError([[1 - ro, ro], [ro, 1 - ro]]))
        return AerSimulator(noise_model=nm, seed_simulator=seed, method=method)
    raise ValueError(f"unknown noise spec {noise!r}")


# ----------------------------------------------------------------- circuits
def append_controlled_u(qc, control, data, Q, phase_map="paper"):
    """Controlled diag(exp(i (a E(x) + b))): P(b) on the control, CP(a Q_ii)
    for the linear terms and CCP(a Q_ij) (MCPhase) for i < j."""
    n = len(data)
    a, b = vqpm_root.phase_map_coefficients(Q, phase_map)
    Qs = np.asarray(Q, dtype=float) * a
    qc.p(b, control)  # global phase offset of U, applied when control = 1
    for i in range(n):
        if abs(Qs[i, i]) > 1e-10:
            qc.cp(Qs[i, i], control, data[i])
    for i in range(n):
        for j in range(i + 1, n):
            if abs(Qs[i, j]) > 1e-10:
                qc.mcp(Qs[i, j], [control, data[j]], data[i])
    return qc


def controlled_u_circuit(Q, phase_map="paper"):
    n = Q.shape[0]
    qc_ = QuantumRegister(1, "control")
    qd = QuantumRegister(n, "data")
    qc = QuantumCircuit(qc_, qd)
    append_controlled_u(qc, qc_[0], list(qd), Q, phase_map)
    return qc


def vqpm_iteration_circuit(Q, phase_map="paper", measure="all", target_qubit=None):
    """Parameterised circuit of one VQPM iteration (Ry angles as parameters).

    measure='all': clbit 0 = control, clbit i+1 = data qubit i.
    measure='single': clbit 0 = control, clbit 1 = data[target_qubit].
    """
    n = Q.shape[0]
    theta = ParameterVector("theta", n)
    qr_c = QuantumRegister(1, "control")
    qr_d = QuantumRegister(n, "data")
    if measure == "all":
        cr = ClassicalRegister(n + 1, "c")
    else:
        cr = ClassicalRegister(2, "c")
    qc = QuantumCircuit(qr_c, qr_d, cr)
    for i in range(n):
        qc.ry(theta[i], qr_d[i])  # theta = 0 -> |0>, pi -> |1> (locked)
    qc.h(qr_c[0])
    append_controlled_u(qc, qr_c[0], list(qr_d), Q, phase_map)
    qc.h(qr_c[0])
    qc.measure(qr_c[0], cr[0])
    if measure == "all":
        for i in range(n):
            qc.measure(qr_d[i], cr[i + 1])
    else:
        qc.measure(qr_d[target_qubit], cr[1])
    return qc, theta


def gate_counts(n_values=(8, 12, 16), backends=("all-to-all", "FakeTorino"),
                seed=0, optimization_level=3, phase_map="paper"):
    """Transpiled two-qubit gate count and depth of the controlled-U for dense
    random QUBOs. 'all-to-all' uses basis {cx, rz, sx, x} without a coupling
    map; a fake backend name uses its native gates and coupling map."""
    from qiskit_ibm_runtime import fake_provider

    rows = []
    for n in n_values:
        Q = np.random.default_rng([seed, n]).standard_normal((n, n))
        Q = Q + Q.T
        qc = controlled_u_circuit(Q, phase_map)
        for bk in backends:
            t = time.time()
            if bk == "all-to-all":
                tq = transpile(qc, basis_gates=["cx", "rz", "sx", "x"],
                               optimization_level=optimization_level, seed_transpiler=seed)
            else:
                backend = getattr(fake_provider, bk)()
                tq = transpile(qc, backend=backend, optimization_level=optimization_level,
                               seed_transpiler=seed)
            ops = tq.count_ops()
            two_q = sum(c for g, c in ops.items() if g in ("cx", "cz", "ecr"))
            rows.append({
                "n": n,
                "backend": bk,
                "two_qubit_gates": int(two_q),
                "two_qubit_gate_names": sorted(g for g in ops if g in ("cx", "cz", "ecr")),
                "depth": int(tq.depth()),
                "two_qubit_depth": int(tq.depth(lambda ins: ins.operation.num_qubits == 2)),
                "pairs": n * (n - 1) // 2,
                "transpile_seconds": round(time.time() - t, 2),
            })
            print(rows[-1], flush=True)
    return rows


# ----------------------------------------------------------------- VQPM on Aer
class QuantumCircuitVQPM:
    """VQPM driven by sampled circuits.

    locking_rule:
        'pdiff'           lock if |P0 - P1| > pdiff (published rule, after
                          rounding to `precision`)
        'robust'          lock if |P0 - P1| > pdiff + 2 * sqrt(P0 P1 / M)
                          (robust_qubit_locking; M = post-selected shots)
        'hoeffding_union' lock if |P0 - P1| > 2 eps (+ rounding slack),
                          eps = sqrt(ln(2 n T / delta) / (2 M))
    """

    def __init__(self, n_qubits, max_iter=30, pdiff=0.01, precision=3, rand_seed=42,
                 noise=None, locking_rule="pdiff", measure_mode="all",
                 phase_map="paper", delta=0.05, optimization_level=1,
                 transpile_seed=None, verbose=True, method="automatic", threads=None):
        self.n = n_qubits
        self.max_iter = max_iter
        self.pdiff = pdiff
        self.precision = precision  # Decimal places for probability rounding
        self.noise = noise
        self.simulator = make_simulator(noise, seed=rand_seed, method=method, threads=threads)
        self.locking_rule = locking_rule
        self.measure_mode = measure_mode
        self.phase_map = phase_map
        self.delta = delta
        self.optimization_level = optimization_level
        self.transpile_seed = rand_seed if transpile_seed is None else transpile_seed
        self.verbose = verbose
        self.shots = None  # post-selected shots of the last measurement

        # Initialize state tracking
        self.q_states = {}
        self.current_angles = [np.pi / 2] * self.n  # Start with Hadamard state (Ry(pi/2) = H)
        self.qubit_prob_history = []  # Track qubit probabilities over iterations
        self.prepared_history = []  # P(q=1) of the next input state (after locking)
        self.lock_log = []
        self.shots_used = 0
        self.postselected_shots = 0
        self._compiled = {}

    # -------------------------------------------------------------- circuits
    def create_qubo_unitary_circuit(self, Q_scaled):
        """Create circuit that implements the QUBO unitary (uncontrolled)."""
        qr = QuantumRegister(self.n, "q")
        qc = QuantumCircuit(qr)
        for i in range(self.n):
            if abs(Q_scaled[i, i]) > 1e-10:
                qc.rz(2 * Q_scaled[i, i], i)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if abs(Q_scaled[i, j]) > 1e-10:
                    qc.crz(2 * Q_scaled[i, j], j, i)
        return qc

    def _compiled_circuit(self, Q, key):
        if key not in self._compiled:
            measure, target = key
            qc, theta = vqpm_iteration_circuit(Q, self.phase_map, measure, target)
            tqc = transpile(qc, self.simulator, optimization_level=self.optimization_level,
                            seed_transpiler=self.transpile_seed)
            self._compiled[key] = (tqc, theta)
        return self._compiled[key]

    def _run(self, Q, key, angles, shots):
        tqc, theta = self._compiled_circuit(Q, key)
        bound = tqc.assign_parameters({theta[i]: float(angles[i]) for i in range(self.n)})
        counts = self.simulator.run(bound, shots=shots).result().get_counts()
        self.shots_used += shots
        return counts

    # -------------------------------------------------------------- measurement
    def measure_all_qubits(self, angles, Q, shots):
        """One circuit, all data qubits measured; returns P(q=1) per data
        qubit (post-selected on control = 0) and the post-selected count."""
        counts = self._run(Q, ("all", None), angles, shots)
        ones = np.zeros(self.n)
        m_post = 0
        for bitstring, c in counts.items():
            b = bitstring.replace(" ", "")
            if b[-1] != "0":  # clbit 0 = control
                continue
            m_post += c
            for i in range(self.n):
                if b[-(i + 2)] == "1":  # clbit i+1 = data qubit i
                    ones[i] += c
        self.postselected_shots += m_post
        if m_post == 0:
            return None, 0
        return ones / m_post, m_post

    def measure_single_qubit_probability(self, angles, Q, qubit_idx, shots_per_iter=1000):
        """Original scheme: measure one data qubit (and the control)."""
        counts = self._run(Q, ("single", qubit_idx), angles, shots_per_iter)
        count_1 = 0
        m_post = 0
        for bitstring, c in counts.items():
            b = bitstring.replace(" ", "")
            if b[-1] == "0":
                m_post += c
                if b[-2] == "1":
                    count_1 += c
        self.postselected_shots += m_post
        if m_post == 0:
            return 0.5, 0.5, 0
        return 1 - count_1 / m_post, count_1 / m_post, m_post

    # -------------------------------------------------------------- locking
    def robust_qubit_locking(self, prob_0, prob_1, qubit_idx, confidence=0.95):
        """Use statistical confidence intervals for locking (M = self.shots,
        the post-selected shot count of the measurement)."""
        std_err = np.sqrt(prob_0 * prob_1 / max(self.shots, 1))
        margin = 2 * std_err  # ~95% confidence

        if prob_0 > prob_1 + self.pdiff + margin:
            return True, 0.0  # Lock to |0⟩
        elif prob_1 > prob_0 + self.pdiff + margin:
            return True, np.pi  # Lock to |1⟩
        else:
            return False, 2 * np.arcsin(np.sqrt(prob_1))

    def _decide(self, prob_0, prob_1, qubit_idx, iteration):
        """Returns (locked, angle, threshold)."""
        if self.locking_rule == "robust":
            locked, angle = self.robust_qubit_locking(prob_0, prob_1, qubit_idx)
            thr = self.pdiff + 2 * np.sqrt(prob_0 * prob_1 / max(self.shots, 1))
            return locked, angle, thr
        if self.locking_rule == "hoeffding_union":
            thr = vqpm_root.hoeffding_union_threshold(
                self.n, self.max_iter, self.delta, self.shots, self.precision)
        elif self.locking_rule == "pdiff":
            thr = self.pdiff
        else:
            raise ValueError(f"unknown locking_rule {self.locking_rule!r}")
        if prob_0 > prob_1 + thr:
            return True, 0.0, thr
        if prob_1 > prob_0 + thr:
            return True, np.pi, thr
        return False, 2 * np.arcsin(np.sqrt(prob_1)), thr

    def _round(self, prob_0, prob_1):
        if self.precision is None:
            return prob_0, prob_1
        prob_0 = round(prob_0, self.precision)
        prob_1 = round(prob_1, self.precision)
        total = prob_0 + prob_1  # Re-normalize after rounding
        if total > 0:
            return prob_0 / total, prob_1 / total
        return 0.5, 0.5

    def update_angles(self, angles, Q, shots, iteration, optimum=None):
        """Measure, track probabilities and lock; returns the new angles."""
        new_angles = list(angles)
        iteration_probs = {}
        if self.measure_mode == "all":
            p1, m_post = self.measure_all_qubits(angles, Q, shots)
            self.shots = m_post
        for qubit_idx in range(self.n):
            if qubit_idx in self.q_states:
                prob_0, prob_1 = self.q_states[qubit_idx]
                iteration_probs[qubit_idx + 1] = [prob_0, prob_1]
                continue
            if self.measure_mode == "all":
                if p1 is None:  # nothing survived post-selection
                    iteration_probs[qubit_idx + 1] = [np.nan, np.nan]
                    continue
                prob_0, prob_1 = 1 - p1[qubit_idx], p1[qubit_idx]
            else:
                prob_0, prob_1, m_post = self.measure_single_qubit_probability(
                    angles, Q, qubit_idx, shots)
                self.shots = m_post
            prob_0, prob_1 = self._round(prob_0, prob_1)
            iteration_probs[qubit_idx + 1] = [prob_0, prob_1]
            locked, angle, thr = self._decide(prob_0, prob_1, qubit_idx, iteration)
            new_angles[qubit_idx] = angle
            if locked:
                value = int(angle == np.pi)
                self.q_states[qubit_idx] = [1.0, 0.0] if value == 0 else [0.0, 1.0]
                match = None
                if optimum is not None:
                    match = bool(((optimum >> (self.n - 1 - qubit_idx)) & 1) == value)
                self.lock_log.append({
                    "iteration": iteration, "qubit": qubit_idx + 1, "value": value,
                    "p0": float(prob_0), "p1": float(prob_1),
                    "margin": float(abs(prob_0 - prob_1)), "threshold": float(thr),
                    "postselected": int(self.shots), "matches_optimum": match,
                })
        self.qubit_prob_history.append(iteration_probs)
        self.prepared_history.append([float(np.sin(a / 2) ** 2) for a in new_angles])
        return new_angles

    # kept for backwards compatibility with the original script
    def update_angles_from_individual_measurements(self, angles, Q_scaled, shots_per_qubit=1000):
        mode = self.measure_mode
        self.measure_mode = "per_qubit"
        try:
            return self.update_angles(angles, self._Q, shots_per_qubit, len(self.qubit_prob_history))
        finally:
            self.measure_mode = mode

    def run_vqpm(self, Q, shots_per_qubit=1000, shots=None, optimum=None):
        """Run VQPM; `shots` = shots per circuit (per iteration in 'all' mode,
        per qubit in 'per_qubit' mode). `shots_per_qubit` is the old name."""
        shots = shots_per_qubit if shots is None else shots
        self._Q = np.asarray(Q, dtype=float)
        if self.verbose:
            print(f"Running Quantum VQPM for {self.n} qubits ({self.measure_mode} mode, "
                  f"noise={self.noise})")
        t0 = time.time()
        iteration = -1
        for iteration in range(self.max_iter):
            if self.verbose:
                print(f"Iteration {iteration}: Locked qubits: {len(self.q_states)}/{self.n}")
            self.current_angles = self.update_angles(
                self.current_angles, self._Q, shots, iteration, optimum)
            if len(self.q_states) == self.n:
                if self.verbose:
                    print(f"All qubits locked at iteration {iteration}")
                break

        # Final state: locked values, otherwise the last measured majority
        final_state_bits = []
        for qubit_idx in range(self.n):
            if qubit_idx in self.q_states:
                final_state_bits.append("0" if self.q_states[qubit_idx][0] == 1.0 else "1")
            else:
                last = self.qubit_prob_history[-1][qubit_idx + 1] if self.qubit_prob_history else [1, 0]
                final_state_bits.append("0" if not (last[1] > last[0]) else "1")
        final_bin = "".join(final_state_bits)
        final_state = int(final_bin, 2)
        final_energy = float(vqpm_root.qubo_energies(self.n, self._Q)[final_state])
        return {
            "final_state": final_state,
            "final_state_binary": final_bin,
            "final_energy": final_energy,
            "iterations": iteration + 1,
            "qubit_prob_history": self.qubit_prob_history,
            "prepared_history": self.prepared_history,
            "locked_qubits": self.q_states,
            "lock_log": self.lock_log,
            "shots_used": self.shots_used,
            "postselected_shots": self.postselected_shots,
            "seconds": time.time() - t0,
            "Q": self._Q,
        }


def run_original_simulation(Q, n, max_iter=30, pdiff=0.01, precision=3):
    """Exact (state-vector) VQPM with per-iteration qubit probabilities."""
    from vqpm_prob_tracking import adjust_unitary_phase, vqpm_with_prob_tracking

    Q_adj, u, phases, target_state = adjust_unitary_phase(n, Q)
    result_state, max_prob, q_states, iters, probs, qubit_prob_history = vqpm_with_prob_tracking(
        u, n, target_state=target_state, max_iter=max_iter, pdiff=pdiff, precision=precision
    )
    return {
        "final_state": int(result_state),
        "final_state_binary": format(int(result_state), f"0{n}b"),
        "target_state": int(target_state),
        "target_binary": format(int(target_state), f"0{n}b"),
        "success_probabilities": probs,
        "iterations": iters,
        "max_prob": max_prob,
        "Q": Q,
        "qubit_prob_history": qubit_prob_history,
        "locked_qubits": q_states,
    }


def run_quantum_simulation(Q, n, max_iter=30, pdiff=0.01, shots_per_qubit=500, precision=3,
                           **kwargs):
    """Run the circuit simulation (kwargs go to QuantumCircuitVQPM)."""
    quantum_vqpm = QuantumCircuitVQPM(n_qubits=n, max_iter=max_iter, pdiff=pdiff,
                                      precision=precision, **kwargs)
    optimum = int(np.argmin(vqpm_root.qubo_energies(n, Q)))
    return quantum_vqpm.run_vqpm(Q, shots=shots_per_qubit, optimum=optimum)


def compare_qubit_probabilities(original_probs, quantum_probs, n_qubits, axes=None):
    """Per-qubit P(1) trajectories: exact simulation vs circuit."""
    import matplotlib.pyplot as plt

    min_iters = min(len(original_probs), len(quantum_probs))
    if axes is None:
        fig, axes = plt.subplots(2, (n_qubits + 1) // 2, figsize=(15, 8))
    else:
        fig = None
    axes = np.ravel(axes)
    for qubit in range(1, n_qubits + 1):
        ax = axes[qubit - 1]
        orig_prob_1 = [original_probs[i][qubit][1] for i in range(min_iters)]
        quant_prob_1 = [quantum_probs[i][qubit][1] for i in range(min_iters)]
        ax.plot(range(min_iters), orig_prob_1, "o-", label="Exact", alpha=0.7)
        ax.plot(range(min_iters), quant_prob_1, "s-", label="Circuit", alpha=0.7)
        ax.set_title(f"Qubit {qubit}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("P(1)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    for i in range(n_qubits, len(axes)):
        axes[i].set_visible(False)
    if fig is not None:
        fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=12, help="number of QUBO variables")
    ap.add_argument("--seed", type=int, default=42, help="np.random.seed for random_qubo")
    ap.add_argument("--sim-seed", type=int, default=42)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--pdiff", type=float, default=0.1)
    ap.add_argument("--precision", type=int, default=4)
    ap.add_argument("--shots", type=int, default=100000,
                    help="shots per circuit (per iteration in 'all' mode)")
    ap.add_argument("--noise", default="none",
                    help="none | fake:<FakeBackend> | depol:p1,p2,readout")
    ap.add_argument("--measure-mode", choices=["all", "per_qubit"], default="all")
    ap.add_argument("--locking-rule", choices=["pdiff", "robust", "hoeffding_union"],
                    default="pdiff")
    ap.add_argument("--gate-counts", type=int, nargs="*", default=None,
                    help="only report transpiled controlled-U gate counts for these n")
    ap.add_argument("--gate-count-backends", nargs="+", default=["all-to-all", "FakeTorino"])
    ap.add_argument("--out", default=None, help="JSON output path")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    if args.gate_counts is not None:
        rows = gate_counts(args.gate_counts or (8, 12, 16), args.gate_count_backends)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(rows, f, indent=1)
        return

    from vqpm_prob_tracking import random_qubo

    n = args.n
    np.random.seed(args.seed)
    Q = random_qubo(n)
    E = vqpm_root.qubo_energies(n, Q)

    print("VQPM Quantum Circuit Implementation")
    print("=" * 50)
    print(f"Parameters: n={n}, max_iter={args.max_iter}, pdiff={args.pdiff}, "
          f"precision={args.precision}, shots={args.shots}, noise={args.noise}")

    original = run_original_simulation(Q, n, max_iter=args.max_iter, pdiff=args.pdiff,
                                       precision=args.precision)
    quantum = run_quantum_simulation(
        Q, n, max_iter=args.max_iter, pdiff=args.pdiff, shots_per_qubit=args.shots,
        precision=args.precision, rand_seed=args.sim_seed, noise=args.noise,
        measure_mode=args.measure_mode, locking_rule=args.locking_rule)

    print("\nORIGINAL SIMULATION:")
    print(f"  Final state: {original['final_state']} ({original['final_state_binary']})")
    print(f"  Target state: {original['target_state']} ({original['target_binary']})")
    print(f"  Iterations: {original['iterations']}")
    print("\nQUANTUM CIRCUIT SIMULATION:")
    print(f"  Final state: {quantum['final_state']} ({quantum['final_state_binary']})")
    print(f"  Iterations: {quantum['iterations']}  shots used: {quantum['shots_used']}")
    print(f"  Locked qubits: {len(quantum['locked_qubits'])}/{n}")
    print(f"\nQUBO values: target {E.min():.4f}  original {E[original['final_state']]:.4f}  "
          f"circuit {E[quantum['final_state']]:.4f}")

    if args.out:
        out = {k: v for k, v in quantum.items() if k not in ("Q", "locked_qubits")}
        out["args"] = vars(args)
        out["optimum"] = int(np.argmin(E))
        out["original_final_state"] = original["final_state"]
        out["original_qubit_prob_history"] = original["qubit_prob_history"]
        with open(args.out, "w") as f:
            json.dump(out, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    if not args.no_plot:
        import matplotlib.pyplot as plt

        compare_qubit_probabilities(original["qubit_prob_history"],
                                    quantum["qubit_prob_history"], n)
        plt.show()


if __name__ == "__main__":
    main()
