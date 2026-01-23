import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakePeekskill
from vqpm_prob_tracking import vqpm_for_qubo, vqpm_with_prob_tracking, adjust_unitary_phase, random_qubo, test_u_for_qubo, calculate_prob


import warnings
warnings.filterwarnings('ignore')

class QuantumCircuitVQPM:
    def __init__(self, n_qubits, max_iter=30, pdiff=0.01, precision=3, rand_seed=42):
        self.n = n_qubits
        self.max_iter = max_iter
        self.pdiff = pdiff
        self.precision = precision # Decimal places for probability rounding
        self.simulator = AerSimulator(seed_simulator=rand_seed)
        # self.simulator = AerSimulator.from_backend(FakePeekskill())
        
        # Initialize state tracking
        self.q_states = {}
        self.current_angles = [np.pi/2] * self.n  # Start with Hadamard state (Ry(pi/2) = H)
        self.qubit_prob_history = []  # Track qubit probabilities over iterations
        
    def create_qubo_unitary_circuit(self, Q_scaled):
        """Create circuit that implements the QUBO unitary."""
        qr = QuantumRegister(self.n, 'q')
        qc = QuantumCircuit(qr)
        
        # Apply phase gates for each QUBO term
        # Single-qubit terms (diagonal)
        for i in range(self.n):
            if abs(Q_scaled[i,i]) > 1e-10:
                qc.rz(2 * Q_scaled[i,i], i)
        
        # Two-qubit terms (off-diagonal)
        for i in range(self.n):

            for j in range(i+1, self.n):
                if abs(Q_scaled[i,j]) > 1e-10:
                    # qc.cx(i, j)
                    qc.crz(2 * Q_scaled[i,j], j, i)
                    # qc.cx(i, j)
        
        return qc
    
    def create_single_qubit_measurement_circuit(self, angles, Q_scaled, target_qubit):
        """Create circuit to measure a single qubit with post-selection on control."""
        # Create registers: 1 control + n data qubits
        qr_control = QuantumRegister(1, 'control')
        qr_data = QuantumRegister(self.n, 'data')
        cr_control = ClassicalRegister(1, 'c_control')
        cr_target = ClassicalRegister(1, 'c_target')
        qc = QuantumCircuit(qr_control, qr_data, cr_control, cr_target)
        
        # Step 1: Prepare initial state on all data qubits
        for i, angle in enumerate(angles):
            if angle == 0.0:
                # Locked to |0⟩
                continue
            elif angle == np.pi:
                # Locked to |1⟩
                qc.x(qr_data[i])
            else:
                qc.ry(angle, qr_data[i])
        
        # Step 2: Apply first Hadamard to control qubit only
        qc.h(qr_control[0])
        
        # Step 3: Apply global phase pi/4 to control qubit after Hadamard
        qc.p(np.pi/4, qr_control[0])
        
        # Step 4: Controlled-QUBO unitary     

        for i in range(self.n):
            if abs(Q_scaled[i,i]) > 1e-10:
                qc.cp(Q_scaled[i,i], qr_control[0], qr_data[i])
        
        # Two-qubit terms (off-diagonal)
        for i in range(self.n):

            for j in range(i+1, self.n):
                if abs(Q_scaled[i,j]) > 1e-10:
                    qc.mcp(Q_scaled[i,j], [qr_control[0], qr_data[j]], qr_data[i])
 
        
        # Step 5: Apply second Hadamard to control qubit only
        qc.h(qr_control[0])
        
        # Step 6: Measure control qubit and target qubit only
        qc.measure(qr_control[0], cr_control[0])
        qc.measure(qr_data[target_qubit], cr_target[0])
        # Return a text drawing of the circuit.
        # Return a text drawing of the circuit.
        # print(qc.draw())
        return qc


    def robust_qubit_locking(self, prob_0, prob_1, qubit_idx, confidence=0.95):
        """Use statistical confidence intervals for locking"""
        std_err = np.sqrt(prob_0 * prob_1 / self.shots)
        margin = 2 * std_err  # ~95% confidence
        
        if prob_0 > prob_1 + self.pdiff + margin:
            return True, 0.0  # Lock to |0⟩
        elif prob_1 > prob_0 + self.pdiff + margin:
            return True, np.pi  # Lock to |1⟩
        else:
            return False, 2 * np.arcsin(np.sqrt(prob_1))

    def measure_single_qubit_probability(self, angles, Q_scaled, qubit_idx, shots_per_iter=1000):
        """Measure probability for a single qubit with post-selection on control=0."""
        circuit = self.create_single_qubit_measurement_circuit(angles, Q_scaled, qubit_idx)
        transpiled_circuit = transpile(circuit, self.simulator)
        
        # Run on quantum simulator
        job = self.simulator.run(transpiled_circuit, shots=shots_per_iter)
        result = job.result()
        counts = result.get_counts()
        # print(f"Qubit {qubit_idx}: Measurement counts: {counts}")
        # Parse results with post-selection on control=0
        count_0 = 0
        count_1 = 0
        total_post_selected = 0
        
        for bitstring, count in counts.items():
            # Bitstring format: 'target control' (due to measurement order)
            bitstring = bitstring.replace(' ', '')  # Remove spaces if any
            # print(f"Processing bitstring: {bitstring} with count: {count}")
            # print(len(bitstring))
            if len(bitstring) == 2:
                target_bit = bitstring[0]  # First measured (target)
                control_bit = bitstring[1]  # Second measured (control)
                
                if control_bit == '0':  # Post-select on control=0
                    total_post_selected += count
                    if target_bit == '0':
                        count_0 += count
                    else:
                        count_1 += count
        # print(f"Qubit {qubit_idx}: Post-selected counts - 0: {count_0}, 1: {count_1}, Total: {total_post_selected}")
        if total_post_selected > 0:
            prob_0 = count_0 / total_post_selected
            prob_1 = count_1 / total_post_selected
        else:
            prob_0 = 0.5
            prob_1 = 0.5
        print(f"Qubit {qubit_idx}: Probabilities - P(0): {prob_0}, P(1): {prob_1}") 
        return prob_0, prob_1, total_post_selected

    def update_angles_from_individual_measurements(self, angles, Q_scaled, shots_per_qubit=1000):
        """Update angles based on individual qubit measurements and track probabilities."""
        new_angles = angles.copy()
        iteration_probs = {}
        
        for qubit_idx in range(self.n):
            if qubit_idx in self.q_states:
                # Qubit is already locked - use locked probabilities
                state_q = self.q_states[qubit_idx]
                prob_0, prob_1 = state_q[0], state_q[1]
            else:
                # Measure this specific qubit
                prob_0, prob_1, total_shots = self.measure_single_qubit_probability(
                    angles, Q_scaled, qubit_idx, shots_per_qubit
                )
                
                # # Apply precision rounding
                prob_0 = round(prob_0, self.precision)
                prob_1 = round(prob_1, self.precision)

                # Re-normalize after rounding
                total = prob_0 + prob_1
                if total > 0:
                    prob_0 /= total
                    prob_1 /= total
                else:
                    prob_0 = 0.5
                    prob_1 = 0.5
            
            # Store probabilities for this iteration
            iteration_probs[qubit_idx + 1] = [prob_0, prob_1]
            
            # Check for locking (only for non-locked qubits)
            if qubit_idx not in self.q_states:
                if prob_0 > prob_1 + self.pdiff:
                    self.q_states[qubit_idx] = [1.0, 0.0]  # Lock to |0⟩
                    new_angles[qubit_idx] = 0.0
                elif prob_1 > prob_0 + self.pdiff:
                    self.q_states[qubit_idx] = [0.0, 1.0]  # Lock to |1⟩
                    new_angles[qubit_idx] = np.pi
                else:
                    # Update angle based on probability
                    new_angles[qubit_idx] = 2 * np.arcsin(np.sqrt(prob_1))
        
        # Store probability history for comparison
        self.qubit_prob_history.append(iteration_probs)
        
        return new_angles

    def run_vqpm(self, Q, shots_per_qubit=1000):
        """Run VQPM using individual qubit measurements."""
        # Scale Q matrix
        max_q = np.sum(np.abs(np.triu(Q)))
        Q_scaled = Q / max_q * np.pi / 4
        
        print(f"Running Quantum VQPM for {self.n} qubits")
        
        for iteration in range(self.max_iter):
            print(f"Iteration {iteration}: Locked qubits: {len(self.q_states)}/{self.n}")
            
            # Update angles based on individual qubit measurements
            self.current_angles = self.update_angles_from_individual_measurements(
                self.current_angles, Q_scaled, shots_per_qubit
            )
            
            # Check convergence - break if all qubits are locked
            if len(self.q_states) == self.n:
                print(f"All qubits locked at iteration {iteration}")
                break
        
        # Determine final state from individual qubit measurements
        final_state_bits = []
        for qubit_idx in range(self.n):
            if qubit_idx in self.q_states:
                # Use locked value
                if self.q_states[qubit_idx][0] == 1.0:
                    final_state_bits.append('0')
                else:
                    final_state_bits.append('1')
            else:
                # Use last measured probability
                last_probs = self.qubit_prob_history[-1][qubit_idx + 1]
                if last_probs[0] > last_probs[1]:
                    final_state_bits.append('0')
                else:
                    final_state_bits.append('1')
        
        final_bin = ''.join(final_state_bits)
        final_state = int(final_bin, 2)
        
        # Calculate final energy
        final_energy = 0.0
        for i in range(self.n):
            for j in range(i, self.n):
                if final_bin[i] == "1" and final_bin[j] == "1":
                    final_energy += Q[i][j]
        
        return {
            'final_state': final_state,
            'final_state_binary': final_bin,
            'final_energy': final_energy,
            'iterations': iteration + 1,
            'qubit_prob_history': self.qubit_prob_history,
            'locked_qubits': self.q_states,
            'Q': Q
        }




def run_original_simulation(Q, n, max_iter=30, pdiff=0.01, precision=3):
    """Run the original simulation for comparison and track qubit probabilities."""
    print("=" * 60)
    print("RUNNING ORIGINAL VQPM SIMULATION")
    print("=" * 60)
    
    #Prepare phase-adjusted unitary using original function
    Q_adj, u, phases, target_state = adjust_unitary_phase(n, Q)
    # Run original VQPM
    result_state, max_prob, q_states, iters, probs, qubit_prob_history = vqpm_with_prob_tracking(
        u, n, target_state=target_state, max_iter=max_iter, pdiff=pdiff, precision=precision
    )
    
    target_bin = format(target_state, f'0{n}b')
    result_bin = format(result_state, f'0{n}b')
    
    return {
        'final_state': result_state,
        'final_state_binary': result_bin,
        'target_state': target_state,
        'target_binary': target_bin,
        'success_probabilities': probs,
        'iterations': iters,
        'max_prob': max_prob,
        'Q': Q,
        'qubit_prob_history': qubit_prob_history,
        'locked_qubits': q_states
    }


def run_quantum_simulation(Q, n, max_iter=30, pdiff=0.01,shots_per_qubit=500, precision=3):
    """Run quantum circuit simulation"""
    print("=" * 60)
    print("RUNNING QUANTUM CIRCUIT VQPM SIMULATION") 
    print("=" * 60)
    
    # Initialize quantum VQPM
    quantum_vqpm = QuantumCircuitVQPM(
        n_qubits=n, max_iter=max_iter, pdiff=pdiff, precision=precision
    )
    
    # Generate QUBO problem using original function
    # Q = random_qubo(n)
    
    # Run quantum VQPM
    results = quantum_vqpm.run_vqpm(Q, shots_per_qubit=shots_per_qubit)
    
    return results


def compare_qubit_probabilities(original_probs, quantum_probs, n_qubits):
    """Compare qubit probabilities between original and quantum simulations."""
    # Find the minimum number of iterations to compare
    min_iters = min(len(original_probs), len(quantum_probs))
    
    fig, axes = plt.subplots(2, (n_qubits + 1) // 2, figsize=(15, 8))
    axes = axes.flatten()
    
    for qubit in range(1, n_qubits + 1):
        ax = axes[qubit - 1]
        
        # Extract probabilities for this qubit
        orig_prob_1 = [original_probs[i][qubit][1] for i in range(min_iters)]
        quant_prob_1 = [quantum_probs[i][qubit][1] for i in range(min_iters)]
        
        ax.plot(range(min_iters), orig_prob_1, 'o-', label='Original', alpha=0.7)
        ax.plot(range(min_iters), quant_prob_1, 's-', label='Quantum', alpha=0.7)
        ax.set_title(f'Qubit {qubit}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('P(1)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_qubits, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig


n = 11  # Small number for clear visualization
max_iter = 100 # should be high enough to ensure convergence for large n
pdiff = 0.1 # this should be higher than at least noise level
precision = 4 # Decimal places for probability rounding(used in angle estimation)
shots_per_qubit = 100000 # Number of shots per qubit measurement: higher for better statistics

# # Generate QUBO problem using original function
np.random.seed(42)
Q = random_qubo(n)
utest = test_u_for_qubo(n, Q)




print("VQPM Quantum Circuit Implementation")
print("=" * 50)
print(f"Parameters: n={n}, max_iter={max_iter}, pdiff={pdiff}, precision={precision}")
print(f"Shots per qubit: {shots_per_qubit}")

# Run both simulations
original_results = run_original_simulation(Q, n, 
                                           max_iter=max_iter, 
                                           pdiff=pdiff, precision=precision)
print("\n")
quantum_results = run_quantum_simulation(Q, n, 
                                         max_iter=max_iter, 
                                         pdiff=pdiff, 
                                         shots_per_qubit=shots_per_qubit, 
                                         precision=precision)

# Print comparison
print("\n" + "=" * 60)
print("COMPARISON RESULTS")
print("=" * 60)

print("ORIGINAL SIMULATION:")
print(f"  Final state: {original_results['final_state']} ({original_results['final_state_binary']})")
print(f"  Target state: {original_results['target_state']} ({original_results['target_binary']})")
print(f"  Iterations: {original_results['iterations']}")
print(f"  Locked qubits: {len(original_results['locked_qubits'])}/{n}")

print("\nQUANTUM CIRCUIT SIMULATION:")
print(f"  Final state: {quantum_results['final_state']} ({quantum_results['final_state_binary']})")
print(f"  Iterations: {quantum_results['iterations']}")
print(f"  Locked qubits: {len(quantum_results['locked_qubits'])}/{n}")

print(f"\nQUBO Values:")
print(f"  Target: {np.min(utest):.4f}")
print(f"  Original found: {utest[original_results['final_state']]:.4f}")
print(f"  Quantum found: {utest[quantum_results['final_state']]:.4f}")

# Compare qubit probabilities
if original_results['qubit_prob_history'] and quantum_results['qubit_prob_history']:
    prob_fig = compare_qubit_probabilities(
        original_results['qubit_prob_history'],
        quantum_results['qubit_prob_history'],
        n
    )
    # plt.savefig("qubit_probability_comparison.png", dpi=100)
    # plt.savefig("qubit_probability_comparison.pdf", dpi=100)
    plt.show()

# Plot convergence of locked qubits
plt.figure(figsize=(10, 5))

# # Count locked qubits per iteration
# orig_locked = []
# for i, probs in enumerate(original_results['qubit_prob_history']):
#     locked_count = sum(1 for q in range(1, n+1) if q in original_results['locked_qubits'] and 
#                         (i == 0 or q not in [q for q, _ in original_results['qubit_prob_history'][i-1].items() 
#                         if original_results['qubit_prob_history'][i-1][q][0] in [0.0, 1.0]]))
#     orig_locked.append(locked_count)

# quant_locked = []
# for i, probs in enumerate(quantum_results['qubit_prob_history']):
#     locked_count = len(quantum_results['locked_qubits'])
#     quant_locked.append(locked_count)

# plt.subplot(1, 2, 1)
# plt.plot(range(len(orig_locked)), np.cumsum(orig_locked), 'o-', label='Original')
# plt.plot(range(len(quant_locked)), np.cumsum(quant_locked), 's-', label='Quantum')
# plt.xlabel('Iteration')
# plt.ylabel('Cumulative Locked Qubits')
# plt.title('Qubit Locking Progress')
# plt.legend()
# plt.grid(True, alpha=0.3)

# Energy comparison
plt.subplot(1, 2, 2)
methods = ['Target', 'Original', 'Quantum']
energies = [
    min(utest),
    utest[original_results['final_state']],
    utest[quantum_results['final_state']]
]
colors = ['green', 'blue', 'red']

bars = plt.bar(methods, energies, color=colors, alpha=0.7)
plt.ylabel('QUBO Value')
plt.title('Energy Comparison')

# Add value labels on bars
for bar, energy in zip(bars, energies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{energy:.4f}', ha='center', va='bottom')

plt.tight_layout()
# plt.savefig("vqpm_comparison_summary.png", dpi=100)
# plt.savefig("vqpm_comparison_summary.pdf", dpi=100)
plt.show()


