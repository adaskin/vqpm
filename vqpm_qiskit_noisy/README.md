# Qiskit Aer (Noisy) Implementation of Variational Quantum Power Method (VQPM)

The implementation features a **high-pdiff strategy** (0.1) that delays qubit locking until probability differences are more reliable, making it particularly suitable for noisy quantum environments.

## Key Features
- **Quantum Circuit Implementation**: Full quantum circuit realization of VQPM using Qiskit
- **High-pdiff Strategy**: Uses `pdiff=0.1` to minimize premature locking in noisy conditions
- **Individual Qubit Measurement**: Measures each qubit separately with post-selection on control=0
- **Comparison Tool**: Compares quantum circuit results with original classical VQPM simulation
- **Probability Tracking**: Tracks qubit probabilities across iterations for analysis

## Algorithm Description
VQPM adapts the classical power iteration method to quantum hardware:
1. **Amplitude Amplification**: Uses `(I+U)^k` to amplify the dominant eigenstate (optimal solution)
2. **Individual Qubit Measurement**: Measures each qubit with Hadamard-based amplitude estimation
3. **Delayed Locking Strategy**: With `pdiff=0.1`, qubits are locked only when probability differences are substantial
4. **Iterative Refinement**: Continues until all qubits are locked or max iterations reached

## Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `n` | 11 | Number of qubits (problem size) |
| `max_iter` | 100 | Maximum iterations for convergence |
| `pdiff` | **0.1** | High threshold for qubit locking |
| `precision` | 4 | Decimal precision for probability rounding |
| `shots_per_qubit` | 100,000 | Shots per qubit measurement |

## Why High pdiff (0.1)?
The high `pdiff` value serves several purposes:
- **Noise Resilience**: Requires clearer probability differences before locking, reducing false positives
- **Delayed Decision Making**: Allows more iterations for amplitude amplification before committing to qubit values
- **Statistical Significance**: Ensures locking decisions are based on reliable probability estimates
- **Adaptive Convergence**: Works well with high shot counts (100k shots) for accurate probability estimation

## Usage
```python
from vqpm_qiskit import run_quantum_simulation, run_original_simulation

# Generate random QUBO problem
import numpy as np
from vqpm_prob_tracking import random_qubo
n = 11
Q = random_qubo(n)

# Run quantum circuit VQPM with high pdiff
results = run_quantum_simulation(
    Q=Q,
    n=n,
    max_iter=100,
    pdiff=0.1,
    shots_per_qubit=100000,
    precision=4
)

# Compare with original VQPM
original_results = run_original_simulation(
    Q=Q,
    n=n,
    max_iter=100,
    pdiff=0.1,
    precision=4
)
```

## Output
The code provides:
1. **Iteration-by-iteration progress**: Locked qubits count per iteration
2. **Final solution**: Binary string and corresponding QUBO value
3. **Probability tracking**: P(1) for each qubit across iterations
4. **Comparison plots**: 
   - Qubit probability evolution (original vs quantum)
   - Energy comparison bar chart
   - Convergence visualization

## Dependencies
- Python 3.8+
- Qiskit 0.45+
- Qiskit Aer
- NumPy
- Matplotlib
- Additional local module: `vqpm_prob_tracking`

## File Structure
```
vqpm_qiskit.py          # Main quantum circuit implementation
vqpm_prob_tracking.py   # Original VQPM simulation (imported)
README.md              # This file
```

## Key Functions
- `QuantumCircuitVQPM`: Main class implementing quantum circuit VQPM
- `create_qubo_unitary_circuit`: Builds QUBO phase unitary
- `measure_single_qubit_probability`: Measures individual qubits with post-selection
- `run_vqpm`: Executes the full VQPM algorithm
- `compare_qubit_probabilities`: Creates comparison plots

## Noise Considerations
The implementation is designed with noisy quantum hardware in mind:
- **High shot count**: 100k shots per qubit for statistical accuracy
- **Post-selection**: Uses control=0 measurements to filter out noise
- **Rounding precision**: 4 decimal places to mitigate floating-point errors
- **Robust locking**: Statistical margin based on shot count variance

## Applications
- **QUBO Problem Solving**: Max-cut, graph partitioning, combinatorial optimization
- **Quantum Algorithm Research**: VQPM implementation and benchmarking
- **Noise Resilience Studies**: High-pdiff strategy for noisy quantum environments
- **Educational Tool**: Learning quantum power methods and variational algorithms


