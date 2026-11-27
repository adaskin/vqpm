# VQPM as a Qiskit circuit (ideal and noisy)

`vqpm_qiskit.py` runs VQPM as a quantum circuit on the Qiskit Aer simulator, with or without a noise model, and
counts the gates of the transpiled controlled-U. The paper uses it for the circuit size (Sec. 5.2) and, through
`experiments/noisy_runs.py`, for the noise study (Sec. 5.3).

## The circuit

Each iteration

1. prepares the product state $\prod_q R_y(\theta_q)\ket{0}$ on the $n$ data qubits, with
   $\sin^2(\theta_q/2)=\pi_q$, the current marginal of qubit $q$;
2. applies a Hadamard gate on a control qubit, the controlled diagonal unitary
   $\mathcal U=\mathrm{diag}(e^{i\lambda(x)})$ with the phase map $\lambda(x)=\pi/4+(\pi/4)E(x)/S$, and a second
   Hadamard gate on the control;
3. measures the control and all data qubits, keeps the shots with control $0$, and takes the new marginals from
   them (rounded to `--precision` decimals);
4. locks a qubit when $|P_0-P_1|$ exceeds the threshold (`--locking-rule`).

The circuit is transpiled once with the $R_y$ angles as parameters, which are re-bound in every iteration. Data
qubit $i$ holds QUBO variable $i$ (bit $i$ of the state index, bit 0 the most significant, as in `vqpm.py`).

## Command line

```bash
# ideal simulator
python vqpm_qiskit_noisy/vqpm_qiskit.py --n 8 --pdiff 0.01 --precision 3 --shots 100000
# calibrated noise model of a fake IBM backend (with routing onto its coupling map)
python vqpm_qiskit_noisy/vqpm_qiskit.py --n 8 --noise fake:FakeTorino
# depolarizing errors p1 (1-qubit gates), p2 (2-qubit gates) and a symmetric readout error r
python vqpm_qiskit_noisy/vqpm_qiskit.py --n 8 --noise depol:0.00027,0.0041,0.023
# two-qubit gate count and depth of the transpiled controlled-U
python vqpm_qiskit_noisy/vqpm_qiskit.py --gate-counts 8 12 16 --gate-count-backends all-to-all FakeTorino FakeSherbrooke
```

| option | meaning | default |
|---|---|---|
| `--n` | number of QUBO variables | 12 |
| `--seed` | `np.random.seed` for `random_qubo` | 42 |
| `--sim-seed` | seed of the simulator | 42 |
| `--max-iter` | iteration budget | 100 |
| `--pdiff` | locking threshold | 0.1 |
| `--precision` | decimals of the marginals | 4 |
| `--shots` | shots per iteration (per qubit with `--measure-mode per_qubit`) | 100000 |
| `--noise` | `none`, `fake:<Backend>` (e.g. `fake:FakeTorino`) or `depol:p1,p2,r` | `none` |
| `--measure-mode` | `all`: one circuit per iteration measures every qubit; `per_qubit`: one circuit per data qubit | `all` |
| `--locking-rule` | `pdiff` (fixed threshold), `robust` (lock if $\lvert P_0-P_1\rvert > p_{diff}+2\sqrt{P_0P_1/M}$, with $M$ post-selected shots) or `hoeffding_union` (the union-bound threshold of `vqpm.py`) | `pdiff` |
| `--gate-counts N ...` | only count gates for these n | |
| `--gate-count-backends` | `all-to-all` and/or fake backend names for the gate counts | `all-to-all FakeTorino` |
| `--out` | JSON file for the results | |
| `--no-plot` | do not plot | |

The script also runs the state-vector VQPM of `vqpm_prob_tracking.py` on the same instance, prints both results
and, without `--no-plot`, plots the marginals of both against the iterations.

The density-matrix simulation is slow: one noisy iteration takes about 25 seconds at $n=10$ and 13 minutes at
$n=12$, and the routed FakeTorino circuit needs about 17 GB of memory at $n=10$.

## Use from Python

```python
import numpy as np
from vqpm import random_qubo
from vqpm_qiskit_noisy.vqpm_qiskit import QuantumCircuitVQPM

np.random.seed(0)
Q = random_qubo(8)
runner = QuantumCircuitVQPM(8, max_iter=100, pdiff=0.01, precision=3)
result = runner.run_vqpm(Q, shots=100000)        # dict: final_state_binary, final_energy, iterations, lock_log, ...
print(result["final_state_binary"], result["iterations"])
```

Several instances and simulator seeds, with summaries and plots: `experiments/noisy_runs.py` (see
`experiments/README.md`).

## Files

- `vqpm_qiskit.py`: the circuit, the simulator and noise models, the gate counts and the command line.
- `vqpm_prob_tracking.py`: a copy of the state-vector VQPM that records the marginals of every iteration, used for
  the comparison plot.
