# Variational Quantum Power Method (VQPM) for QUBO Problems

**The code used for the paper:** **From Theory to Practice: Analyzing VQPM for Quantum Optimization of QUBO Problem, Ammar Daskin, May 2025, [https://arxiv.org/abs/2505.12990](https://arxiv.org/abs/2505.12990)**

The main file for the simulation is `vqpm.y` which can be run alone. 

The other files are used to generate figures in the paper. 
- `run_multiple.py` has functions to generate multiple qubo instances and run them.
- `plot_qubit_vs_metrics.py` plots the results of multiple random trials.
- `comparison_to_qaoa.py` A simple comparison to qaoa. This implementation may not be a perfect implementation. The performance of the algorithm depends on many factors. Therefore, if you are going to use QAOA please refer to quantum software packages.


Below is the overall outline of the algorithm and the overal explanations for running the code (*I got from the DeepSeek-R1 by giving the simulation code, May 2025, private chats. The simulation code in many places is also written with the help of AI tools.*): 


### **Algorithmic Flowchart for VQPM**  
```plaintext
Start  
│  
├─ 1. Initialize:  
│   - Random QUBO matrix Q  
│   - Adjust Q's phase → Q_scaled  
│   - Construct diagonal unitary U from Q_scaled  
│   - Initialize state vector ψ₀ = uniform superposition  
│  
├─ 2. For each iteration (up to max_iter):  
│   │  
│   ├─ a. Apply variational circuit:  
│   │     ψ₁ = U ⋅ ψ₀  
│   │     ψ_final = (ψ₀ + ψ₁) / √2 (simulate I+U and normalize)  
│   │  
│   ├─ b. Calculate probabilities:  
│   │     p_min = max |ψ_final|²  
│   │     If p_min ≥ 0.5 → break (converged but may not be the target state)  
│   │  
│   ├─ c. Dynamic pdiff adjustment (if chosen):  
│   │     Update pdiff via Hoeffding bound:  
│   │     pdiff = √[ln(2/δ_i) / (2 ⋅ 10nM)]  
│   │  
│   ├─ d. Prepare new state:  
│   │     For each qubit q:  
│   │       Measure Prob(q=0) vs Prob(q=1)  
│   │       If |Prob₀ - Prob₁| ≥ pdiff → collapse qubit  
│   │       Else → keep superposition  
│   │     Update ψ₀ for next iteration  
│   │  
│   └─ e. Check convergence:  
│       If ψ₀ matches target state or target_state prob is became 0 → break  
│  
└─ 3. Output:  
    - Found state (binary string)  
    - Success probability  
    - Convergence metrics  
```  

## 🚀 Basic Usage
Run the script directly:
```bash
python vqpm.py
```
**Dependencies**
- Python 3.7+
- `numpy`
- `matplotlib`

Install dependencies via:
```bash
pip install numpy matplotlib
```

By default, it solves a **15-qubit QUBO problem** and generates two plots:
1. **Energy landscape** of QUBO values.
2. **Convergence progress** of success probabilities.
## 📝 Notes
- **Classical Simulation Complexity**: Runtime scales as \( O(2^n) \). The computations are vectorized but not parallized. For **\( n > 20 \)**, use a supercomputer or quantum hardware.
- **Reproducibility**: The script sets `np.random.seed(42)` by default. Remove this line for random results.
- **Locking Policies**: 
  - `'yes'`: Prefered, faster convergence (only use max 30 iterations).
  - `'no'`:  slows convergence more than 500 iterations.
---

## 🔧 Key Parameters (Customizable)

**Example customization:**
```python
# Inside vqpm.py's __main__ block:
n = 10  # Smaller problem for testing
result_state, max_prob, _, _, iters, probs = vqpm_for_qubo(
    u,
    n,
    max_iter=30,
    pdiff=0.02,
    precision=3,
    dynamic_pdiff_policy='none',
    locking='yes',
)
```
Adjust these in the `if __name__ == "__main__"` block of `vqpm.py`:

| Parameter                | Description                                                                 | Suggested Values       |
|--------------------------|-----------------------------------------------------------------------------|----------------------|
| `n`                      | Number of qubits (problem size)                                             | `n = 10`            |
| `max_iter`               | Maximum iterations for convergence                                          | `20`                |
| `pdiff`                  | Threshold for collapsing qubit probabilities                               | `0.01`              |
| `precision`               | Measurement precision(np.round()) of the qubits convergence                                          | `3`                |
| `dynamic_pdiff_policy`   | Strategy to adjust `pdiff` dynamically (`None`, `'hoeffding'`, `'scaledby2'`) | `'none'`       |
| `locking`                | Qubit state collapse policy (`'yes'`, `'no'`, `'only_when_p0_increases'`)   | `'yes'`             |
| `qubit_weights`                | scale pdiff with influences (`None`, a list). Use it only with hoeffding etc.   | `compute_qubit_weights(Q)`             |
|================|---------------------------------|----------------------|
| `shots_per_iter`         | Measurement shots per iteration (for Hoeffding policy)                      | `100` (adjustable inside hoeffding)               |
| `delta_total`            | Total allowed failure probability (Hoeffding bound)                         | `0.5` (adjustable inside hoeffding)        |


---

## ⚙️ Parameter Tuning Guide
-  Use `pdiff=0.01` and `max_iter=20`.
- **Locking Policies**:  
   - `'yes'`: Aggressively locks qubits once probabilities cross `pdiff`.  
    - If you use `'no'` then you should set at least `max_iter=500`

---
## 📊 Outputs
1. **Terminal Output**:
   - Expected vs. found solution states (binary strings).
   - QUBO values and success probabilities.
   - Example:
     ```
     ================= Problem Summary =================
     QUBO Matrix (first 5x5):
     [[ 1.624 -0.611 ... ]
      ...
     
     =============== Algorithm Results =================
     Expected state:   32768   000000000000001
     Found state   :   32768   000000000000001
     Diff bits     :       0   000000000000000

     QUBO Values:
     Expected: -5.4321 | Found: -5.4321
     
     Probabilities in the final state:
     Max probability:       0.5123
     Expected state prob:   0.5123
     ```

2. **Plots**:
   - `Energy Landscape`: Shows QUBO values for all states.
   - `Convergence Progress`: Tracks success probability vs. iterations.



---

## 📚 References
- From Theory to Practice: Analyzing VQPM for Quantum Optimization of QUBO Problem, Ammar Daskin, May 2025, [https://arxiv.org/abs/2505.12990](https://arxiv.org/abs/2505.12990).
-  Simulation code and this README file is written with the assistance from **DeepSeek/copilot**. 
