"""
QUBO (Quadratic Unconstrained Binary Optimization) module.
This module provides functions to generate random QUBO matrices,
create unitary matrices from QUBO matrices, and adjust phases for quantum computations.
@author: adaskin,
@date: 2025-05-18
"""

import numpy as np



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