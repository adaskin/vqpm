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


# The QUBO helpers are implemented (vectorised) in vqpm.py; this module
# re-exports them so that old imports keep working with a single source.
from vqpm import (  # noqa: E402,F401
    PHASE_MAPS,
    adjust_unitary_phase,
    bit_matrix,
    phase_map_coefficients,
    qubo_energies,
    test_u_for_qubo,
    unitary_for_qubo,
)
