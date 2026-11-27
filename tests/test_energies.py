"""Vectorised QUBO energies vs the original Python double loop."""

import numpy as np
import pytest

import qubo
import vqpm


def loop_energies(n, Q):
    """The original implementation (vqpm.py@d1e1d24, test_u_for_qubo)."""
    vals = np.zeros(2**n)
    for k in range(2**n):
        b = bin(k)[2:].zfill(n)
        total = 0.0
        for i in range(n):
            for j in range(i, n):
                if b[i] == "1" and b[j] == "1":
                    total += Q[i][j]
        vals[k] = total
    return vals


def brief_formula(n, Q):
    k = np.arange(2**n)
    B = ((k[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1).astype(float)
    return np.sum((B @ np.triu(Q)) * B, axis=1)


@pytest.mark.parametrize("n", range(1, 9))
def test_sequential_is_bitwise_equal_to_loop(n):
    rng = np.random.default_rng(n)
    for _ in range(5):
        Q = rng.standard_normal((n, n))
        Q = Q + Q.T
        ref = loop_energies(n, Q)
        assert np.array_equal(vqpm.qubo_energies(n, Q), ref)
        assert np.array_equal(vqpm.test_u_for_qubo(n, Q), ref)
        assert np.array_equal(vqpm.unitary_for_qubo(n, Q), np.exp(1j * ref))
        # non-symmetric input: only the upper triangle is used, as before
        A = rng.standard_normal((n, n))
        assert np.array_equal(vqpm.qubo_energies(n, A), loop_energies(n, A))


@pytest.mark.parametrize("n", range(1, 9))
def test_matmul_formula_matches_loop(n):
    rng = np.random.default_rng(100 + n)
    Q = rng.standard_normal((n, n))
    Q = Q + Q.T
    ref = loop_energies(n, Q)
    assert np.allclose(brief_formula(n, Q), ref, atol=1e-12)
    assert np.allclose(vqpm.qubo_energies(n, Q, method="matmul"), ref, atol=1e-12)


def test_chunking_does_not_change_values():
    rng = np.random.default_rng(7)
    n = 10
    Q = rng.standard_normal((n, n))
    for method in ("sequential", "matmul"):
        a = vqpm.qubo_energies(n, Q, method=method)
        b = vqpm.qubo_energies(n, Q, method=method, chunk_bits=4)
        assert np.array_equal(a, b)


def test_bit_convention_msb_first():
    n = 4
    Q = np.diag([1.0, 2.0, 4.0, 8.0])
    E = vqpm.qubo_energies(n, Q)
    assert E[0b1000] == 1.0  # bit 0 = most significant
    assert E[0b0001] == 8.0
    assert np.array_equal(vqpm.bit_matrix(3)[0b110], [1.0, 1.0, 0.0])


def test_qubo_module_uses_same_functions():
    rng = np.random.default_rng(3)
    n = 6
    Q = rng.standard_normal((n, n))
    Q = Q + Q.T
    assert np.array_equal(qubo.test_u_for_qubo(n, Q), loop_energies(n, Q))
    a = qubo.adjust_unitary_phase(n, Q)
    b = vqpm.adjust_unitary_phase(n, Q)
    assert np.array_equal(a[1], b[1]) and a[3] == b[3]
