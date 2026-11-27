"""QAOA options (optimizers, restarts, INTERP, counts), compare_algorithms
instance handling, and the simulated-annealing baseline."""

import contextlib
import io

import numpy as np
import pytest

import baselines
import comparison_to_qaoa as C
import comparison_to_qaoa_d1e1d24 as R
import vqpm
from common import generate_instance


def scaled(n, i=0, seed=0):
    Q = generate_instance(n, seed, i)
    Q_adj, u, _, t = vqpm.adjust_unitary_phase(n, Q)
    return Q, Q_adj, t


def test_hamiltonian_bitwise_equal_to_old_loop():
    for n in range(2, 8):
        Q = generate_instance(n, 1, n)
        assert np.array_equal(C.QAOA(Q).Hp, R.QAOA(Q).Hp)


def test_default_run_identical_to_published():
    for i in range(3):
        _, Qa, _ = scaled(6, i)
        np.random.seed(i)
        a = R.QAOA(Qa).run(p=3, maxiter=200)
        np.random.seed(i)
        b = C.QAOA(Qa).run(p=3, maxiter=200)
        assert a[0] == b[0] and np.array_equal(a[1], b[1])


def test_adjoint_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    for n, p in ((3, 1), (5, 3), (6, 4)):
        _, Qa, _ = scaled(n, p)
        q = C.QAOA(Qa)
        q.p = p
        x = rng.random(2 * p) * np.pi
        E, g = q._expectation_and_grad(x)
        assert np.isclose(E, q._expectation(x))
        h = 1e-6
        I = np.eye(2 * p)
        gfd = [(q._expectation(x + h * I[k]) - q._expectation(x - h * I[k])) / (2 * h)
               for k in range(2 * p)]
        assert np.allclose(g, gfd, atol=1e-8)


def test_interp_init_formula():
    # p = 2 -> 3: x'_k = (k/p) x_{k-1} + ((p-k)/p) x_k with x_{-1} = x_p = 0
    b, g = np.array([1.0, 2.0]), np.array([3.0, 5.0])
    out = C.interp_init(np.concatenate([b, g]), 2)
    assert np.allclose(out, [1.0, 1.5, 2.0, 3.0, 4.0, 5.0])


@pytest.mark.parametrize("optimizer", C.QAOA_OPTIMIZERS)
def test_optimizers_restarts_and_counts(optimizer):
    _, Qa, t = scaled(5, 1)
    q = C.QAOA(Qa)
    state, probs = q.run(p=2, maxiter=60, n_restarts=3, optimizer=optimizer, rng=0)
    lr = q.last_run
    assert lr["n_restarts"] == 3 and len(lr["restart_energies"]) == 3
    assert np.isclose(lr["energy"], min(lr["restart_energies"]))
    assert lr["nfev_total"] + lr["ngrad_total"] > 0
    assert lr["evals_fd_equiv_total"] >= lr["nfev_total"]
    assert state == int(np.argmax(probs)) and np.isclose(probs.sum(), 1)
    if optimizer == "COBYLA":
        assert lr["ngrad_total"] == 0
    if optimizer in ("gd", "adam"):
        assert lr["nfev_total"] == 0 and lr["ngrad_total"] <= 3 * 60
    # the energy reported is the expectation at the returned angles
    assert np.isclose(lr["energy"], np.dot(probs, q.Hp))


def test_seeded_restarts_are_distinct_and_reproducible():
    _, Qa, _ = scaled(5, 2)
    a = C.QAOA(Qa)
    a.run(p=2, maxiter=50, n_restarts=3, rng=5)
    b = C.QAOA(Qa)
    b.run(p=2, maxiter=50, n_restarts=3, rng=5)
    assert a.last_run["restart_energies"] == b.last_run["restart_energies"]
    assert len(set(np.round(a.last_run["restart_energies"], 10))) > 1


def test_interp_chain():
    _, Qa, _ = scaled(5, 3)
    q = C.QAOA(Qa)
    chain = q.run_interp(4, maxiter=100, n_restarts=2, gradient="adjoint", rng=1)
    assert sorted(chain) == [1, 2, 3, 4]
    costs = [chain[p]["evals_fd_equiv_total"] for p in range(1, 5)]
    assert costs == sorted(costs)  # cumulative cost of the chain
    assert chain[4]["energy"] <= chain[1]["energy"] + 1e-9
    st, pr = q.run(p=3, maxiter=100, n_restarts=2, init="interp", gradient="adjoint", rng=1)
    assert np.isclose(q.last_run["energy"], chain[3]["energy"])


def test_compare_algorithms_uses_given_instances():
    n, trials = 4, 3
    Qs = {i: generate_instance(n, 7, i) for i in range(trials)}
    with contextlib.redirect_stdout(io.StringIO()):
        res = C.compare_algorithms(trials=trials, n=n, p=1, trialQs=Qs)
    # the VQPM half is deterministic: recompute it from the given instances
    for i in range(trials):
        _, u, _, t = vqpm.adjust_unitary_phase(n, Qs[i])
        with contextlib.redirect_stdout(io.StringIO()):
            r = vqpm.vqpm_for_qubo(u, n, expected_idx=t, max_iter=30, pdiff=0.01,
                                   precision=3, locking="yes")
        assert res["final_prob_vqpm"][i] == r[5][-1]
    # without trialQs it no longer crashes
    with contextlib.redirect_stdout(io.StringIO()):
        C.compare_algorithms(trials=1, n=3, p=1, trialQs=None)


def test_compare_algorithms_legacy_instances_ignore_trialQs():
    n, trials = 4, 2
    np.random.seed(11)
    Qs = C.multiple_random_qubos(n, trials)
    with contextlib.redirect_stdout(io.StringIO()):
        res = C.compare_algorithms(trials=trials, n=n, p=1, trialQs=Qs,
                                   legacy_instances=True)
    # replay the published RNG sequence: Qs drawn, then per trial a fresh Q
    np.random.seed(11)
    C.multiple_random_qubos(n, trials)
    for i in range(trials):
        Q = vqpm.random_qubo(n)
        _, u, _, t = vqpm.adjust_unitary_phase(n, Q)
        with contextlib.redirect_stdout(io.StringIO()):
            r = vqpm.vqpm_for_qubo(u, n, expected_idx=t, max_iter=30, pdiff=0.01,
                                   precision=3, locking="yes")
        assert res["final_prob_vqpm"][i] == r[5][-1]
        np.random.rand(2 * 1)  # QAOA's initial angles


def test_simulated_annealing():
    for i in range(5):
        n = 8
        Q = generate_instance(n, 3, i)
        E = vqpm.qubo_energies(n, Q)
        r = baselines.simulated_annealing(Q, n_sweeps=300, n_restarts=2, rng=i)
        assert np.isclose(r["energy"], E[r["state"]])
        assert r["state"] == int(np.argmin(E))
        assert r["proposals"] == 2 * 300 * n
    # same bit convention as vqpm
    x = [1, 0, 1, 1]
    assert baselines.bits_to_state(x) == 0b1011
