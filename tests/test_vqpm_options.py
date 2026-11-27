"""Tests of the new (off-by-default) VQPM options, n <= 8."""

import contextlib
import io

import numpy as np
import pytest

import vqpm
import vqpm_d1e1d24 as ref
from common import derive_result, generate_instance


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def instance(n, i=0, seed=0):
    Q = generate_instance(n, seed, i)
    _, u, phases, target = vqpm.adjust_unitary_phase(n, Q)
    return Q, u, target


# ------------------------------------------------------------- defaults
@pytest.mark.parametrize("policy", [None, "hoeffding", "scaledby2"])
@pytest.mark.parametrize("locking", ["yes", "no", "only_when_p0_increases"])
def test_default_path_identical_to_published(policy, locking):
    for seed in range(6):
        n = 3 + seed
        np.random.seed(seed)
        Q = ref.random_qubo(n)
        a = ref.adjust_unitary_phase(n, Q)
        b = vqpm.adjust_unitary_phase(n, Q)
        assert np.array_equal(a[1], b[1]) and a[3] == b[3]
        kw = dict(expected_idx=a[3], max_iter=40, pdiff=0.01, precision=3,
                  dynamic_pdiff_policy=policy, locking=locking)
        out_a, out_b = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out_a):
            ra = ref.vqpm_for_qubo(a[1], n, **kw)
        with contextlib.redirect_stdout(out_b):
            rb = vqpm.vqpm_for_qubo(b[1], n, **kw)
        assert len(rb) == 6
        assert ra[0] == rb[0] and ra[1] == rb[1] and ra[3] == rb[3] and ra[4] == rb[4]
        assert np.array_equal(ra[5], rb[5])
        assert out_a.getvalue() == out_b.getvalue()


def test_return_log_appends_info_only():
    n = 6
    Q, u, t = instance(n)
    r6 = quiet(vqpm.vqpm_for_qubo, u, n, expected_idx=t, max_iter=30)
    r7 = quiet(vqpm.vqpm_for_qubo, u, n, expected_idx=t, max_iter=30, return_log=True)
    assert len(r6) == 6 and len(r7) == 7
    for a, b in zip(r6, r7[:6]):
        if isinstance(a, np.ndarray):
            assert np.array_equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
        else:
            assert a == b
    info = r7[6]
    for key in ("lock_log", "iterations", "stop_reason", "shots_used", "stalled"):
        assert key in info


def test_verbose_false_is_silent_and_same():
    n = 5
    Q, u, t = instance(n, 1)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = vqpm.vqpm_for_qubo(u, n, expected_idx=t, max_iter=30, verbose=False,
                               dynamic_pdiff_policy="hoeffding")
    assert buf.getvalue() == ""
    r2 = quiet(vqpm.vqpm_for_qubo, u, n, expected_idx=t, max_iter=30,
               dynamic_pdiff_policy="hoeffding")
    assert r[0] == r2[0] and np.array_equal(r[5], r2[5])


# ------------------------------------------------------------- phase maps
@pytest.mark.parametrize("pm", vqpm.PHASE_MAPS)
def test_phase_maps(pm):
    lo, hi = {"paper": (0, np.pi / 2), "tight": (0, np.pi / 2),
              "tight_wide": (0, 0.9 * np.pi), "shifted": (np.pi / 2, np.pi)}[pm]
    for i in range(5):
        n = 4 + i
        Q = generate_instance(n, 1, i)
        E = vqpm.qubo_energies(n, Q)
        Qs, u, phases, target = vqpm.adjust_unitary_phase(n, Q, phase_map=pm)
        a, b = vqpm.phase_map_coefficients(Q, pm)
        assert np.allclose(phases, a * E + b, atol=1e-12)
        assert phases.min() >= lo - 1e-12 and phases.max() <= hi + 1e-12
        assert target == np.argmin(E)
        assert np.allclose(u, np.exp(1j * phases))
        assert np.allclose(vqpm.qubo_energies(n, Qs), a * E)
    if pm == "tight":  # bounds attained when E(x) reaches L and U
        Q = np.array([[-1.0, -2.0], [-2.0, -3.0]])  # L = -6 at 11, U = 0 at 00
        _, _, ph, _ = vqpm.adjust_unitary_phase(2, Q, phase_map="tight")
        assert np.isclose(ph.min(), 0) and np.isclose(ph.max(), np.pi / 2)


def test_paper_phase_map_is_default():
    n = 6
    Q = generate_instance(n, 0, 3)
    a = vqpm.adjust_unitary_phase(n, Q)
    b = vqpm.adjust_unitary_phase(n, Q, phase_map="paper")
    c = ref.adjust_unitary_phase(n, Q)
    assert np.array_equal(a[1], b[1]) and np.array_equal(a[1], c[1])


# ------------------------------------------------------------- lock log
def test_lock_log_consistent_with_q_states():
    for i in range(10):
        n = 4 + i % 5
        Q, u, t = instance(n, i)
        r = quiet(vqpm.vqpm_for_qubo, u, n, expected_idx=t, max_iter=30, return_log=True)
        q_states, info = r[2], r[6]
        log = info["lock_log"]
        assert len(log) == len(q_states) == info["n_locked"]
        assert len({e["qubit"] for e in log}) == len(log)  # each qubit locks once
        for e in log:
            assert e["value"] == int(q_states[e["qubit"]][1] == 1.0)
            assert e["margin"] > e["threshold"]
            bit = (t >> (n - e["qubit"])) & 1
            assert e["matches_optimum"] == (bit == e["value"])
            assert 0 <= e["iteration"] < r[4] or info["stop_reason"] == "target_lost"


def test_lock_log_without_optimum():
    n = 5
    Q, u, t = instance(n)
    r = quiet(vqpm.vqpm_for_qubo, u, n, max_iter=30, return_log=True)
    assert all(e["matches_optimum"] is None for e in r[6]["lock_log"])


# ------------------------------------------------------------- stop rules
def test_max_prob_rule_needs_no_oracle():
    for i in range(10):
        n = 4 + i % 5
        Q, u, t = instance(n, i, seed=2)
        a = quiet(vqpm.vqpm_for_qubo, u, n, expected_idx=t, max_iter=50,
                  stop_rule="max_prob", return_log=True)
        b = quiet(vqpm.vqpm_for_qubo, u, n, expected_idx=None, max_iter=50,
                  stop_rule="max_prob", return_log=True)
        assert a[0] == b[0] and a[4] == b[4]  # expected_idx only for reporting
        info = a[6]
        assert info["stop_reason"] in ("max_prob", "all_locked", "max_iter")
        if info["stop_reason"] == "all_locked":
            assert info["n_locked"] == n and a[1] == 1.0
        if info["stop_reason"] == "max_prob":
            assert a[1] >= 0.5


def test_bad_stop_rule():
    Q, u, t = instance(4)
    with pytest.raises(ValueError):
        vqpm.vqpm_for_qubo(u, 4, stop_rule="nope", verbose=False)


@pytest.mark.parametrize("precision", [3, 6, None])
def test_trajectory_derivation_matches_direct_runs(precision):
    """One stop_rule='none' trajectory gives the outcome of every
    (stop rule, budget) pair, identical to running them directly."""
    for i in range(8):
        n = 4 + i % 5
        Q, u, t = instance(n, i, seed=3)
        E = vqpm.qubo_energies(n, Q)
        long = quiet(vqpm.vqpm_for_qubo, u, n, expected_idx=t, max_iter=60,
                     precision=precision, stop_rule="none", return_log=True,
                     record_trajectory=True, fast_forward_fixed_point=True, energies=E)
        info = long[6]
        for rule in ("oracle", "max_prob"):
            for budget in (5, 30, 60):
                d = derive_result(info["trajectory"], info["lock_log"], n, t, rule, budget)
                r = quiet(vqpm.vqpm_for_qubo, u, n, expected_idx=t, max_iter=budget,
                          precision=precision, stop_rule=rule, return_log=True, energies=E)
                if d["expected_energy_is_state"]:
                    assert np.isclose(E[d["found_state"]], r[6]["final_expected_energy"])
                else:
                    assert np.isclose(d["expected_energy"], r[6]["final_expected_energy"])
                assert d["found_state"] == int(r[0]), (i, rule, budget)
                assert d["iterations"] == r[6]["iterations"]
                assert d["stop_reason"] == r[6]["stop_reason"]
                assert d["n_locks"] == r[6]["n_locked"]
                assert np.isclose(d["target_prob"], r[5][-1]) or r[6]["stop_reason"] == "target_lost"
                if d["stop_reason"] == "max_iter":
                    assert d["stall_length"] == r[6]["stall_length"]


def test_fast_forward_gives_identical_results():
    for i in range(10):
        n = 4 + i % 5
        Q, u, t = instance(n, i, seed=4)
        kw = dict(expected_idx=t, max_iter=200, stop_rule="none", return_log=True)
        a = quiet(vqpm.vqpm_for_qubo, u, n, **kw)
        b = quiet(vqpm.vqpm_for_qubo, u, n, fast_forward_fixed_point=True, **kw)
        assert a[0] == b[0] and a[1] == b[1] and a[4] == b[4]
        assert np.array_equal(a[5], b[5])
        assert a[6]["stall_length"] == b[6]["stall_length"]
        assert a[6]["stalled"] == b[6]["stalled"]


# ------------------------------------------------------------- iteration budget
def test_resolve_max_iter():
    assert vqpm.resolve_max_iter(30, 7) == 30
    assert vqpm.resolve_max_iter("10n", 7) == 70
    assert vqpm.resolve_max_iter("2.5n", 7) == 18
    assert vqpm.resolve_max_iter("3*n", 4) == 12
    with pytest.raises(ValueError):
        vqpm.resolve_max_iter("abc", 4)
    Q, u, t = instance(5)
    r = quiet(vqpm.vqpm_for_qubo, u, 5, expected_idx=t, max_iter="2n",
              stop_rule="none", return_log=True)
    assert r[6]["max_iter"] == 10 and r[4] == 10


def test_precision_none_runs():
    n = 6
    Q, u, t = instance(n, 2)
    r = quiet(vqpm.vqpm_for_qubo, u, n, expected_idx=t, max_iter=100,
              precision=None, return_log=True)
    assert r[6]["iterations"] >= 1


# ------------------------------------------------------------- finite shots
def test_shots_reproducible_and_global_rng_untouched():
    n = 6
    Q, u, t = instance(n, 5)
    np.random.seed(123)
    state = np.random.get_state()[1].copy()
    a = quiet(vqpm.vqpm_for_qubo, u, n, expected_idx=t, max_iter=30, shots=2000,
              rng=7, return_log=True)
    assert np.array_equal(np.random.get_state()[1], state)
    b = quiet(vqpm.vqpm_for_qubo, u, n, expected_idx=t, max_iter=30, shots=2000,
              rng=np.random.default_rng(7), return_log=True)
    assert a[0] == b[0] and np.array_equal(a[5], b[5])
    info = a[6]
    assert info["shots_used"] == 2000 * info["iterations"]
    assert 0 < info["postselected_shots"] <= info["shots_used"]


def test_shots_converge_to_exact_marginals():
    """With many shots the sampled run follows the exact run."""
    agree = 0
    for i in range(10):
        n = 5
        Q, u, t = instance(n, i, seed=6)
        ex = quiet(vqpm.vqpm_for_qubo, u, n, expected_idx=t, max_iter=30,
                   precision=2, return_log=True)
        sa = quiet(vqpm.vqpm_for_qubo, u, n, expected_idx=t, max_iter=30,
                   precision=2, shots=10**8, rng=i, return_log=True)
        agree += ex[6]["found_state_exact"] == sa[6]["found_state_exact"]
    assert agree >= 9


def test_sampled_marginal_estimator_unbiased():
    """Check the post-selection/multinomial estimator on one iteration."""
    n = 4
    Q, u, t = instance(n, 0)
    exact = quiet(vqpm.vqpm_for_qubo, u, n, max_iter=1, stop_rule="none",
                  record_trajectory="full", return_log=True, locking="no")
    # exact first-iteration marginals of the output state
    psi = (1 + u) / 2 / np.sqrt(2**n)
    psi = psi / np.linalg.norm(psi)
    m_exact = np.array([vqpm.calculate_prob(psi, q)[1] for q in range(1, n + 1)])
    # sampled: precision None, locking 'no', so the prepared marginals are the estimates
    ests = []
    for s in range(200):
        r = quiet(vqpm.vqpm_for_qubo, u, n, max_iter=1, stop_rule="none",
                  record_trajectory="full", return_log=True, locking="no",
                  precision=None, shots=4000, rng=s)
        ests.append(r[6]["trajectory"]["marginals"][0])
    ests = np.array(ests)
    assert np.allclose(ests.mean(axis=0), m_exact, atol=4 * ests.std(axis=0).max() / np.sqrt(200) + 1e-3)
    assert exact[6]["shots_used"] == 0


def test_hoeffding_union():
    n = 6
    Q, u, t = instance(n, 1)
    with pytest.raises(ValueError):
        vqpm.vqpm_for_qubo(u, n, dynamic_pdiff_policy="hoeffding_union", verbose=False)
    # threshold formula
    thr = vqpm.hoeffding_union_threshold(10, 30, 0.05, 1000)
    assert np.isclose(thr, 2 * np.sqrt(np.log(2 * 10 * 30 / 0.05) / 2000))
    assert vqpm.hoeffding_union_threshold(10, 30, 0.05, 0) == np.inf
    assert np.isclose(vqpm.hoeffding_union_threshold(10, 30, 0.05, 1000, precision=3),
                      thr + 1e-3)
    r = quiet(vqpm.vqpm_for_qubo, u, n, expected_idx=t, max_iter=50, shots=20000,
              rng=1, dynamic_pdiff_policy="hoeffding_union", delta=0.05, return_log=True)
    for e in r[6]["lock_log"]:
        assert e["margin"] > e["threshold"] > 0.03  # 2*eps for M <= 20000


def test_old_hoeffding_docstring_claim():
    """delta_total * H_T: the old per-iteration split sums to about 2.0."""
    T, delta = 30, 0.5
    total = sum(delta / (T - i) for i in range(T))
    assert np.isclose(total, delta * sum(1 / k for k in range(1, T + 1)))
    assert 1.99 < total < 2.0
