"""Check that the default code path still reproduces published numbers.

Checks (seeded with the legacy global NumPy RNG, exactly as the original scripts):

* ``fig6a``   -- 100 random QUBOs, n=15, pdiff=0.01, precision=3, max_iter=30,
                 locking='yes' after ``np.random.seed(42)``.
                 Published mean bit difference: 1.16.
* ``scaling`` -- VQPM half of the VQPM-vs-QAOA scaling figure
                 (``comparison_to_qaoa.compare_scaling_across_n``): seed 42,
                 n = 4, 6, ..., 16, 10 instances each. QAOA is not run; its RNG
                 consumption (``np.random.rand(2*p)`` for p in 2, 4, 6, 8) is
                 mimicked so the QUBO instances are the published ones.
                 Published mean final target probability: 0.46, 0.40, 0.18 at
                 n = 12, 14, 16.

Usage::

    python experiments/reproduce_published.py --impl current   # vqpm.py
    python experiments/reproduce_published.py --impl legacy    # frozen d1e1d24 copy
"""

import argparse
import contextlib
import importlib.util
import io
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PUBLISHED = {
    "fig9": {"vqpm_prob": 0.43, "qaoa_prob": 0.55, "vqpm_bits": 0.43, "qaoa_bits": 0.33},
    "fig6a_mean_bit_diff": 1.16,
    "scaling_target_prob": {12: 0.46, 14: 0.40, 16: 0.18},
}


def load_impl(name):
    if name == "current":
        import vqpm

        return vqpm
    path = os.path.join(ROOT, "tests", "reference", "vqpm_d1e1d24.py")
    spec = importlib.util.spec_from_file_location("vqpm_d1e1d24", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bitdiff(a, b, n):
    return bin(int(a) ^ int(b))[2:].zfill(n).count("1")


def check_fig6a(mod, n=15, trials=100, seed=42):
    np.random.seed(seed)
    diffs, probs = [], []
    for _ in range(trials):
        Q = mod.random_qubo(n)
        _, u, _, target = mod.adjust_unitary_phase(n, Q)
        with contextlib.redirect_stdout(io.StringIO()):
            found, _, _, _, _, p = mod.vqpm_for_qubo(
                u, n, expected_idx=target, max_iter=30, pdiff=0.01, precision=3,
                dynamic_pdiff_policy="none", qubit_weights=None, locking="yes",
            )
        diffs.append(_bitdiff(found, target, n))
        probs.append(p[-1])
    return {"mean_bit_diff": float(np.mean(diffs)),
            "mean_final_target_prob": float(np.mean(probs)),
            "bit_diffs": diffs}


def check_scaling(mod, n_values=range(4, 17, 2), trials=10, seed=42,
                  p_values=(2, 4, 6, 8)):
    np.random.seed(seed)
    out = {}
    for n in n_values:
        diffs, probs = [], []
        for _ in range(trials):
            Q = mod.random_qubo(n)
            _, u, _, target = mod.adjust_unitary_phase(n, Q)
            with contextlib.redirect_stdout(io.StringIO()):
                found, _, _, _, _, p = mod.vqpm_for_qubo(
                    u, n, expected_idx=target, max_iter=30, pdiff=0.01,
                    precision=3, dynamic_pdiff_policy=None, qubit_weights=None,
                    locking="yes",
                )
            diffs.append(_bitdiff(found, target, n))
            probs.append(p[-1])
            for pp in p_values:  # QAOA.run draws np.random.rand(2 * p) per depth
                np.random.rand(2 * pp)
        out[int(n)] = {"mean_bit_diff": float(np.mean(diffs)),
                       "mean_target_prob": float(np.mean(probs))}
    return out


def check_fig9(seed=42, n=8, p=8, trials=100):
    """VQPM vs QAOA (p=8), n=8, 100 trials: comparison_to_qaoa.main1 without
    plotting (current code only; QAOA uses the published L-BFGS-B setup)."""
    import comparison_to_qaoa as C

    np.random.seed(seed)
    trialQs = C.multiple_random_qubos(n, trials)
    with contextlib.redirect_stdout(io.StringIO()):
        res = C.compare_algorithms(trials=trials, n=n, p=p, trialQs=trialQs,
                                   legacy_instances=True)
    return {
        "vqpm_prob": float(np.mean(res["final_prob_vqpm"])),
        "qaoa_prob": float(np.mean(res["final_prob_qaoa"])),
        "vqpm_bits": float(np.mean(res["bit_diff_vqpm"])),
        "qaoa_bits": float(np.mean(res["bit_diff_qaoa"])),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impl", choices=["current", "legacy"], default="current")
    ap.add_argument("--checks", nargs="+", default=["fig6a", "scaling"],
                    choices=["fig6a", "scaling", "fig9"])
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    mod = load_impl(args.impl)
    res = {"impl": args.impl}
    ok = True
    if "fig6a" in args.checks:
        t = time.time()
        r = check_fig6a(mod)
        r["seconds"] = time.time() - t
        res["fig6a"] = r
        good = round(r["mean_bit_diff"], 2) == PUBLISHED["fig6a_mean_bit_diff"]
        ok &= good
        print(f"[fig6a] mean bit diff = {r['mean_bit_diff']:.4f} "
              f"(published {PUBLISHED['fig6a_mean_bit_diff']}) "
              f"{'OK' if good else 'MISMATCH'}  [{r['seconds']:.1f}s]")
    if "scaling" in args.checks:
        t = time.time()
        r = check_scaling(mod)
        res["scaling"] = r
        res["scaling_seconds"] = time.time() - t
        for n, v in r.items():
            pub = PUBLISHED["scaling_target_prob"].get(n)
            tag = ""
            if pub is not None:
                good = round(v["mean_target_prob"], 2) == pub
                ok &= good
                tag = f"(published {pub}) {'OK' if good else 'MISMATCH'}"
            print(f"[scaling] n={n:2d} mean target prob = {v['mean_target_prob']:.4f} "
                  f"mean bit diff = {v['mean_bit_diff']:.2f} {tag}")
        print(f"[scaling] [{res['scaling_seconds']:.1f}s]")
    if "fig9" in args.checks:
        if args.impl != "current":
            raise SystemExit("fig9 check needs --impl current")
        t = time.time()
        r = check_fig9()
        r["seconds"] = time.time() - t
        res["fig9"] = r
        for k, pub in PUBLISHED["fig9"].items():
            good = round(r[k], 2) == pub
            ok &= good
            print(f"[fig9] {k} = {r[k]:.4f} (published {pub}) {'OK' if good else 'MISMATCH'}")
        print(f"[fig9] [{r['seconds']:.1f}s]")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
