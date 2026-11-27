"""Circuit VQPM under noise with the locking settings of the rest of the paper.

Run from the repository root (needs qiskit 2.6.2 and qiskit-aer 0.18.4, see requirements.txt):
    python3 theory_checks/noise_rerun.py > theory_checks/noise_rerun.out
Writes theory_checks/noise_rerun.jsonl (one record per run, resumable). Uses run_task of
experiments/noisy_runs.py unchanged (Aer density matrices, measure_mode 'all', marginals from the shots with
control 0, returned string = locked bits + majority of the last marginals), on the seed-2026 instances 0..4.

Settings: n = 8, 5 instances x 3 simulator seeds, at most 100 iterations, 1e5 shots per iteration unless noted.
  p_diff = 0.01, three decimals (the main setting of the paper): no noise, depolarizing (p1, p2, r) =
      (1e-5, 1e-4, 1e-3), (1e-4, 1e-3, 1e-2), (2.7e-4, 4.1e-3, 2.3e-2) (FakeTorino medians); the last also with
      1e6 shots; exact state-vector VQPM with the same settings.
  p_diff = 0.1, four decimals (the setting of experiments/reports/noise_sweep_n8.md): no noise and the
      FakeTorino medians, rerun here because the per-iteration marginals of those runs are not in the repository.
  n = 10: p_diff = 0.01, three decimals, no noise and the FakeTorino medians, 5 instances x 1 seed.
"""

import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import common  # noqa: E402
import noisy_runs  # noqa: E402

OUT = os.path.join(HERE, "noise_rerun.jsonl")
MED = "depol:2.7e-4,4.1e-3,2.3e-2"
SEED = 2026


def tasks():
    T = []

    def add(n, i, noise, s, pdiff, prec, shots=100000):
        kw = dict(n=n, i=i, count=5, seed=SEED, inst_dir=common.DEFAULT_INSTANCE_DIR, max_iter=100, pdiff=pdiff,
                  precision=prec)
        if noise == "exact":
            T.append(dict(kw, kind="exact", key="exact_n%d_i%d_pd%g_pr%d" % (n, i, pdiff, prec)))
        else:
            T.append(dict(kw, kind="circuit", noise=noise, sim_seed=s, shots=shots, locking_rule="pdiff", threads=1,
                          method="density_matrix",
                          key="circ_%s_n%d_i%d_s%d_sh%d_pd%g_pr%d" % (noise, n, i, s, shots, pdiff, prec)))

    for i in range(5):
        add(8, i, "exact", 0, 0.01, 3)
        add(10, i, "exact", 0, 0.01, 3)
        for s in range(3):
            for noise in ["none", "depol:1e-5,1e-4,1e-3", "depol:1e-4,1e-3,1e-2", MED]:
                add(8, i, noise, s, 0.01, 3)
            add(8, i, MED, s, 0.01, 3, shots=1000000)
            for noise in ["none", MED]:
                add(8, i, noise, s, 0.1, 4)
        for noise in ["none", MED]:
            add(10, i, noise, 0, 0.01, 3)
    return T


def run(task):
    rec = noisy_runs.run_task(task)
    rec.update({k: task[k] for k in ("pdiff", "precision")})
    return rec


if __name__ == "__main__":
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)["key"] for l in open(OUT)}
    todo = [t for t in tasks() if t["key"] not in done]
    for n in (8, 10):
        common.get_instances(n, 5, SEED)
    todo.sort(key=lambda t: t["n"])             # n = 8 first
    t0 = time.time()
    with Pool(int(os.environ.get("PROCS", os.cpu_count() or 1))) as pool, open(OUT, "a") as f:
        for k, rec in enumerate(pool.imap_unordered(run, todo)):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print("[%d/%d] %s %.0fs" % (k + 1, len(todo), rec["key"], time.time() - t0), file=sys.stderr, flush=True)
