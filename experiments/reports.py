"""Reports for the experiments: provenance, data locations, method,
results and regeneration commands, one Markdown file per experiment under
experiments/reports/, plus INDEX.md. Also dated backup archives of the raw
data outside the repository.

    .venv/bin/python experiments/reports.py            # (re)write all reports
    .venv/bin/python experiments/reports.py --archive  # also archive raw data + instances

Experiment scripts call on_finish(<name>) when they end, which archives that
experiment's raw files and rewrites every report, so finished experiments
document themselves without anyone watching the runs. Hand-written findings
live in experiments/reports/findings/<name>.md and are included verbatim.
"""

import argparse
import csv
import datetime
import glob
import hashlib
import json
import os
import platform
import subprocess
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = os.path.join(ROOT, "experiments", "reports")
RES = os.path.join(ROOT, "experiments", "results")
INST = os.path.join(ROOT, "experiments", "instances")
LOGS = os.path.join(ROOT, "experiments", "logs")
# Dated archives of the raw data are kept outside the repository; set VQPM_BACKUP_DIR to choose where.
BACKUP = os.environ.get("VQPM_BACKUP_DIR", os.path.join(os.path.dirname(ROOT), "vqpm-backups"))
PY = ".venv/bin/python"

METRICS_TEXT = (
    "Metrics (per run, energies in the units of the original, unscaled Q): found optimum = the "
    "returned bitstring is the exact ground state (found by enumerating all 2^n energies); bit "
    "difference = Hamming distance to the ground state; energy gap = E_found - E_min; normalised "
    "energy gap = (E_found - E_min)/(E_max - E_min) in [0, 1]; normalised expected-energy gap = "
    "(<E> - E_min)/(E_max - E_min) with <E> the expected energy of the final output distribution "
    "(VQPM, QAOA). Intervals: Wilson 95% for rates, mean +- 1.96 s/sqrt(N) for means."
)

EXPERIMENTS = {
    "vqpm_grid": dict(
        title="Exp. 1: VQPM baseline grid (precision x iteration budget x stop rule)",
        purpose="Iteration budget vs rounding precision (the decline of the success rate with n "
                "under the 30-iteration cap); baseline numbers for the wrong-lock statistics "
                "and the oracle-free stop rule.",
        raw=["experiments/results/vqpm_grid/raw/grid.jsonl",
             "experiments/results/vqpm_grid/raw/grid_expected_energy.jsonl"],
        out_dir="experiments/results/vqpm_grid",
        script="experiments/vqpm_grid.py",
        regen=[f"{PY} experiments/vqpm_grid.py --plot-only --out experiments/results/vqpm_grid/raw/grid.jsonl",
               f"{PY} experiments/wrong_locks.py --grid experiments/results/vqpm_grid/raw/grid.jsonl"],
        method=(
            "For every n, instance (100 per n, seed 2026) and rounding precision of the marginals "
            "(3, 6 decimals, or exact = no rounding), VQPM (exact marginals, pdiff = 0.01, paper "
            "phase map) is run once with stop_rule='none' for the largest budget max(300, 10n), "
            "recording the trajectory (per iteration: target probability, largest basis-state "
            "probability and its index, number of locked qubits, whether the prepared state changed, "
            "and, for runs made after commit ed0b4c4 or backfilled, the expected energy) and the lock "
            "log. Outcomes for the stop rules 'oracle' (stop when P(target) >= 0.5, abort when it is 0) "
            "and 'max_prob' (stop when the largest probability >= 0.5 or all qubits are locked) and "
            "budgets 30, 100, 300, 10n are read off that trajectory (common.derive_result; tested to "
            "equal direct runs). Stall flag: the budget ran out and nothing changed in the last >= 10 "
            "iterations. Shots-equivalent (for the planned figure): resolving a marginal to d decimals "
            "needs about 10^(2d) shots per iteration (standard error 1/sqrt(M) ~ 10^-d), so "
            "shots_equiv_total = iterations x 10^(2d); undefined for exact marginals."),
        tables="grid",
    ),
    "wrong_locks": dict(
        title="Exp. 2: wrong-lock statistics (from the Exp. 1 lock logs)",
        purpose="How often VQPM locks a qubit to the wrong value, when, and with what margin.",
        raw=["experiments/results/vqpm_grid/raw/grid.jsonl"],
        out_dir="experiments/results/vqpm_grid",
        files_glob="wrong_locks*",
        script="experiments/wrong_locks.py",
        regen=[f"{PY} experiments/wrong_locks.py --grid experiments/results/vqpm_grid/raw/grid.jsonl"],
        method=("Every lock event of Exp. 1 is compared with the corresponding bit of the exact ground "
                "state (matches_optimum). Reported per precision and n over the whole recorded "
                "trajectory: wrong locks / all locks, fraction of runs with >= 1 wrong lock, median "
                "iteration of the first wrong lock, median margin |P0 - P1| of correct and wrong locks. "
                "The failure split per stop rule and budget (wrong lock / budget exhausted / stopped "
                "at a wrong state) is in vqpm_grid_summary.csv (fail_* columns)."),
        tables="wrong_locks",
    ),
    "phase_map": dict(
        title="Exp. 3: phase-map ablation",
        purpose="Sensitivity to the map from QUBO values to eigenphases.",
        raw=["experiments/results/phase_map/raw/phase_map.jsonl"],
        out_dir="experiments/results/phase_map",
        script="experiments/phase_map_ablation.py",
        regen=[f"{PY} experiments/phase_map_ablation.py --plot-only --out experiments/results/phase_map/raw/phase_map.jsonl"],
        method=("Maps (vqpm.phase_map_coefficients): paper E*pi/(4S)+pi/4 with S = sum|triu(Q)|; tight "
                "(E-L)/(U-L)*pi/2 with L, U the sums of the negative / positive entries of triu(Q); "
                "tight_wide (E-L)/(U-L)*0.9*pi; shifted E*pi/(4S)+3pi/4. n = 8..16 step 2, 100 "
                "instances, precision 3, pdiff 0.01, both stop rules, budgets 30 and 100 (one "
                "trajectory per run, read off as in Exp. 1)."),
        tables="phase_map",
    ),
    "finite_shots": dict(
        title="Exp. 4: finite shots",
        purpose="VQPM with marginals estimated from samples; fixed pdiff vs a valid "
                "union-bound Hoeffding locking rule; total shot cost.",
        raw=["experiments/results/finite_shots/raw/finite_shots.jsonl"],
        out_dir="experiments/results/finite_shots",
        script="experiments/finite_shots.py",
        regen=[f"{PY} experiments/finite_shots.py --plot-only --out experiments/results/finite_shots/raw/finite_shots.jsonl"],
        method=("Each iteration, M ~ Binomial(shots, P0) samples survive post-selection of the control "
                "on 0 and are drawn from p(x) cos^2(phase_x/2)/P0; qubit marginals are the sample "
                "frequencies (rounded to 3 decimals). Locking: fixed pdiff = 0.01, or hoeffding_union "
                "(lock only if |P0 - P1| > 2 eps + 10^-3, eps = sqrt(ln(2nT/delta)/(2M)), delta = 0.05, "
                "T = 100). n = 8..16 step 2, 100 instances, shots per iteration 1e3..1e6, both stop "
                "rules, max_iter 100. Returned state = most frequent sample of the last iteration (also "
                "reported: locked bits + marginal majority, and the exact argmax)."),
        tables="finite_shots",
    ),
    "qaoa_vs_vqpm": dict(
        title="Exp. 5: QAOA (4 optimizers) vs VQPM vs simulated annealing",
        purpose="A fair QAOA baseline (restarts, INTERP, several optimizers, evaluation counts).",
        raw=["experiments/results/qaoa_vs_vqpm/raw/qaoa_vs_vqpm.jsonl"],
        out_dir="experiments/results/qaoa_vs_vqpm",
        script="experiments/qaoa_vs_vqpm.py",
        regen=[f"{PY} experiments/qaoa_vs_vqpm.py --plot-only --out experiments/results/qaoa_vs_vqpm/raw/qaoa_vs_vqpm.jsonl"],
        method=("Statevector QAOA on the paper's scaled QUBO, p in {2, 4, 6, 8}, optimizers L-BFGS-B "
                "(exact adjoint gradients), COBYLA, gradient descent (lr 1) and Adam (lr 0.3), cap 500 "
                "iterations, 5 random restarts (best energy kept; same initial angles for every "
                "optimizer) and INTERP (chain 1..8, 5 random depth-1 starts). Evaluations: nfev + "
                "gradient evaluations x (2p+1) (finite-difference-equivalent). VQPM: precision 3, pdiff "
                "0.01, both stop rules, budgets 30 and 100. SA: 10, 100, 1000 sweeps. 100 instances for "
                "n <= 10, 50 for n = 12..16."),
        tables="qaoa",
    ),
    "noise_sweep_n8": dict(
        title="Exp. 6c: noise-strength sweep at n = 8 (Qiskit Aer, density matrix)",
        purpose="How much circuit noise VQPM tolerates.",
        raw=["experiments/results/noisy/raw/noise_sweep_n8.jsonl"],
        out_dir="experiments/results/noisy",
        files_glob="noise_sweep_n8*",
        script="experiments/noisy_runs.py",
        regen=[f"{PY} experiments/noisy_runs.py --plot-only --out experiments/results/noisy/raw/noise_sweep_n8.jsonl"],
        method="NOISY",
        tables="noisy",
    ),
    "faketorino_n8": dict(
        title="Exp. 6b: full FakeTorino noise model at n = 8",
        purpose="VQPM on a realistic (calibrated, routed) device model.",
        raw=["experiments/results/noisy/raw/faketorino_n8.jsonl"],
        out_dir="experiments/results/noisy",
        files_glob="faketorino_n8*",
        script="experiments/noisy_runs.py",
        regen=[f"{PY} experiments/noisy_runs.py --plot-only --out experiments/results/noisy/raw/faketorino_n8.jsonl"],
        method="NOISY",
        tables="noisy",
    ),
    "depol_torino": dict(
        title="Exp. 6a: Torino-calibrated depolarising + readout noise, n = 8, 10",
        purpose="VQPM under device-level error rates without routing overhead.",
        raw=["experiments/results/noisy/raw/depol_torino.jsonl"],
        out_dir="experiments/results/noisy",
        files_glob="depol_torino*",
        script="experiments/noisy_runs.py",
        regen=[f"{PY} experiments/noisy_runs.py --plot-only --out experiments/results/noisy/raw/depol_torino.jsonl"],
        method="NOISY",
        tables="noisy",
    ),
    "qaoa_optimizer_pilot": dict(
        title="Pilot: QAOA optimizer and learning-rate choice (n = 8, tuning instances)",
        purpose="Choose learning rates on separate instances; check L-BFGS-B vs COBYLA and "
                "adjoint vs finite-difference gradients.",
        raw=["experiments/results/qaoa_pilot/raw/pilot_lr_seed777.jsonl"],
        out_dir="experiments/results/qaoa_pilot",
        script="experiments/qaoa_vs_vqpm.py",
        seed=777,
        regen=[f"{PY} experiments/qaoa_vs_vqpm.py --plot-only --seed 777 --out experiments/results/qaoa_pilot/raw/pilot_lr_seed777.jsonl"],
        method=("20 instances (seed 777, not used elsewhere), n = 8, p = 4 and 8, 5 restarts, random and "
                "INTERP starts, cap 500. Optimizers: L-BFGS-B (adjoint and finite-difference gradients), "
                "COBYLA, gradient descent (lr 0.03 .. 10), Adam (lr 0.01 .. 0.3)."),
        tables="qaoa",
    ),
    "reproduction": dict(
        title="Reproduction of the original figures",
        purpose="Check that the code with default options reproduces the original figures made with "
                "plot_qubit_vs_metrics.py and comparison_to_qaoa.py.",
        raw=["experiments/results/reproduction/reproduce_published_current.json",
             "experiments/results/reproduction/reproduce_published_legacy_d1e1d24.json",
             "experiments/results/reproduction/reproduce_fig9_current.json"],
        out_dir="experiments/results/reproduction",
        script="experiments/reproduce_published.py",
        regen=[f"{PY} experiments/reproduce_published.py --impl current",
               f"{PY} experiments/reproduce_published.py --impl legacy",
               f"{PY} experiments/reproduce_published.py --impl current --checks fig9"],
        method=("Fig. 6a: np.random.seed(42), 100 random QUBOs (vqpm.random_qubo), n = 15, pdiff 0.01, "
                "precision 3, max_iter 30, locking 'yes'. Scaling figure (VQPM half): seed 42, n = 4..16 "
                "step 2, 10 instances, QAOA's RNG draws np.random.rand(2p) for p = 2, 4, 6, 8 mimicked. "
                "Fig. 9: comparison_to_qaoa.main1 without plotting (legacy_instances=True). 'legacy' runs "
                "the frozen published vqpm.py (tests/reference)."),
        tables="reproduction",
    ),
    "gate_counts": dict(
        title="Circuit cost: transpiled controlled-U and noisy-simulation feasibility",
        purpose="Hardware cost of one VQPM iteration.",
        raw=["experiments/results/gate_counts/controlled_u_gate_counts.json"],
        out_dir="experiments/results/gate_counts",
        script="vqpm_qiskit_noisy/vqpm_qiskit.py",
        regen=[f"{PY} vqpm_qiskit_noisy/vqpm_qiskit.py --gate-counts 8 12 16 --gate-count-backends "
               "all-to-all FakeTorino FakeSherbrooke --out experiments/results/gate_counts/controlled_u_gate_counts.json"],
        method=("Dense random QUBO (seed [0, n]); controlled-U = P(b) on the control, CP for diagonal "
                "terms, MCPhase (two controls) for each pair; transpiled with optimization_level 3 to "
                "{cx, rz, sx, x} without coupling map, and onto FakeTorino (CZ, heavy-hex) and "
                "FakeSherbrooke (ECR)."),
        tables="gate_counts",
    ),
}

SHORT = {
    "vqpm_grid": "iteration budget vs precision; baseline",
    "wrong_locks": "wrong locks",
    "phase_map": "phase maps",
    "finite_shots": "finite shots",
    "qaoa_vs_vqpm": "QAOA baseline",
    "noise_sweep_n8": "noise",
    "faketorino_n8": "noise",
    "depol_torino": "noise",
    "qaoa_optimizer_pilot": "optimizer choice",
    "reproduction": "original figures",
    "gate_counts": "circuit cost",
}

NOISY_METHOD = (
    "Circuit VQPM (vqpm_qiskit_noisy/vqpm_qiskit.py, measure_mode='all'): each iteration one circuit "
    "prepares prod_q Ry(theta_q)|0>, applies H - controlled-U - H on the control and measures the "
    "control and all data qubits; marginals come from the shots with control = 0. Aer density-matrix "
    "simulation (shot count nearly free), 1e5 shots per iteration, pdiff 0.1, precision 4, max_iter "
    "100, stop when all qubits are locked; returned state = locked bits + majority of the last "
    "marginals. 5 instances (seed 2026) x 3 simulator seeds. Noise: 'none' = ideal sampling; "
    "'depol:p1,p2,ro' = depolarising p1 on sx/x, p2 on CX, symmetric readout error ro, all-to-all "
    "(2.7e-4, 4.1e-3, 2.3e-2 are the FakeTorino medians); 'fake:FakeTorino' = calibrated per-qubit "
    "noise with routing on the heavy-hex coupling map. 'exact' rows = state-vector VQPM with the same "
    "pdiff/precision/budget (stop_rule 'none'). Representative trajectory plot = median-error run, "
    "not hand-picked."
)


# ------------------------------------------------------------------ helpers
def rel(p):
    return os.path.relpath(p, ROOT)


def absp(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def jsonl_info(path):
    metas, n, keys = [], 0, set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "meta" in r and "key" not in r:
                metas.append(r["meta"])
            else:
                n += 1
    return metas, n


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def fmt(x, d=2):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "-" if x in (None, "") else str(x)
    if v != v:
        return "-"
    return f"{v:.{d}f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def ci(r, key, d=2, rate=False):
    if rate:
        m, lo, hi = r.get("found_optimum_rate"), r.get("found_optimum_lo"), r.get("found_optimum_hi")
    else:
        m, lo, hi = r.get(f"mean_{key}"), r.get(f"mean_{key}_lo"), r.get(f"mean_{key}_hi")
    if m in (None, ""):
        return "-"
    return f"{fmt(m, d)} [{fmt(lo, d)}, {fmt(hi, d)}]"


def machine():
    try:
        cpu = [ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")][0]
    except Exception:
        cpu = platform.processor()
    try:
        mem = int([ln.split()[1] for ln in open("/proc/meminfo") if ln.startswith("MemTotal")][0]) // 2**20
    except Exception:
        mem = "?"
    return f"{cpu}, {os.cpu_count()} logical CPUs, {mem} GiB RAM, {platform.system()} {platform.release()}"


def versions():
    pins = {}
    p = os.path.join(ROOT, "requirements.txt")
    if os.path.exists(p):
        for ln in open(p):
            if "==" in ln:
                k, v = ln.strip().split("==")
                pins[k] = v
    pins["python"] = platform.python_version()
    return pins


def running():
    try:
        ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    except Exception:
        return ""
    return ps


# ------------------------------------------------------------------ results tables
def tables_grid(e):
    out_dir = absp(e["out_dir"])
    summ = read_csv(os.path.join(out_dir, "vqpm_grid_summary.csv"))
    if not summ:
        return "No summary yet."
    budgets = ["30", "100", "300", "10n"]
    precs = ["3", "6", "exact"]
    parts = []
    for rule in ("oracle", "max_prob"):
        ns = sorted({int(r["n"]) for r in summ})
        hdr = ["n"] + [f"p{p} T{b}" for p in precs for b in budgets]
        rows = []
        for n in ns:
            row = [n]
            for p in precs:
                for b in budgets:
                    m = [r for r in summ if r["precision"] == p and r["stop_rule"] == rule and r["budget"] == b
                         and int(r["n"]) == n]
                    row.append(fmt(m[0]["found_optimum_rate"]) if m else "-")
            rows.append(row)
        parts.append(f"**Found-optimum rate, stop rule `{rule}`** (columns: precision pP, budget T; "
                     f"100 instances per n)\n\n" + md_table(hdr, rows))
    for rule in ("oracle",):
        ns = sorted({int(r["n"]) for r in summ})
        hdr = ["n", "prec", "found-opt [95% CI]", "bit diff [CI]", "norm. energy gap [CI]",
               "norm. expected-E gap [CI]", "iters used", "hit budget", "stalled", "runs w/ wrong lock"]
        rows = []
        for n in ns:
            for p in precs:
                m = [r for r in summ if r["precision"] == p and r["stop_rule"] == rule and r["budget"] == "300"
                     and int(r["n"]) == n]
                if m:
                    r = m[0]
                    rows.append([n, p, ci(r, "", rate=True), ci(r, "bit_diff"), ci(r, "norm_energy_gap", 3),
                                 ci(r, "norm_expected_energy_gap", 3), fmt(r["mean_iterations"], 1),
                                 fmt(r["hit_budget_rate"]), fmt(r["stalled_rate"]), fmt(r["runs_with_wrong_lock"])])
        parts.append(f"**Details at T = 300, stop rule `{rule}`**\n\n" + md_table(hdr, rows))
    return "\n\n".join(parts)


def tables_wrong_locks(e):
    rows = read_csv(os.path.join(absp(e["out_dir"]), "wrong_locks_summary.csv"))
    if not rows:
        return "No summary yet."
    hdr = ["prec", "n", "locks", "wrong/locks", "runs w/ wrong lock", "median 1st wrong-lock iter",
           "median margin correct", "median margin wrong"]
    body = [[r["precision"], r["n"], r["locks"], fmt(r["wrong_lock_rate_per_lock"], 3),
             fmt(r["runs_with_wrong_lock"]), fmt(r["median_first_wrong_lock_iter"], 1),
             fmt(r["median_margin_correct"], 3), fmt(r["median_margin_wrong"], 3)] for r in rows]
    return md_table(hdr, body)


def tables_phase_map(e):
    rows = read_csv(os.path.join(absp(e["out_dir"]), "phase_map_summary.csv"))
    if not rows:
        return "No summary yet."
    hdr = ["map", "rule", "T", "n", "found-opt [CI]", "bit diff", "norm. energy gap [CI]", "iters"]
    body = [[r["phase_map"], r["stop_rule"], r["budget"], r["n"], ci(r, "", rate=True), fmt(r["mean_bit_diff"]),
             ci(r, "norm_energy_gap", 3), fmt(r["mean_iterations"], 1)] for r in rows if r["budget"] == "100"]
    return md_table(hdr, body)


def tables_finite_shots(e):
    rows = read_csv(os.path.join(absp(e["out_dir"]), "finite_shots_summary.csv"))
    if not rows:
        return "No summary yet."
    hdr = ["policy", "rule", "shots/iter", "n", "found-opt [CI]", "bit diff", "norm. energy gap [CI]",
           "runs w/ wrong lock", "iters", "total shots"]
    body = [[r["policy"], r["stop_rule"], r["shots"], r["n"], ci(r, "", rate=True), fmt(r["mean_bit_diff"]),
             ci(r, "norm_energy_gap", 3), fmt(r["runs_with_wrong_lock"]), fmt(r["mean_iterations"], 1),
             fmt(r["mean_shots_used"], 0)] for r in rows]
    return md_table(hdr, body)


def tables_qaoa(e):
    out_dir = absp(e["out_dir"])
    files = sorted(glob.glob(os.path.join(out_dir, "*_summary.csv")))
    rows = read_csv(files[0]) if files else []
    if not rows:
        return "No summary yet."
    hdr = ["method", "n", "runs", "found-opt [CI]", "P(opt)", "bit diff", "norm. energy gap [CI]",
           "norm. expected-E gap", "evals"]
    body = [[r["method"], r["n"], r["runs"], ci(r, "", rate=True), fmt(r["mean_p_optimum"]), fmt(r["mean_bit_diff"]),
             ci(r, "norm_energy_gap", 3), fmt(r.get("mean_norm_expected_energy_gap"), 3), fmt(r["mean_evals"], 0)]
            for r in rows]
    return md_table(hdr, body)


def tables_noisy(e, name):
    rows = read_csv(os.path.join(absp(e["out_dir"]), f"{name}_summary.csv"))
    if not rows:
        return "No summary yet."
    hdr = ["noise", "n", "runs", "found-opt [CI]", "bit diff [CI]", "norm. energy gap [CI]", "iters",
           "wrong locks/run", "shots used"]
    body = [[r["noise"], r["n"], r["runs"], ci(r, "", rate=True), ci(r, "bit_diff"), ci(r, "norm_energy_gap", 3),
             fmt(r["mean_iterations"], 1), fmt(r["mean_wrong_locks"]), fmt(r["mean_shots_used"], 0)] for r in rows]
    return md_table(hdr, body)


def tables_reproduction(e):
    out = []
    for p in e["raw"]:
        ap = absp(p)
        if not os.path.exists(ap):
            continue
        d = json.load(open(ap))
        if "fig6a" in d:
            out.append(f"- `{os.path.basename(p)}` ({d['impl']}): Fig. 6a mean bit difference "
                       f"{d['fig6a']['mean_bit_diff']:.4f} (published 1.16)")
        if "scaling" in d:
            s = d["scaling"]
            out.append("  scaling-figure VQPM mean final target probability: "
                       + ", ".join(f"n={n}: {v['mean_target_prob']:.4f}" for n, v in s.items())
                       + " (published 0.46 / 0.40 / 0.18 at n = 12 / 14 / 16)")
        if "fig9" in d:
            f9 = d["fig9"]
            out.append(f"- `{os.path.basename(p)}`: Fig. 9 VQPM prob {f9['vqpm_prob']:.4f} (0.43), QAOA prob "
                       f"{f9['qaoa_prob']:.4f} (0.55), VQPM bits {f9['vqpm_bits']:.2f} (0.43), QAOA bits "
                       f"{f9['qaoa_bits']:.2f} (0.33)")
    return "\n".join(out) or "No results yet."


def tables_gate_counts(e):
    p = absp(e["raw"][0])
    if not os.path.exists(p):
        return "No results yet."
    rows = json.load(open(p))
    hdr = ["n", "target", "2q gates", "depth", "2q depth", "pairs"]
    return md_table(hdr, [[r["n"], r["backend"], f"{r['two_qubit_gates']} {'/'.join(r['two_qubit_gate_names'])}",
                           r["depth"], r["two_qubit_depth"], r["pairs"]] for r in rows])


# ------------------------------------------------------------------ report
def write_report(name):
    e = EXPERIMENTS[name]
    raws = [absp(p) for p in e["raw"]]
    existing = [p for p in raws if os.path.exists(p)]
    if not existing:
        return None
    ps = running()
    script = e["script"]
    src = "experiments/vqpm_grid.py" if name == "wrong_locks" else script
    is_running = any(src in ln and ".venv" in ln and "--plot-only" not in ln
                     and (os.path.basename(e["raw"][0]) in ln or name in ("vqpm_grid", "wrong_locks"))
                     for ln in ps.splitlines())
    lines = [f"# {e['title']}", "", f"_Report generated {datetime.datetime.now().isoformat(timespec='seconds')}"
             f" by `experiments/reports.py`; status: {'RUNNING (partial data)' if is_running else 'finished'}._",
             "", "## Purpose", "", e["purpose"], "", "## Method", "",
             NOISY_METHOD if e["method"] == "NOISY" else e["method"], "", METRICS_TEXT, ""]

    # provenance
    lines += ["## Provenance", ""]
    for p in existing:
        if p.endswith(".jsonl"):
            metas, n = jsonl_info(p)
            for k, m in enumerate(metas):
                lines.append(f"- Invocation {k + 1} of `{rel(p)}`: started {m.get('time')}, git commit "
                             f"`{m.get('git_commit')}`{' (dirty tree)' if m.get('git_dirty') else ''}, "
                             f"python {m.get('python')}, numpy {m.get('numpy')}, scipy {m.get('scipy')}")
                if m.get("argv"):
                    lines.append(f"  - command: `{PY} {' '.join(m['argv'])}`")
                if m.get("args"):
                    lines.append(f"  - parameters: `{json.dumps(m['args'], sort_keys=True).replace(ROOT + os.sep, '')}`")
    if name in ("vqpm_grid", "wrong_locks"):
        lines.append("- The n = 21, 22 records were added by a second invocation of the same command with "
                     "`--n 21 22` (it resumes the same file); expected energies of records made before commit "
                     "ed0b4c4 come from `--backfill-expected-energy` (sidecar file).")
    lines += [f"- Library versions (pinned in requirements.txt): "
              + ", ".join(f"{k} {v}" for k, v in versions().items()),
              f"- Machine: {machine()}",
              f"- End time (last write of the raw data): "
              + ", ".join(f"`{os.path.basename(p)}` {datetime.datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec='seconds')}"
                          for p in existing)]
    for p in existing:
        if p.endswith(".jsonl"):
            metas, _ = jsonl_info(p)
            if metas and metas[0].get("time"):
                t0 = datetime.datetime.fromisoformat(metas[0]["time"])
                t1 = datetime.datetime.fromtimestamp(os.path.getmtime(p))
                lines.append(f"- Wall time `{os.path.basename(p)}` (first start to last write, includes waits "
                             f"between invocations): {(t1 - t0).total_seconds() / 3600:.2f} h")
    seed = e.get("seed", 2026)
    inst = sorted(glob.glob(os.path.join(INST, f"qubo_n*_seed{seed}.npz")))
    if name not in ("reproduction", "gate_counts"):
        lines.append(f"- Instances: seed {seed}, files `{rel(INST)}/qubo_n<n>_seed{seed}.npz` "
                     f"(instance i is `default_rng([{seed}, n, i])`, Q = A + A^T, A_ij ~ N(0, 1); each file "
                     f"stores Q, optimum, E_min, E_max, gap); sha256 listed below")
    lines.append("")

    # data
    lines += ["## Where the data is", "", "Paths are relative to the repository root.", ""]
    hdr = ["file (repo-relative)", "records / size", "sha256"]
    rows = []
    for p in existing:
        if p.endswith(".jsonl"):
            _, n = jsonl_info(p)
            size = f"{n} records, {os.path.getsize(p) / 2**20:.1f} MiB"
        else:
            size = f"{os.path.getsize(p) / 2**10:.1f} KiB"
        rows.append([f"`{rel(p)}`", size, f"`{sha256(p)}`"])
    if name not in ("reproduction", "gate_counts"):
        for p in inst:
            rows.append([f"`{rel(p)}`", f"{os.path.getsize(p) / 2**10:.0f} KiB", f"`{sha256(p)}`"])
    lines += [md_table(hdr, rows), ""]
    out_dir = absp(e["out_dir"])
    pat = e.get("files_glob")
    files = sorted(glob.glob(os.path.join(out_dir, pat))) if pat else sorted(
        f for f in glob.glob(os.path.join(out_dir, "*")) if os.path.isfile(f)
        and not os.path.basename(f).startswith("wrong_locks"))
    if files:
        lines += ["Summary tables and plots (committed to git unless noted):", ""]
        for f in files:
            note = " (not committed: large; regenerate)" if os.path.getsize(f) > 2**21 else ""
            lines.append(f"- `{rel(f)}`{note}")
        lines.append("")
    lines += ["Raw JSONL files are not in the repository because of their size. The scripts write them to "
              "`experiments/results/<experiment>/raw/`, and the summary tables above are computed from them.", ""]

    # results
    lines += ["## Results", ""]
    t = e["tables"]
    if t == "grid":
        lines.append(tables_grid(e))
    elif t == "wrong_locks":
        lines.append(tables_wrong_locks(e))
    elif t == "phase_map":
        lines.append(tables_phase_map(e))
    elif t == "finite_shots":
        lines.append(tables_finite_shots(e))
    elif t == "qaoa":
        lines.append(tables_qaoa(e))
    elif t == "noisy":
        lines.append(tables_noisy(e, name))
    elif t == "reproduction":
        lines.append(tables_reproduction(e))
    elif t == "gate_counts":
        lines.append(tables_gate_counts(e))
    lines.append("")
    fnd = os.path.join(REPORTS, "findings", f"{name}.md")
    lines += ["## Findings and caveats", ""]
    if os.path.exists(fnd):
        lines.append(open(fnd).read().strip())
    else:
        lines.append("_Not yet written: the tables above are generated automatically; interpretation is "
                     f"added in `experiments/reports/findings/{name}.md`._")
    lines += ["", "## Regenerate tables and plots from the saved data (no re-run)", "", "```bash",
              *e["regen"], f"{PY} experiments/reports.py", "```", ""]
    os.makedirs(REPORTS, exist_ok=True)
    path = os.path.join(REPORTS, f"{name}.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path, ("running" if is_running else "finished")


def write_index(status):
    lines = ["# Experiments: index", "",
             f"_Updated {datetime.datetime.now().isoformat(timespec='seconds')}. One-call progress check: "
             f"`{PY} experiments/status.py`._", "",
             md_table(["experiment", "purpose", "status", "report"],
                      [[EXPERIMENTS[k]["title"], SHORT.get(k, ""), status.get(k, "not started"),
                        f"[{k}.md]({k}.md)" if k in status else "-"] for k in EXPERIMENTS]), ""]
    with open(os.path.join(REPORTS, "INDEX.md"), "w") as f:
        f.write("\n".join(lines))


def write_all():
    status = {}
    for name in EXPERIMENTS:
        try:
            r = write_report(name)
        except Exception as ex:  # never break an experiment because of a report
            print(f"report {name} failed: {ex!r}")
            continue
        if r:
            status[name] = r[1]
    write_index(status)
    return status


def archive():
    """Copy every raw file (and the instance files) whose content changed since
    the last archive into BACKUP/<date>/<experiment>_raw_<time>.tar.gz; the
    archives' checksums go to BACKUP/<date>/SHA256SUMS and the per-file
    checksums of what was archived to BACKUP/archived.json."""
    os.makedirs(BACKUP, exist_ok=True)
    man_path = os.path.join(BACKUP, "archived.json")
    manifest = json.load(open(man_path)) if os.path.exists(man_path) else {}
    day = datetime.date.today().isoformat()
    d = os.path.join(BACKUP, day)
    os.makedirs(d, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%H%M%S")
    groups = {name: [absp(p) for p in e["raw"]] for name, e in EXPERIMENTS.items()}
    groups["instances"] = sorted(glob.glob(os.path.join(INST, "*.npz")))
    made, seen = [], set()
    for name, files in groups.items():
        changed = []
        for p in files:
            if p in seen or not os.path.exists(p):
                continue
            seen.add(p)
            h = sha256(p)
            if manifest.get(rel(p)) != h:
                changed.append((p, h))
        if not changed:
            continue
        tgt = os.path.join(d, f"{name}_raw_{stamp}.tar.gz")
        with tarfile.open(tgt, "w:gz") as tf:
            for p, _ in changed:
                tf.add(p, arcname=rel(p))
        with open(os.path.join(d, f"{name}_raw_{stamp}.files.sha256"), "w") as f:
            for p, h in changed:
                f.write(f"{h}  {rel(p)}\n")
        for p, h in changed:
            manifest[rel(p)] = h
        made.append(tgt)
    with open(os.path.join(d, "SHA256SUMS"), "a") as f:
        for p in made:
            f.write(f"{sha256(p)}  {os.path.basename(p)}\n")
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    return made


def on_finish(name=None):
    """Called at the end of an experiment script: archive new or changed raw
    data and rewrite all reports. Never raises."""
    try:
        made = archive()
        status = write_all()
        print(f"reports updated ({len(status)} experiments); archived: {', '.join(map(os.path.basename, made))}",
              flush=True)
    except Exception as ex:
        print(f"on_finish({name}) failed: {ex!r}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", action="store_true")
    args = ap.parse_args()
    if args.archive:
        for p in archive():
            print("archived", p)
    st = write_all()
    for k, v in st.items():
        print(f"{k}: {v} -> experiments/reports/{k}.md")


if __name__ == "__main__":
    main()
