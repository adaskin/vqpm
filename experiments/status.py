"""One-call status of the experiments.

    .venv/bin/python experiments/status.py

Shows which experiment processes are running, the last progress line of
each lane log, how many raw records each experiment has, and headline
numbers from each summary CSV (found-optimum rate, mean bit difference and
mean normalised energy gap; CSVs are rewritten when an experiment finishes).
"""

import csv
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "experiments", "results")
LOGS = os.path.join(ROOT, "experiments", "logs")

EXPERIMENTS = [
    ("1 VQPM grid", "vqpm_grid/raw/grid.jsonl", "vqpm_grid/vqpm_grid_summary.csv"),
    ("2 wrong locks", None, "vqpm_grid/wrong_locks_summary.csv"),
    ("3 phase maps", "phase_map/raw/phase_map.jsonl", "phase_map/phase_map_summary.csv"),
    ("4 finite shots", "finite_shots/raw/finite_shots.jsonl", "finite_shots/finite_shots_summary.csv"),
    ("5 QAOA vs VQPM", "qaoa_vs_vqpm/raw/qaoa_vs_vqpm.jsonl", "qaoa_vs_vqpm/qaoa_vs_vqpm_summary.csv"),
    ("6a noise sweep n=8", "noisy/raw/noise_sweep_n8.jsonl", "noisy/noise_sweep_n8_summary.csv"),
    ("6b FakeTorino n=8", "noisy/raw/faketorino_n8.jsonl", "noisy/faketorino_n8_summary.csv"),
    ("6c depol (Torino) n=8,10", "noisy/raw/depol_torino.jsonl", "noisy/depol_torino_summary.csv"),
]


def lines(path):
    if not path or not os.path.exists(path):
        return 0
    with open(path, "rb") as f:
        return max(0, sum(1 for _ in f) - 1)  # minus the metadata line


def read_csv(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def f(x, d=2):
    try:
        return f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return "-"


def headline(name, rows):
    out = []
    if name.startswith("1"):
        for prec in ("3", "6", "exact"):
            for rule in ("oracle", "max_prob"):
                for b in ("30", "300"):
                    rs = sorted([r for r in rows if r["precision"] == prec and r["stop_rule"] == rule
                                 and r["budget"] == b], key=lambda r: int(r["n"]))
                    if rs:
                        out.append(f"  prec={prec:5s} {rule:8s} T={b:3s} found-opt by n: "
                                   + " ".join(f"{r['n']}:{f(r['found_optimum_rate'])}" for r in rs))
    elif name.startswith("2"):
        for r in rows:
            out.append(f"  prec={r['precision']:5s} n={r['n']:>2s} wrong/locks={f(r['wrong_lock_rate_per_lock'], 3)}"
                       f" runs w/ wrong lock={f(r['runs_with_wrong_lock'])}")
    elif name.startswith("3"):
        for r in rows:
            if r["budget"] == "100" and r["stop_rule"] == "oracle":
                out.append(f"  {r['phase_map']:10s} n={r['n']:>2s} found-opt={f(r['found_optimum_rate'])}"
                           f" bits={f(r['mean_bit_diff'])} normE={f(r.get('mean_norm_energy_gap'), 3)}")
    elif name.startswith("4"):
        for r in rows:
            if r["stop_rule"] == "oracle":
                out.append(f"  {r['policy']:15s} shots={r['shots']:>7s} n={r['n']:>2s} found-opt="
                           f"{f(r['found_optimum_rate'])} normE={f(r.get('mean_norm_energy_gap'), 3)}"
                           f" shots used={f(r['mean_shots_used'], 0)}")
    elif name.startswith("5"):
        for r in rows:
            if r["kind"] != "qaoa" or r["p"] in ("8",):
                out.append(f"  {r['method'][:34]:34s} n={r['n']:>2s} found-opt={f(r['found_optimum_rate'])}"
                           f" P(opt)={f(r['mean_p_optimum'])} normE={f(r.get('mean_norm_energy_gap'), 3)}"
                           f" evals={f(r['mean_evals'], 0)}")
    else:
        for r in rows:
            out.append(f"  {r['noise']:26s} n={r['n']:>2s} runs={r['runs']:>3s} found-opt="
                       f"{f(r['found_optimum_rate'])} bits={f(r['mean_bit_diff'])}"
                       f" normE={f(r.get('mean_norm_energy_gap'), 3)}")
    return out


def main():
    ps = subprocess.run(["ps", "-eo", "etime,args"], capture_output=True, text=True).stdout
    running = sorted({ln.split("experiments/")[1].split()[0]
                      for ln in ps.splitlines() if "experiments/" in ln and ".venv" in ln
                      and "status.py" not in ln})
    print("running:", ", ".join(running) if running else "nothing")
    for lane in ("lane1", "lane2"):
        p = os.path.join(LOGS, f"{lane}.log")
        if os.path.exists(p):
            with open(p) as fh:
                tail = [ln.rstrip() for ln in fh if ln.strip()][-1:]
            print(f"{lane}: {tail[0][:150] if tail else ''}")
    for name, raw, summ in EXPERIMENTS:
        rows = read_csv(os.path.join(R, summ))
        n = lines(os.path.join(R, raw)) if raw else None
        state = "summary written" if rows is not None else "no summary yet"
        print(f"\n[{name}] raw records: {n if n is not None else '-'}; {state}")
        if rows:
            for ln in headline(name, rows)[:40]:
                print(ln)


if __name__ == "__main__":
    main()
