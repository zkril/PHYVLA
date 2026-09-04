"""Run P1 pipeline for multiple tasks and print a comparison table.

Usage:
    python scripts/run_multi_task_pipeline.py --task-ids 1 2 3 4 5 6 7
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS      = PROJECT_ROOT / "scripts"
SRC          = PROJECT_ROOT / "src"
for _p in (str(SCRIPTS), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def run(cmd: list) -> None:
    print("$", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def task_outdir(tid: int) -> Path:
    # task02 uses the original outputs/physics/ layout
    if tid == 2:
        return PROJECT_ROOT / "outputs"
    return PROJECT_ROOT / f"outputs/task{tid:02d}"


def physics_npz(tid: int) -> Path:
    if tid == 2:
        return PROJECT_ROOT / "outputs/physics/task02_demo00_contact_force_data.npz"
    d = task_outdir(tid)
    return d / f"task{tid:02d}_demo00_physics.npz"


def events_npz(tid: int) -> Path:
    return task_outdir(tid) / "contact_events_v1.npz"


def stats_json(tid: int) -> Path:
    return task_outdir(tid) / "normalization_v1.json"


def tokens_npz(tid: int) -> Path:
    return task_outdir(tid) / "physics_tokens_demo0.npz"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--suite",      default="libero_spatial")
    p.add_argument("--task-ids",   type=int, nargs="+", default=list(range(1, 8)))
    p.add_argument("--demo-index", type=int, default=0)
    p.add_argument("--epochs",     type=int, default=200)
    p.add_argument("--device",     default="cuda")
    p.add_argument("--skip-extract", action="store_true",
                   help="Skip extraction if npz already exists")
    return p.parse_args()


def extract_task(args, tid: int, lang: str) -> None:
    out = task_outdir(tid)
    out.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    npz = physics_npz(tid)
    if args.skip_extract and npz.exists():
        print(f"[task {tid}] physics npz exists, skipping extraction")
    else:
        run([py, str(SCRIPTS / "visualize_contact_forces.py"),
             "--suite", args.suite, "--task-id", str(tid),
             "--demo-index", str(args.demo_index),
             "--max-contacts", "128",
             "--output-npz",   str(npz),
             "--output-video", str(out / f"task{tid:02d}_demo00.mp4"),
             "--output-json",  str(out / f"task{tid:02d}_demo00_physics.json"),
        ])

    sj = stats_json(tid)
    if args.skip_extract and sj.exists():
        print(f"[task {tid}] stats exists, skipping normalization")
    else:
        run([py, str(SCRIPTS / "compute_normalization.py"),
             "--input", str(npz), "--output", str(sj)])

    ev = events_npz(tid)
    if args.skip_extract and ev.exists():
        print(f"[task {tid}] events exists, skipping")
    else:
        run([py, str(SCRIPTS / "build_contact_events.py"),
             "--input", str(npz), "--output", str(ev)])

    tk = tokens_npz(tid)
    if args.skip_extract and tk.exists():
        print(f"[task {tid}] tokens exists, skipping build")
    else:
        run([py, str(SCRIPTS / "build_offline_tokens.py"),
             "--physics", str(npz), "--events", str(ev),
             "--stats", str(sj), "--output", str(tk),
             "--language", lang])


def train_task(args, tid: int) -> dict:
    """Train and return final metrics dict."""
    py = sys.executable
    result = subprocess.run(
        [py, str(SCRIPTS / "train_tokenizer.py"),
         "--physics", str(physics_npz(tid)),
         "--events",  str(events_npz(tid)),
         "--stats",   str(stats_json(tid)),
         "--epochs",  str(args.epochs),
         "--log-every", str(args.epochs),   # only print final
         "--device",  args.device],
        capture_output=True, text=True,
    )
    stdout = result.stdout + result.stderr
    # Parse final metrics from stdout
    metrics = {"task_id": tid, "pass": False}
    for line in stdout.splitlines():
        for key, short in [
            ("z_robot   reconstruction RMSE", "robot_rmse"),
            ("z_afford  phase accuracy",       "phase_acc"),
            ("z_contact relation F1",          "contact_f1"),
            ("z_obj[0]  role accuracy",        "role_acc"),
            ("z_obj[0]  param RMSE",           "param_rmse"),
            ("z_obj[0]  pose RMSE",            "pose_rmse"),
        ]:
            if key in line:
                try:
                    val = float(line.split("=")[1].split()[0])
                    metrics[short] = val
                except Exception:
                    pass
        if "ALL PASS" in line:
            metrics["pass"] = True
    return metrics


def print_table(rows: list, langs: dict) -> None:
    print("\n" + "=" * 100)
    print("MULTI-TASK P1 ACCEPTANCE SUMMARY")
    print("=" * 100)
    hdr = f"{'Task':>5}  {'robot_rmse':>10}  {'phase_acc':>9}  {'contact_f1':>10}  {'role_acc':>8}  {'param_rmse':>10}  {'pose_rmse':>9}  {'PASS':>6}"
    print(hdr)
    print("-" * 100)
    for m in rows:
        tid = m["task_id"]
        status = "✅" if m.get("pass") else "❌"
        print(
            f"  {tid:>3}  "
            f"{m.get('robot_rmse', 0):.4f}      "
            f"{m.get('phase_acc',  0):.4f}     "
            f"{m.get('contact_f1', 0):.4f}       "
            f"{m.get('role_acc',   0):.4f}    "
            f"{m.get('param_rmse', 0):.4f}       "
            f"{m.get('pose_rmse',  0):.4f}    "
            f"  {status}"
        )
    n_pass = sum(1 for m in rows if m.get("pass"))
    print(f"\n  {n_pass}/{len(rows)} tasks passed all criteria")
    print("=" * 100)


def main() -> None:
    args  = parse_args()
    tids  = args.task_ids

    from libero.libero import benchmark
    bd    = benchmark.get_benchmark_dict()
    suite = bd[args.suite]()
    langs = {i: suite.get_task(i).language for i in tids}

    print(f"\nTasks: {tids}")
    for tid in tids:
        print(f"  {tid}: {langs[tid]}")

    # ── Step 1: Extract all tasks ───────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1: Physics extraction")
    print("=" * 60)
    for tid in tids:
        print(f"\n── Task {tid}: {langs[tid]} ──")
        extract_task(args, tid, langs[tid])

    # ── Step 2: Train all tasks ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Training & acceptance check")
    print("=" * 60)
    all_metrics = []
    for tid in tids:
        print(f"\n── Task {tid} ({langs[tid][:50]}...) ──")
        m = train_task(args, tid)
        all_metrics.append(m)
        status = "✅ PASS" if m.get("pass") else "❌ FAIL"
        print(f"   {status}  phase={m.get('phase_acc',0):.3f}  "
              f"role={m.get('role_acc',0):.3f}  "
              f"param={m.get('param_rmse',0):.4f}  "
              f"contact_f1={m.get('contact_f1',0):.3f}")

    # ── Summary ─────────────────────────────────────────────────────
    print_table(all_metrics, langs)

    out_path = PROJECT_ROOT / "outputs/multi_task_results.json"
    with open(out_path, "w") as f:
        json.dump({"tasks": all_metrics, "languages": langs}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
