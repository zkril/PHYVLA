"""General P1+P2+P3+P4 pipeline runner for any libero_spatial task.

Runs（libero_replay 环境）：
  P2: XML 提取 → raw_ctrl 导出
  P1: 物理提取 → 归一化 → 事件 → token → 训练

P3/P4（mujoco_warp，phyvla-p2 环境）由 run_multitask_p3p4.py 处理。
已移除 MJX 步骤（MJCF 裁减 / parity 验证），mujoco_warp 使用 model_raw.xml。

Usage:
    # 仅 libero_replay 环境（P1+P2 XML 提取）：
    python scripts/run_task_pipeline.py --task-id 3 \
        --py-libero /path/to/libero_replay/bin/python

    # 同时提供 phyvla-p2 路径（已弃用，P3/P4 改由 run_multitask_p3p4.py 处理）：
    python scripts/run_task_pipeline.py --task-id 3 \
        --py-libero  /path/to/libero_replay/bin/python \
        --py-p2      /path/to/phyvla-p2/bin/python
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS      = PROJECT_ROOT / "scripts"
SRC          = PROJECT_ROOT / "src"

for _p in (str(SCRIPTS), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def run(cmd: list, **kwargs) -> None:
    print("\n$", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kwargs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--suite",        default="libero_spatial")
    p.add_argument("--task-id",      type=int, required=True)
    p.add_argument("--demo-index",   type=int, default=0)
    p.add_argument("--max-contacts", type=int, default=128)
    p.add_argument("--epochs",       type=int, default=200)
    p.add_argument("--device",       default="cuda")
    # P2 环境路径（可选）
    p.add_argument("--py-libero",    type=str, default=None,
                   help="libero_replay env python 路径（用于 P2 XML 提取）")
    p.add_argument("--py-p2",        type=str, default=None,
                   help="phyvla-p2 env python 路径（用于 P2 MJCF 裁减）")
    p.add_argument("--py-starvla",   type=str, default=None,
                   help="starVLA env python 路径（用于 train_tokenizer，支持 H100）")
    p.add_argument("--output-dir",   type=str, default=None,
                   help="自定义输出目录（默认: outputs/task{id}）")
    p.add_argument("--skip-p2",      action="store_true",
                   help="跳过 P2 步骤（已有 reduced MJCF 时使用）")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    tid    = args.task_id

    # 支持自定义输出目录
    if args.output_dir:
        outdir = Path(args.output_dir)
    else:
        outdir = PROJECT_ROOT / f"outputs/task{tid:02d}"
    outdir.mkdir(parents=True, exist_ok=True)

    # 解析任务语言
    from libero.libero import benchmark
    bd    = benchmark.get_benchmark_dict()
    suite = bd[args.suite]()
    task  = suite.get_task(tid)
    lang  = task.language
    print(f"\n{'='*60}")
    print(f"Task {tid}: {lang}")
    print(f"Output dir: {outdir}")
    print(f"{'='*60}")

    physics_npz = outdir / f"task{tid:02d}_demo{args.demo_index:02d}_physics.npz"
    events_npz  = outdir / "contact_events_v1.npz"
    tokens_npz  = outdir / f"physics_tokens_demo{args.demo_index}.npz"
    stats_json  = outdir / "normalization_v1.json"
    raw_xml     = outdir / "model_raw.xml"
    reduced_xml = outdir / "model_mjx_reduced_v1.xml"
    ctrl_trace  = outdir / "controller_trace_v1.npz"

    py = sys.executable

    # ── P2: XML 提取 + raw_ctrl 导出 ────────────────────────────
    # 注：mujoco_warp 使用 model_raw.xml（完整模型），不再需要 reduced MJCF
    if not args.skip_p2:
        if args.py_libero:
            py_lr = args.py_libero
            print("\n[P2] XML 提取（libero_replay 环境）")
            run([py_lr, str(SCRIPTS / "extract_task_xml.py"),
                 "--suite", args.suite, "--task-id", str(tid),
                 "--demo-index", str(args.demo_index),
                 "--output", str(raw_xml),
            ])
            print("\n[P2] raw_ctrl 导出（libero_replay 环境）")
            run([py_lr, str(SCRIPTS / "export_controller_trace.py"),
                 "--suite", args.suite, "--task-id", str(tid),
                 "--demo-index", str(args.demo_index),
                 "--output", str(ctrl_trace),
            ])
        else:
            print("[P2] 未提供 --py-libero，跳过 XML 提取和 raw_ctrl 导出")
    else:
        print("[P2] --skip-p2，跳过所有 P2 步骤")

    # ── P1 step 1: 物理提取 ──────────────────────────────────
    run([py, str(SCRIPTS / "visualize_contact_forces.py"),
         "--suite", args.suite, "--task-id", str(tid),
         "--demo-index", str(args.demo_index),
         "--max-contacts", str(args.max_contacts),
         "--output-npz",   str(physics_npz),
         "--output-video", str(outdir / f"task{tid:02d}_demo{args.demo_index:02d}.mp4"),
         "--output-json",  str(outdir / f"task{tid:02d}_demo{args.demo_index:02d}_physics.json"),
    ])

    # ── P1 step 2: 归一化 ────────────────────────────────────
    run([py, str(SCRIPTS / "compute_normalization.py"),
         "--input", str(physics_npz), "--output", str(stats_json),
    ])

    # ── P1 step 3: 接触事件 ──────────────────────────────────
    run([py, str(SCRIPTS / "build_contact_events.py"),
         "--input", str(physics_npz), "--output", str(events_npz),
    ])

    # Train and persist the encoder before exporting tokens from those weights.
    py_train = args.py_starvla if args.py_starvla else py
    tokenizer_checkpoint = tokens_npz.with_name("tokenizer_last.pt")
    run([py_train, str(SCRIPTS / "train_tokenizer.py"),
         "--physics", str(physics_npz), "--events", str(events_npz),
         "--stats", str(stats_json), "--language", lang,
         "--epochs", str(args.epochs), "--device", args.device,
         "--output", str(tokenizer_checkpoint),
    ])
    run([py_train, str(SCRIPTS / "build_offline_tokens.py"),
         "--physics", str(physics_npz), "--events", str(events_npz),
         "--stats", str(stats_json), "--output", str(tokens_npz),
         "--language", lang, "--tokenizer-checkpoint", str(tokenizer_checkpoint),
    ])



if __name__ == "__main__":
    main()

