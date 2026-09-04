"""保存任务配置信息到 JSON，供 phyvla-p2 环境使用。

因为 phyvla-p2 没有安装 LIBERO，所以需要先在 libero_replay 环境中
解析任务信息（target/goal body 名称等）并保存，然后 phyvla-p2 读取。

运行（libero_replay）：
    python scripts/save_task_info.py --suite libero_spatial --task-ids 1 2 3 4 5 6 7 8 9 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def save_task_info(suite: str, task_id: int, demo_index: int) -> Path:
    from utils.task_config import TaskConfig
    from libero.libero import get_libero_path

    cfg = TaskConfig.from_task_id(suite=suite, task_id=task_id, demo_index=demo_index)

    info = {
        "suite":       cfg.suite,
        "task_id":     cfg.task_id,
        "language":    cfg.language,
        "target_body": cfg.target_body,
        "goal_body":   cfg.goal_body,
        "bddl_path":   str(cfg.bddl_path),
        "demo_path":   str(cfg.demo_path),
    }

    out_path = cfg.output_dir / "task_info.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(f"  ✅ task{task_id:02d}: {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--suite",     default="libero_spatial")
    p.add_argument("--task-ids",  type=int, nargs="+", default=list(range(1, 11)))
    p.add_argument("--demo-index", type=int, default=0)
    args = p.parse_args()

    print(f"保存任务配置信息（suite={args.suite}，共 {len(args.task_ids)} 个任务）")
    for tid in args.task_ids:
        try:
            save_task_info(args.suite, tid, args.demo_index)
        except FileNotFoundError as e:
            print(f"  ⚠️  task{tid:02d}: model_raw.xml 不存在，跳过（{e}）")
        except Exception as e:
            print(f"  ❌  task{tid:02d}: {e}")

    print("\n完成！phyvla-p2 环境可通过 TaskConfig.from_task_id_phyvla_p2() 读取。")


if __name__ == "__main__":
    main()
