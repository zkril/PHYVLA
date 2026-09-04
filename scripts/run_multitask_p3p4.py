"""批量运行 P3+P4 流程，适用于所有任务。

将 P3（rollout features）和 P4（梯度修正）扩展到多个任务，
所有任务特定配置（target/goal body 等）通过 task_info.json 自动解析。

两阶段运行（两个 conda 环境）：

阶段一（libero_replay）：保存任务配置
    conda activate libero_replay
    python scripts/save_task_info.py --task-ids 1 2 3 4 5 6 7 8 9 10

阶段二（phyvla-p2）：批量计算 rollout features + 梯度
    conda activate phyvla-p2
    python scripts/run_multitask_p3p4.py --task-ids 1 2 3 4 5 6 7 8 9 10

阶段三（libero_replay）：批量渲染对比视频
    conda activate libero_replay
    python scripts/run_multitask_p3p4.py --stage render --task-ids 1 2 3 4 5 6 7 8 9 10
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS      = PROJECT_ROOT / "scripts"


def run_cmd(cmd: list[str], label: str) -> tuple[bool, float]:
    """运行命令，返回 (success, elapsed_s)。"""
    t0 = time.time()
    print(f"\n  $ {' '.join(str(c) for c in cmd)}")
    try:
        subprocess.run([str(c) for c in cmd], check=True)
        return True, time.time() - t0
    except subprocess.CalledProcessError as e:
        print(f"  ❌ {label} 失败（返回码 {e.returncode}）")
        return False, time.time() - t0


def check_prerequisites(task_id: int) -> dict[str, bool]:
    """检查任务所需的前置文件是否存在。"""
    out_dir = PROJECT_ROOT / f"outputs/task{task_id:02d}"
    return {
        "model_raw.xml":         (out_dir / "model_raw.xml").exists(),
        "controller_trace.npz":  (out_dir / "controller_trace_v1.npz").exists(),
        "task_info.json":        (out_dir / "task_info.json").exists(),
    }


def run_p3_rollout(task_id: int) -> bool:
    """为单个任务计算 rollout features（phyvla-p2 环境）。"""
    ok, elapsed = run_cmd(
        [sys.executable, SCRIPTS / "build_rollout_features.py",
         "--task-id", str(task_id)],
        f"task{task_id:02d} rollout features",
    )
    if ok:
        print(f"  ✅ task{task_id:02d} rollout features 完成（{elapsed:.1f}s）")
    return ok


def run_p4_grad(task_id: int, start_frame: int = 55, n_iters: int = 5) -> bool:
    """为单个任务运行梯度修正（phyvla-p2 环境）。"""
    ok, elapsed = run_cmd(
        [sys.executable, SCRIPTS / "demo_gradient_refinement.py",
         "--task-id", str(task_id),
         "--start-frame", str(start_frame),
         "--n-iters", str(n_iters),
         "--step-size", "5.0",
         "--eps", "2.0"],
        f"task{task_id:02d} gradient refinement",
    )
    if ok:
        print(f"  ✅ task{task_id:02d} 梯度修正完成（{elapsed:.1f}s）")
    return ok


def run_render(task_id: int, start_frame: int = 55) -> bool:
    """为单个任务渲染对比视频（phyvla-p2 环境，mujoco.Renderer 单实例）。"""
    ok, elapsed = run_cmd(
        [sys.executable, SCRIPTS / "render_warp_comparison.py",
         "--task-id", str(task_id),
         "--start-frame", str(start_frame)],
        f"task{task_id:02d} render",
    )
    if ok:
        print(f"  ✅ task{task_id:02d} 视频渲染完成（{elapsed:.1f}s）")
    return ok


def stage_compute(task_ids: list[int], args) -> dict:
    """阶段二：批量计算 P3 + P4（phyvla-p2 环境）。"""
    results = {}
    print(f"\n{'='*60}")
    print(f"阶段：批量计算 P3 rollout + P4 梯度（{len(task_ids)} 个任务）")
    print(f"{'='*60}")

    for tid in task_ids:
        print(f"\n─── task{tid:02d} ───")
        prereq = check_prerequisites(tid)
        missing = [k for k, v in prereq.items() if not v]
        if missing:
            print(f"  ⚠️  跳过 task{tid:02d}：缺少 {missing}")
            results[tid] = {"skip": True, "reason": f"缺少 {missing}"}
            continue

        res = {"rollout": False, "grad": False}

        # P3：rollout features
        rollout_npz = PROJECT_ROOT / f"outputs/task{tid:02d}/rollout_features.npz"
        if rollout_npz.exists() and not args.force:
            print(f"  ⏭  rollout_features.npz 已存在，跳过（--force 强制重算）")
            res["rollout"] = True
        else:
            res["rollout"] = run_p3_rollout(tid)

        # P4：梯度修正
        if res["rollout"]:
            res["grad"] = run_p4_grad(tid, start_frame=args.start_frame)

        results[tid] = res

    return results


def stage_render(task_ids: list[int], args) -> dict:
    """阶段三：批量渲染（libero_replay 环境）。"""
    results = {}
    print(f"\n{'='*60}")
    print(f"阶段：批量渲染对比视频（{len(task_ids)} 个任务）")
    print(f"{'='*60}")

    for tid in task_ids:
        qpos_file = PROJECT_ROOT / f"outputs/task{tid:02d}/qpos_refined_p4.npy"
        if not qpos_file.exists():
            print(f"  ⚠️  跳过 task{tid:02d}：qpos_refined_p4.npy 不存在")
            results[tid] = {"skip": True}
            continue

        print(f"\n─── task{tid:02d} ───")
        results[tid] = {"render": run_render(tid, start_frame=args.start_frame)}

    return results


def print_summary(task_ids: list[int], results: dict, stage: str) -> None:
    print(f"\n{'='*60}")
    print(f"批量 {stage} 汇总")
    print(f"{'='*60}")

    passed = skipped = failed = 0
    for tid in task_ids:
        res = results.get(tid, {})
        if res.get("skip"):
            status = "⚠️  跳过"
            skipped += 1
        elif all(v for k, v in res.items() if isinstance(v, bool)):
            status = "✅ 通过"
            passed += 1
        else:
            status = "❌ 失败"
            failed += 1
        print(f"  task{tid:02d}: {status}  {res}")

    print(f"\n总计：✅{passed}  ⚠️{skipped}  ❌{failed}  / {len(task_ids)} 任务")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task-ids",    type=int, nargs="+", default=list(range(1, 11)))
    p.add_argument("--suite",       default="libero_spatial")
    p.add_argument("--stage",       choices=["compute", "render", "all"], default="compute",
                   help="compute=P3+P4计算(phyvla-p2)  render=渲染(libero_replay)  all=两阶段")
    p.add_argument("--start-frame", type=int, default=55)
    p.add_argument("--force",       action="store_true",
                   help="强制重算已有的 rollout_features.npz")
    args = p.parse_args()

    tids = sorted(set(args.task_ids))
    print(f"多任务 P3+P4 Pipeline")
    print(f"  suite    = {args.suite}")
    print(f"  task_ids = {tids}")
    print(f"  stage    = {args.stage}")
    print(f"  start_frame = {args.start_frame}")

    all_results = {}

    if args.stage in ("compute", "all"):
        all_results["compute"] = stage_compute(tids, args)
        print_summary(tids, all_results["compute"], "P3+P4 计算")

    if args.stage in ("render", "all"):
        all_results["render"] = stage_render(tids, args)
        print_summary(tids, all_results["render"], "视频渲染")

    # 保存汇总结果
    out_file = PROJECT_ROOT / "outputs/multitask_p3p4_results.json"
    out_file.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n汇总结果已保存：{out_file}")


if __name__ == "__main__":
    main()
