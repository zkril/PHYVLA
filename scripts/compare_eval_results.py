#!/usr/bin/env python3
"""对比LIBERO-plus Baseline vs PhyVLA评估结果"""

import argparse
import json
from pathlib import Path
import numpy as np


def load_results(results_dir: Path) -> dict:
    """加载评估结果JSON文件"""
    results_file = results_dir / "results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Results not found: {results_file}")

    with open(results_file) as f:
        return json.load(f)


def compute_statistics(results: dict) -> dict:
    """计算统计指标"""
    success_rates = []

    for task_name, task_results in results.items():
        if isinstance(task_results, dict) and "success_rate" in task_results:
            success_rates.append(task_results["success_rate"])

    if not success_rates:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "count": 0
        }

    return {
        "mean": np.mean(success_rates),
        "std": np.std(success_rates),
        "min": np.min(success_rates),
        "max": np.max(success_rates),
        "count": len(success_rates)
    }


def compare_results(baseline_dir: Path, phyvla_dir: Path, suite: str, output: Path = None):
    """对比两个评估结果"""

    print("=" * 70)
    print(f"LIBERO-plus 评估对比: {suite}")
    print("=" * 70)
    print()

    # 加载结果
    print("加载结果...")
    try:
        baseline_results = load_results(baseline_dir)
        phyvla_results = load_results(phyvla_dir)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    # 计算统计
    baseline_stats = compute_statistics(baseline_results)
    phyvla_stats = compute_statistics(phyvla_results)

    # 总体对比
    print("总体统计:")
    print()
    print("  Baseline (StarVLA):")
    print(f"    平均成功率: {baseline_stats['mean']*100:.1f}%")
    print(f"    标准差:     {baseline_stats['std']:.3f}")
    print(f"    最小/最大:  {baseline_stats['min']*100:.1f}% / {baseline_stats['max']*100:.1f}%")
    print(f"    任务数:     {baseline_stats['count']}")
    print()

    print("  PhyVLA (StarVLA + Physics):")
    print(f"    平均成功率: {phyvla_stats['mean']*100:.1f}%")
    print(f"    标准差:     {phyvla_stats['std']:.3f}")
    print(f"    最小/最大:  {phyvla_stats['min']*100:.1f}% / {phyvla_stats['max']*100:.1f}%")
    print(f"    任务数:     {phyvla_stats['count']}")
    print()

    # 改进指标
    abs_improvement = (phyvla_stats['mean'] - baseline_stats['mean']) * 100
    rel_improvement = ((phyvla_stats['mean'] - baseline_stats['mean']) / baseline_stats['mean'] * 100) if baseline_stats['mean'] > 0 else 0

    print("  改进:")
    print(f"    绝对改进:   {abs_improvement:+.1f}%")
    print(f"    相对改进:   {rel_improvement:+.1f}%")
    print()

    # 逐任务对比
    print("-" * 70)
    print("逐任务对比:")
    print("-" * 70)
    print(f"{'Task':<20} {'Baseline':>12} {'PhyVLA':>12} {'Δ Abs':>12} {'Δ Rel':>12}")
    print("-" * 70)

    task_comparisons = []
    for task_name in sorted(baseline_results.keys()):
        if task_name not in phyvla_results:
            continue

        baseline_sr = baseline_results[task_name].get("success_rate", 0)
        phyvla_sr = phyvla_results[task_name].get("success_rate", 0)

        abs_diff = (phyvla_sr - baseline_sr) * 100
        rel_diff = ((phyvla_sr - baseline_sr) / baseline_sr * 100) if baseline_sr > 0 else 0

        print(f"{task_name:<20} {baseline_sr*100:>11.1f}% {phyvla_sr*100:>11.1f}% "
              f"{abs_diff:>11.1f}% {rel_diff:>11.1f}%")

        task_comparisons.append({
            "task": task_name,
            "baseline": baseline_sr,
            "phyvla": phyvla_sr,
            "abs_improvement": abs_diff,
            "rel_improvement": rel_diff
        })

    print("-" * 70)
    print()

    # 胜率统计
    wins = sum(1 for tc in task_comparisons if tc["phyvla"] > tc["baseline"])
    ties = sum(1 for tc in task_comparisons if tc["phyvla"] == tc["baseline"])
    losses = sum(1 for tc in task_comparisons if tc["phyvla"] < tc["baseline"])

    print("胜率统计:")
    print(f"  PhyVLA胜: {wins} / {len(task_comparisons)} ({wins/len(task_comparisons)*100:.1f}%)")
    print(f"  平局:     {ties} / {len(task_comparisons)} ({ties/len(task_comparisons)*100:.1f}%)")
    print(f"  PhyVLA负: {losses} / {len(task_comparisons)} ({losses/len(task_comparisons)*100:.1f}%)")
    print()

    # 保存对比结果
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        comparison_data = {
            "suite": suite,
            "baseline": {
                "dir": str(baseline_dir),
                "stats": baseline_stats
            },
            "phyvla": {
                "dir": str(phyvla_dir),
                "stats": phyvla_stats
            },
            "improvements": {
                "absolute": abs_improvement,
                "relative": rel_improvement
            },
            "win_rate": {
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "total": len(task_comparisons)
            },
            "task_comparisons": task_comparisons
        }

        with open(output, "w") as f:
            json.dump(comparison_data, f, indent=2)

        print(f"对比结果已保存: {output}")
        print()


def main():
    parser = argparse.ArgumentParser(description="对比LIBERO-plus评估结果")
    parser.add_argument("--baseline-dir", type=Path, required=True,
                        help="Baseline结果目录")
    parser.add_argument("--phyvla-dir", type=Path, required=True,
                        help="PhyVLA结果目录")
    parser.add_argument("--suite", default="libero_spatial",
                        help="Suite名称")
    parser.add_argument("--output", type=Path,
                        help="输出对比JSON文件路径")

    args = parser.parse_args()

    compare_results(args.baseline_dir, args.phyvla_dir, args.suite, args.output)


if __name__ == "__main__":
    main()
