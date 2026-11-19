#!/usr/bin/env python3
"""生成Baseline vs PhyVLA对比报告
位置: scripts/generate_comparison_report.py
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_task_results(results_dir: Path, task_ids: List[int]) -> Dict:
    """加载某个方法的所有任务结果"""
    results = {}
    for task_id in task_ids:
        json_path = results_dir / f"task_{task_id}.json"
        if json_path.exists():
            with open(json_path) as f:
                results[task_id] = json.load(f)
        else:
            print(f"Warning: {json_path} not found")
    return results


def compute_statistics(results: Dict) -> Dict:
    """计算统计数据"""
    success_rates = [r['success_rate'] for r in results.values()]
    avg_steps = [r['avg_steps'] for r in results.values()]
    avg_rewards = [r['avg_reward'] for r in results.values()]

    return {
        'num_tasks': len(results),
        'mean_success_rate': float(np.mean(success_rates)),
        'std_success_rate': float(np.std(success_rates)),
        'min_success_rate': float(np.min(success_rates)),
        'max_success_rate': float(np.max(success_rates)),
        'mean_steps': float(np.mean(avg_steps)),
        'mean_reward': float(np.mean(avg_rewards)),
    }


def compare_methods(baseline: Dict, phyvla: Dict) -> Dict:
    """对比两种方法"""
    task_ids = sorted(set(baseline.keys()) & set(phyvla.keys()))

    per_task_comparison = {}
    improvements = []

    for task_id in task_ids:
        b = baseline[task_id]
        p = phyvla[task_id]

        sr_baseline = b['success_rate']
        sr_phyvla = p['success_rate']
        improvement = sr_phyvla - sr_baseline
        improvements.append(improvement)

        per_task_comparison[f"task_{task_id}"] = {
            'task_name': b.get('task_name', f"task_{task_id}"),
            'baseline_sr': sr_baseline,
            'phyvla_sr': sr_phyvla,
            'improvement': improvement,
            'relative_improvement': (improvement / max(sr_baseline, 0.01)) if sr_baseline > 0 else None,
        }

    overall_stats = {
        'mean_improvement': float(np.mean(improvements)),
        'std_improvement': float(np.std(improvements)),
        'tasks_improved': sum(1 for x in improvements if x > 0),
        'tasks_degraded': sum(1 for x in improvements if x < 0),
        'tasks_unchanged': sum(1 for x in improvements if x == 0),
    }

    return {
        'per_task': per_task_comparison,
        'overall': overall_stats,
    }


def generate_report(
    baseline_dir: Path,
    phyvla_dir: Path,
    suite: str,
    output_path: Path,
):
    """生成完整对比报告"""

    # 自动检测task IDs
    baseline_files = list(baseline_dir.glob("task_*.json"))
    task_ids = sorted([
        int(f.stem.split('_')[1])
        for f in baseline_files
    ])

    print(f"Loading results for {len(task_ids)} tasks: {task_ids}")

    # 加载结果
    baseline_results = load_task_results(baseline_dir, task_ids)
    phyvla_results = load_task_results(phyvla_dir, task_ids)

    # 计算统计
    baseline_stats = compute_statistics(baseline_results)
    phyvla_stats = compute_statistics(phyvla_results)

    # 对比
    comparison = compare_methods(baseline_results, phyvla_results)

    # 生成报告
    report = {
        'suite': suite,
        'num_tasks': len(task_ids),
        'task_ids': task_ids,

        'baseline': {
            'method': 'StarVLA (baseline)',
            'statistics': baseline_stats,
        },

        'phyvla': {
            'method': 'PhyVLA (physics-augmented)',
            'statistics': phyvla_stats,
        },

        'comparison': comparison,

        'summary': {
            'baseline_mean_sr': baseline_stats['mean_success_rate'],
            'phyvla_mean_sr': phyvla_stats['mean_success_rate'],
            'absolute_improvement': comparison['overall']['mean_improvement'],
            'relative_improvement': (
                comparison['overall']['mean_improvement'] /
                max(baseline_stats['mean_success_rate'], 0.01)
            ),
            'tasks_improved': comparison['overall']['tasks_improved'],
            'tasks_degraded': comparison['overall']['tasks_degraded'],
        }
    }

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {output_path}")

    # 打印摘要
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"Suite: {suite}")
    print(f"Tasks: {len(task_ids)}")
    print()
    print(f"Baseline Mean Success Rate:  {baseline_stats['mean_success_rate']:.1%}")
    print(f"PhyVLA Mean Success Rate:    {phyvla_stats['mean_success_rate']:.1%}")
    print()
    print(f"Absolute Improvement:        {comparison['overall']['mean_improvement']:+.1%}")
    print(f"Relative Improvement:        {report['summary']['relative_improvement']:+.1%}")
    print()
    print(f"Tasks Improved:              {comparison['overall']['tasks_improved']}")
    print(f"Tasks Degraded:              {comparison['overall']['tasks_degraded']}")
    print(f"Tasks Unchanged:             {comparison['overall']['tasks_unchanged']}")
    print("="*60)

    # 打印per-task结果
    print("\nPER-TASK RESULTS:")
    print("-"*60)
    print(f"{'Task':<15} {'Baseline':<12} {'PhyVLA':<12} {'Change':<12}")
    print("-"*60)

    for task_id in task_ids:
        key = f"task_{task_id}"
        if key in comparison['per_task']:
            tc = comparison['per_task'][key]
            task_name = tc['task_name'][:12]
            baseline_sr = tc['baseline_sr']
            phyvla_sr = tc['phyvla_sr']
            improvement = tc['improvement']

            symbol = "↑" if improvement > 0 else "↓" if improvement < 0 else "="

            print(f"{task_name:<15} {baseline_sr:>10.1%}  {phyvla_sr:>10.1%}  {symbol} {improvement:>+8.1%}")

    print("-"*60)


def main():
    parser = argparse.ArgumentParser(description="Generate comparison report")
    parser.add_argument("--baseline-dir", type=str, required=True)
    parser.add_argument("--phyvla-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--suite", type=str, default="libero_spatial")
    args = parser.parse_args()

    generate_report(
        baseline_dir=Path(args.baseline_dir),
        phyvla_dir=Path(args.phyvla_dir),
        suite=args.suite,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
