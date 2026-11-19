#!/usr/bin/env python3
"""
汇总LIBERO-plus评估结果
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, help="评估结果目录")
    parser.add_argument("--output", required=True, help="输出JSON文件")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    baseline_dir = results_dir / "baseline"
    phyvla_dir = results_dir / "phyvla"

    # 汇总数据结构
    summary = {
        "overall": {},
        "by_suite": {},
        "by_variant": {},
        "by_config": {}
    }

    suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    variants = ["camera", "robot", "language", "light", "background", "noise", "layout"]

    # 统计变量
    total_baseline_success = 0
    total_phyvla_success = 0
    total_trials = 0

    suite_stats = {s: {"baseline": 0, "phyvla": 0, "trials": 0} for s in suites}
    variant_stats = {v: {"baseline": 0, "phyvla": 0, "trials": 0} for v in variants}

    # 遍历所有配置
    for suite in suites:
        for variant in variants:
            config_key = f"{suite}_{variant}"

            # 读取baseline结果
            baseline_file = baseline_dir / config_key / f"{config_key}_results.json"
            if not baseline_file.exists():
                print(f"⚠️  缺少baseline结果: {baseline_file}")
                continue

            with open(baseline_file) as f:
                baseline_data = json.load(f)

            # 读取PhyVLA结果
            phyvla_file = phyvla_dir / config_key / f"{config_key}_results.json"
            if not phyvla_file.exists():
                print(f"⚠️  缺少PhyVLA结果: {phyvla_file}")
                continue

            with open(phyvla_file) as f:
                phyvla_data = json.load(f)

            # 提取成功率
            baseline_rate = baseline_data["overall_success_rate"]
            phyvla_rate = phyvla_data["overall_success_rate"]
            trials = baseline_data["total_trials"]

            baseline_succ = baseline_data["total_successes"]
            phyvla_succ = phyvla_data["total_successes"]

            improvement = phyvla_rate - baseline_rate
            relative_improvement = (improvement / baseline_rate * 100) if baseline_rate > 0 else 0

            # 记录配置级别结果
            summary["by_config"][config_key] = {
                "suite": suite,
                "variant": variant,
                "baseline_rate": baseline_rate,
                "phyvla_rate": phyvla_rate,
                "improvement": improvement,
                "relative_improvement_pct": relative_improvement,
                "trials": trials
            }

            # 累计统计
            total_baseline_success += baseline_succ
            total_phyvla_success += phyvla_succ
            total_trials += trials

            suite_stats[suite]["baseline"] += baseline_succ
            suite_stats[suite]["phyvla"] += phyvla_succ
            suite_stats[suite]["trials"] += trials

            variant_stats[variant]["baseline"] += baseline_succ
            variant_stats[variant]["phyvla"] += phyvla_succ
            variant_stats[variant]["trials"] += trials

            print(f"✓ {config_key}: {baseline_rate:.3f} → {phyvla_rate:.3f} (+{improvement:.3f}, {relative_improvement:+.1f}%)")

    # Overall统计
    summary["overall"] = {
        "baseline_rate": total_baseline_success / total_trials if total_trials > 0 else 0,
        "phyvla_rate": total_phyvla_success / total_trials if total_trials > 0 else 0,
        "improvement": (total_phyvla_success - total_baseline_success) / total_trials if total_trials > 0 else 0,
        "total_trials": total_trials
    }

    # 按suite统计
    for suite, stats in suite_stats.items():
        if stats["trials"] > 0:
            summary["by_suite"][suite] = {
                "baseline_rate": stats["baseline"] / stats["trials"],
                "phyvla_rate": stats["phyvla"] / stats["trials"],
                "improvement": (stats["phyvla"] - stats["baseline"]) / stats["trials"],
                "trials": stats["trials"]
            }

    # 按variant统计
    for variant, stats in variant_stats.items():
        if stats["trials"] > 0:
            summary["by_variant"][variant] = {
                "baseline_rate": stats["baseline"] / stats["trials"],
                "phyvla_rate": stats["phyvla"] / stats["trials"],
                "improvement": (stats["phyvla"] - stats["baseline"]) / stats["trials"],
                "trials": stats["trials"]
            }

    # 保存
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("汇总完成")
    print(f"{'='*60}")
    print(f"Overall: {summary['overall']['baseline_rate']:.3f} → {summary['overall']['phyvla_rate']:.3f} "
          f"(+{summary['overall']['improvement']:.3f})")
    print(f"已保存: {args.output}")


if __name__ == "__main__":
    main()
