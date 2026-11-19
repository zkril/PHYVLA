#!/usr/bin/env python3
"""
可视化LIBERO-plus评估结果
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="汇总结果JSON文件")
    args = parser.parse_args()

    with open(args.results) as f:
        summary = json.load(f)

    output_dir = Path(args.results).parent
    sns.set_theme(style="whitegrid")

    # 1. Overall对比
    fig, ax = plt.subplots(figsize=(8, 5))
    overall = summary["overall"]
    methods = ["Baseline", "PhyVLA"]
    rates = [overall["baseline_rate"] * 100, overall["phyvla_rate"] * 100]
    colors = ["#3498db", "#e74c3c"]

    bars = ax.bar(methods, rates, color=colors, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("LIBERO-plus Overall Performance", fontsize=14, fontweight="bold")
    ax.set_ylim([0, 100])

    # 标注数值
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 标注提升
    improvement = overall["improvement"] * 100
    ax.text(0.5, max(rates) + 5, f'Improvement: +{improvement:.1f}%',
            ha='center', fontsize=12, color='green', fontweight='bold',
            transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(output_dir / "overall_comparison.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_dir / 'overall_comparison.png'}")
    plt.close()

    # 2. 按Suite对比
    fig, ax = plt.subplots(figsize=(10, 6))
    suites = list(summary["by_suite"].keys())
    baseline_rates = [summary["by_suite"][s]["baseline_rate"] * 100 for s in suites]
    phyvla_rates = [summary["by_suite"][s]["phyvla_rate"] * 100 for s in suites]

    x = np.arange(len(suites))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_rates, width, label='Baseline', color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, phyvla_rates, width, label='PhyVLA', color='#e74c3c', alpha=0.8, edgecolor='black')

    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("Performance by Suite", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("libero_", "") for s in suites], fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 100])

    # 标注数值
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "by_suite.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_dir / 'by_suite.png'}")
    plt.close()

    # 3. 按Variant对比
    fig, ax = plt.subplots(figsize=(12, 6))
    variants = list(summary["by_variant"].keys())
    baseline_rates = [summary["by_variant"][v]["baseline_rate"] * 100 for v in variants]
    phyvla_rates = [summary["by_variant"][v]["phyvla_rate"] * 100 for v in variants]

    x = np.arange(len(variants))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_rates, width, label='Baseline', color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, phyvla_rates, width, label='PhyVLA', color='#e74c3c', alpha=0.8, edgecolor='black')

    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("Performance by Variant", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, fontsize=11, rotation=15)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 100])

    # 标注数值
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "by_variant.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_dir / 'by_variant.png'}")
    plt.close()

    # 4. Improvement热力图 (Suite × Variant)
    suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    variants = ["camera", "robot", "language", "light", "background", "noise", "layout"]

    improvement_matrix = np.zeros((len(suites), len(variants)))

    for i, suite in enumerate(suites):
        for j, variant in enumerate(variants):
            config_key = f"{suite}_{variant}"
            if config_key in summary["by_config"]:
                improvement_matrix[i, j] = summary["by_config"][config_key]["improvement"] * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(improvement_matrix, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
                xticklabels=[v.capitalize() for v in variants],
                yticklabels=[s.replace("libero_", "") for s in suites],
                cbar_kws={'label': 'Improvement (%)'}, ax=ax)
    ax.set_title("PhyVLA Improvement Heatmap (Suite × Variant)", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "improvement_heatmap.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_dir / 'improvement_heatmap.png'}")
    plt.close()

    print(f"\n✅ 所有可视化已保存至: {output_dir}")


if __name__ == "__main__":
    main()
