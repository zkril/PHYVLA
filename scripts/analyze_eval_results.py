#!/usr/bin/env python3
"""分析PhyVLA LIBERO评估结果"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

sns.set_style("whitegrid")


def load_results(path):
    with open(path, 'r') as f:
        return json.load(f)


def analyze_results(baseline_path, phyvla_path, output_path):
    baseline = load_results(baseline_path)
    phyvla = load_results(phyvla_path)

    print("="*80)
    print("PhyVLA vs Baseline 评估结果分析")
    print("="*80)

    # 整体成功率
    base_sr = baseline['overall_success_rate']
    phy_sr = phyvla['overall_success_rate']
    improvement = (phy_sr - base_sr) * 100

    print(f"\n整体成功率:")
    print(f"  Baseline: {base_sr*100:.1f}%")
    print(f"  PhyVLA:   {phy_sr*100:.1f}%")
    print(f"  提升:     {improvement:+.1f} 百分点")

    # 任务级别统计
    print(f"\n任务级别成功率:")
    print(f"{'Task':<8} {'Baseline':<12} {'PhyVLA':<12} {'Improvement':<12}")
    print("-"*50)

    base_episodes = baseline['episodes']
    phy_episodes = phyvla['episodes']

    task_ids = sorted(set(ep['task_id'] for ep in base_episodes))
    task_stats = []

    for tid in task_ids:
        base_task = [ep for ep in base_episodes if ep['task_id'] == tid]
        phy_task = [ep for ep in phy_episodes if ep['task_id'] == tid]

        base_sr_task = np.mean([ep['success'] for ep in base_task])
        phy_sr_task = np.mean([ep['success'] for ep in phy_task])
        imp = (phy_sr_task - base_sr_task) * 100

        task_name = base_task[0]['task_name']
        print(f"Task {tid:<3} {base_sr_task*100:>6.1f}%     {phy_sr_task*100:>6.1f}%     {imp:+6.1f}pp")

        task_stats.append({
            'task_id': tid,
            'task_name': task_name,
            'baseline_sr': base_sr_task,
            'phyvla_sr': phy_sr_task,
            'improvement': imp,
        })

    # 统计检验
    print(f"\n统计显著性检验:")
    base_srs = [t['baseline_sr'] for t in task_stats]
    phy_srs = [t['phyvla_sr'] for t in task_stats]

    t_stat, p_val = stats.ttest_rel(phy_srs, base_srs)
    print(f"  配对t检验: t={t_stat:.3f}, p={p_val:.4f}")

    if p_val < 0.05:
        print(f"  ✓ 差异显著 (p < 0.05)")
    else:
        print(f"  差异不显著 (p >= 0.05)")

    # 生成可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 任务成功率对比
    ax = axes[0, 0]
    x = np.arange(len(task_ids))
    width = 0.35

    base_bars = [t['baseline_sr']*100 for t in task_stats]
    phy_bars = [t['phyvla_sr']*100 for t in task_stats]

    ax.bar(x - width/2, base_bars, width, label='Baseline', color='#3498db', alpha=0.8)
    ax.bar(x + width/2, phy_bars, width, label='PhyVLA', color='#e74c3c', alpha=0.8)

    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Task-wise Success Rate')
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{t}" for t in task_ids])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 2. 改进幅度
    ax = axes[0, 1]
    improvements = [t['improvement'] for t in task_stats]
    colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]

    ax.barh(range(len(task_ids)), improvements, color=colors, alpha=0.8)
    ax.set_yticks(range(len(task_ids)))
    ax.set_yticklabels([f"Task {t}" for t in task_ids])
    ax.set_xlabel('Improvement (percentage points)')
    ax.set_title('PhyVLA vs Baseline Improvement')
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.grid(axis='x', alpha=0.3)

    # 3. 成功率分布
    ax = axes[1, 0]
    bp = ax.boxplot([base_bars, phy_bars], labels=['Baseline', 'PhyVLA'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#e74c3c')
    for patch in bp['boxes']:
        patch.set_alpha(0.6)

    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate Distribution')
    ax.grid(axis='y', alpha=0.3)

    # 4. 散点图对比
    ax = axes[1, 1]
    ax.scatter(base_bars, phy_bars, s=100, alpha=0.6, color='#9b59b6')
    ax.plot([0, 100], [0, 100], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Baseline Success Rate (%)')
    ax.set_ylabel('PhyVLA Success Rate (%)')
    ax.set_title('Task-wise Comparison')
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 105])
    ax.set_ylim([0, 105])

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n可视化已保存: {output_path}")

    # 保存汇总JSON
    summary_path = output_path.parent / f"{output_path.stem}_summary.json"
    summary = {
        'baseline': {
            'success_rate': base_sr,
            'num_episodes': baseline['total_episodes'],
            'num_successes': baseline['total_successes'],
        },
        'phyvla': {
            'success_rate': phy_sr,
            'num_episodes': phyvla['total_episodes'],
            'num_successes': phyvla['total_successes'],
        },
        'improvement': improvement,
        'statistical_test': {
            't_statistic': float(t_stat),
            'p_value': float(p_val),
        },
        'task_results': task_stats,
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"汇总已保存: {summary_path}")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--phyvla', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    analyze_results(args.baseline, args.phyvla, args.output)
