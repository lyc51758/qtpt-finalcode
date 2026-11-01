"""
Generate comparison plots for Darkroom baseline experiments.
Reads log.csv from each results directory and produces:
1. Bar chart: Best average reward per method (random vs expert data)
2. Learning curves: Reward vs epoch for all methods

Usage (on server):
    python plot_darkroom_results.py --results_dir ../qtpt_baselines_darkroom/results
"""

import os
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS = ['qtpt', 'spt', 'cql', 'iql', 'qsft', 'qdt']
METHOD_LABELS = {
    'qtpt': 'QTPT (Ours)',
    'spt': 'SPT',
    'cql': 'CQL',
    'iql': 'IQL',
    'qsft': 'Q-SFT',
    'qdt': 'QDT',
}
METHOD_COLORS = {
    'qtpt': '#E74C3C',
    'spt': '#3498DB',
    'cql': '#2ECC71',
    'iql': '#9B59B6',
    'qsft': '#F39C12',
    'qdt': '#1ABC9C',
}
DATA_LABELS = {'0.0': 'Random Data', '1.0': 'Expert Data'}


def load_log(path, method=None):
    """Load log.csv and return list of (epoch, loss, reward).
    For QDT, uses DT-phase results (the final policy) if available,
    otherwise falls back to Q-phase results.
    """
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            phase = row.get('phase', None)
            rows.append({
                'phase': phase,
                'epoch': int(row['epoch']),
                'train_loss': float(row['train_loss']),
                'avg_reward': float(row['avg_reward']),
            })
    if method == 'qdt':
        dt_rows = [r for r in rows if r['phase'] == 'dt']
        q_rows = [r for r in rows if r['phase'] == 'q']
        return dt_rows if dt_rows else q_rows
    return rows


def get_best_reward(rows):
    """Return best avg_reward across all epochs."""
    return max(r['avg_reward'] for r in rows)


def get_rewards_curve(rows):
    """Return (epochs, rewards) arrays."""
    epochs = [r['epoch'] for r in rows]
    rewards = [r['avg_reward'] for r in rows]
    return np.array(epochs), np.array(rewards)


def smooth(values, weight=0.6):
    """Exponential moving average smoothing."""
    smoothed = []
    last = values[0]
    for v in values:
        last = weight * last + (1 - weight) * v
        smoothed.append(last)
    return np.array(smoothed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.results_dir

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect results
    all_data = {}  # {mix: {method: rows}}
    for mix in ['0.0', '1.0']:
        all_data[mix] = {}
        for method in METHODS:
            log_path = os.path.join(args.results_dir, f'{method}_mix{mix}_seed{args.seed}', 'log.csv')
            if os.path.exists(log_path):
                all_data[mix][method] = load_log(log_path, method=method)
                print(f"Loaded {method} mix={mix}: {len(all_data[mix][method])} epochs, "
                      f"best={get_best_reward(all_data[mix][method]):.2f}")
            else:
                print(f"Missing: {log_path}")

    # ---- Plot 1: Bar chart of best rewards ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax_idx, mix in enumerate(['0.0', '1.0']):
        ax = axes[ax_idx]
        methods_present = [m for m in METHODS if m in all_data[mix]]
        best_rewards = [get_best_reward(all_data[mix][m]) for m in methods_present]
        labels = [METHOD_LABELS[m] for m in methods_present]
        colors = [METHOD_COLORS[m] for m in methods_present]

        bars = ax.bar(range(len(methods_present)), best_rewards, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(methods_present)))
        ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=11)
        ax.set_ylabel('Best Average Reward', fontsize=12)
        ax.set_title(f'Darkroom - {DATA_LABELS[mix]}', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars, best_rewards):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.1,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'darkroom_best_reward_bar.png'), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(args.output_dir, 'darkroom_best_reward_bar.pdf'), bbox_inches='tight')
    plt.close()
    print("Saved bar chart.")

    # ---- Plot 2: Learning curves ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax_idx, mix in enumerate(['0.0', '1.0']):
        ax = axes[ax_idx]
        for method in METHODS:
            if method not in all_data[mix]:
                continue
            epochs, rewards = get_rewards_curve(all_data[mix][method])
            smoothed = smooth(rewards, weight=0.6)
            ax.plot(epochs, smoothed, label=METHOD_LABELS[method],
                    color=METHOD_COLORS[method], linewidth=2)
            ax.fill_between(epochs, rewards, smoothed, alpha=0.1, color=METHOD_COLORS[method])

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Average Reward', fontsize=12)
        ax.set_title(f'Darkroom - {DATA_LABELS[mix]}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'darkroom_learning_curves.png'), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(args.output_dir, 'darkroom_learning_curves.pdf'), bbox_inches='tight')
    plt.close()
    print("Saved learning curves.")

    # ---- Summary table ----
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    header = f"{'Method':<15}"
    for mix in ['0.0', '1.0']:
        header += f"{'Random Best':>15}" if mix == '0.0' else f"{'Expert Best':>15}"
    print(header)
    print("-"*45)
    for method in METHODS:
        row = f"{METHOD_LABELS[method]:<15}"
        for mix in ['0.0', '1.0']:
            if method in all_data[mix]:
                best = get_best_reward(all_data[mix][method])
                row += f"{best:>15.2f}"
            else:
                row += f"{'N/A':>15}"
        print(row)
    print("="*70)

    # Save summary as CSV
    summary_path = os.path.join(args.output_dir, 'darkroom_summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Random_Best', 'Expert_Best', 'Random_Last', 'Expert_Last'])
        for method in METHODS:
            row = [METHOD_LABELS[method]]
            for mix in ['0.0', '1.0']:
                if method in all_data[mix]:
                    row.append(f"{get_best_reward(all_data[mix][method]):.2f}")
                else:
                    row.append('N/A')
            for mix in ['0.0', '1.0']:
                if method in all_data[mix]:
                    last = all_data[mix][method][-1]['avg_reward']
                    row.append(f"{last:.2f}")
                else:
                    row.append('N/A')
            writer.writerow(row)
    print(f"Saved summary to {summary_path}")


if __name__ == '__main__':
    main()
