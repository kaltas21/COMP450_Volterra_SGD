"""
Generate Publication-Quality Figures for Volterra SGD Paper
============================================================
Reads existing results and creates properly formatted figures
suitable for academic publication.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

# =============================================================================
# PUBLICATION STYLE CONFIGURATION
# =============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
    'font.size': 12,
    'mathtext.fontset': 'cm',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'axes.axisbelow': True,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'legend.fontsize': 11,
    'legend.framealpha': 0.95,
    'legend.edgecolor': '0.8',
    'lines.linewidth': 2.0,
    'lines.markersize': 6,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color scheme (colorblind-friendly)
COLORS = {
    'sgd': '#E24A33',      # Red
    'volterra': '#348ABD', # Blue
    'lr1': '#E24A33',      # Red (gamma = 0.25)
    'lr2': '#348ABD',      # Blue (gamma = 0.50)
    'lr3': '#988ED5',      # Purple (gamma = 0.90)
}

ASPECT_RATIOS = [0.5, 1.0, 1.5]
N_SAMPLES = 3000

# Output directory
PAPER_DIR = './paper_figures'
os.makedirs(PAPER_DIR, exist_ok=True)


def load_results():
    """Load all experiment results."""
    results = {}
    for exp in ['part2_base', 'part2_nonlinear', 'part2_whitened']:
        path = f'./results/{exp}/all_results.npy'
        if os.path.exists(path):
            results[exp] = np.load(path, allow_pickle=True).item()
            print(f"Loaded {exp}")
        else:
            print(f"Warning: {path} not found")
    return results


def figure1_loss_curves(results, exp_name, title, filename):
    """
    Figure: Loss curves comparing SGD (empirical) vs Volterra (theory).
    Shows convergence dynamics for different aspect ratios and learning rates.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    lr_colors = [COLORS['lr1'], COLORS['lr2'], COLORS['lr3']]
    lr_labels = [r'$\gamma = 0.25\gamma_{safe}$',
                 r'$\gamma = 0.50\gamma_{safe}$',
                 r'$\gamma = 0.90\gamma_{safe}$']

    for idx, r in enumerate(ASPECT_RATIOS):
        ax = axes[idx]
        res = results[r]

        for i, (key, data) in enumerate(sorted(res['results_safe'].items(), key=lambda x: float(x[0]))):
            loss_every = data.get('loss_every', 1)
            t_sgd = np.arange(len(data['sgd_mean'])) * loss_every / N_SAMPLES

            # Plot SGD (solid line with shaded std)
            ax.plot(t_sgd, data['sgd_mean'], color=lr_colors[i], lw=2, alpha=0.9)
            ax.fill_between(t_sgd,
                           data['sgd_mean'] - data['sgd_std'],
                           data['sgd_mean'] + data['sgd_std'],
                           color=lr_colors[i], alpha=0.15)

            # Plot Volterra (dashed line)
            ax.plot(data['t_theory'], data['psi'], '--', color=lr_colors[i], lw=2)

        ax.set_xlabel('Epochs', fontsize=13)
        if idx == 0:
            ax.set_ylabel('Training Loss', fontsize=13)
        ax.set_yscale('log')

        sr = res['spectral_info']['spectral_ratio']
        ax.set_title(f'$r = {r}$ (SR = {sr:.0f})', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 20)

    # Create legend
    legend_elements = [
        Line2D([0], [0], color='gray', lw=2, linestyle='-', label='SGD (empirical)'),
        Line2D([0], [0], color='gray', lw=2, linestyle='--', label='Volterra (theory)'),
        Line2D([0], [0], color='white', lw=0, label=''),  # Spacer
        Line2D([0], [0], color=lr_colors[0], lw=2, label=lr_labels[0]),
        Line2D([0], [0], color=lr_colors[1], lw=2, label=lr_labels[1]),
        Line2D([0], [0], color=lr_colors[2], lw=2, label=lr_labels[2]),
    ]
    axes[2].legend(handles=legend_elements, loc='upper right', fontsize=10,
                   framealpha=0.95, edgecolor='0.8')

    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_DIR, f'{filename}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PAPER_DIR, f'{filename}.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def figure2_scatter_plots(results, exp_name, title, filename):
    """
    Figure: Scatter plot of Volterra predictions vs actual SGD loss.
    Shows prediction accuracy with y=x reference line.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for idx, r in enumerate(ASPECT_RATIOS):
        ax = axes[idx]
        res = results[r]

        all_volterra = []
        all_sgd = []

        for key, data in res['results_safe'].items():
            loss_every = data.get('loss_every', 1)
            t_sgd = np.arange(len(data['sgd_mean'])) * loss_every / N_SAMPLES
            volterra_interp = np.interp(t_sgd, data['t_theory'], data['psi'])
            step = max(1, len(t_sgd) // 100)
            all_volterra.extend(volterra_interp[::step])
            all_sgd.extend(data['sgd_mean'][::step])

        all_volterra = np.array(all_volterra)
        all_sgd = np.array(all_sgd)

        valid = (all_volterra > 0) & (all_sgd > 0) & np.isfinite(all_volterra) & np.isfinite(all_sgd)
        all_volterra = all_volterra[valid]
        all_sgd = all_sgd[valid]

        # Scatter plot
        ax.scatter(all_volterra, all_sgd, alpha=0.5, s=20, c=COLORS['volterra'],
                   edgecolors='none', label='Data points')

        # y=x reference line
        min_val = min(all_volterra.min(), all_sgd.min()) * 0.9
        max_val = max(all_volterra.max(), all_sgd.max()) * 1.1
        ax.plot([min_val, max_val], [min_val, max_val], '--', color=COLORS['sgd'],
                lw=2, label='$y = x$ (perfect)')

        # Compute metrics
        corr, _ = stats.pearsonr(all_volterra, all_sgd)
        ss_res = np.sum((all_sgd - all_volterra) ** 2)
        ss_tot = np.sum((all_sgd - np.mean(all_sgd)) ** 2)
        r2 = 1 - ss_res / ss_tot

        ax.set_xlabel('Volterra Predicted Loss', fontsize=13)
        if idx == 0:
            ax.set_ylabel('Actual SGD Loss', fontsize=13)
        ax.set_title(f'$r = {r}$\n(Pearson $R$ = {corr:.3f}, $R^2$ = {r2:.3f})', fontsize=12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

        if idx == 2:
            ax.legend(loc='lower right', fontsize=10)

    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_DIR, f'{filename}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PAPER_DIR, f'{filename}.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def figure3_metrics_comparison(all_results):
    """
    Figure: Comprehensive metrics comparison across all experiments.
    Shows Pearson R, R², MAPE, and Mean Ratio for Base, Nonlinear, Whitened.
    """
    experiments = ['part2_base', 'part2_nonlinear', 'part2_whitened']
    exp_labels = ['Base (Linear)', 'Nonlinear (ReLU)', 'Whitened']

    metrics = {exp: {'pearson': [], 'r2': [], 'mape': [], 'ratio': []} for exp in experiments}

    for exp in experiments:
        if exp not in all_results:
            continue
        results = all_results[exp]

        for r in ASPECT_RATIOS:
            res = results[r]
            all_volt, all_sgd = [], []

            for data in res['results_safe'].values():
                loss_every = data.get('loss_every', 1)
                t_sgd = np.arange(len(data['sgd_mean'])) * loss_every / N_SAMPLES
                volt_interp = np.interp(t_sgd, data['t_theory'], data['psi'])
                all_volt.extend(volt_interp)
                all_sgd.extend(data['sgd_mean'])

            volt = np.array(all_volt)
            sgd = np.array(all_sgd)
            valid = np.isfinite(volt) & np.isfinite(sgd) & (sgd > 0)
            volt, sgd = volt[valid], sgd[valid]

            pearson_r, _ = stats.pearsonr(volt, sgd)
            ss_res = np.sum((sgd - volt) ** 2)
            ss_tot = np.sum((sgd - np.mean(sgd)) ** 2)
            r2 = 1 - ss_res / ss_tot
            mape = np.mean(np.abs(volt - sgd) / sgd) * 100
            ratio = np.mean(volt / sgd)

            metrics[exp]['pearson'].append(pearson_r)
            metrics[exp]['r2'].append(r2)
            metrics[exp]['mape'].append(mape)
            metrics[exp]['ratio'].append(ratio)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    x = np.arange(len(ASPECT_RATIOS))
    width = 0.25
    exp_colors = ['#E24A33', '#348ABD', '#988ED5']

    # Plot 1: Pearson R
    ax = axes[0, 0]
    for i, (exp, label) in enumerate(zip(experiments, exp_labels)):
        if exp in metrics:
            ax.bar(x + i*width - width, metrics[exp]['pearson'], width,
                   label=label, color=exp_colors[i], edgecolor='black', lw=0.5)
    ax.set_ylabel('Pearson $R$', fontsize=13)
    ax.set_title('(a) Shape Similarity (Pearson Correlation)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f'$r={r}$' for r in ASPECT_RATIOS])
    ax.set_ylim(0.9, 1.01)
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: R² vs y=x
    ax = axes[0, 1]
    for i, (exp, label) in enumerate(zip(experiments, exp_labels)):
        if exp in metrics:
            ax.bar(x + i*width - width, metrics[exp]['r2'], width,
                   label=label, color=exp_colors[i], edgecolor='black', lw=0.5)
    ax.set_ylabel('$R^2$ (vs $y=x$)', fontsize=13)
    ax.set_title('(b) Prediction Accuracy ($R^2$ relative to $y=x$)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f'$r={r}$' for r in ASPECT_RATIOS])
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: MAPE
    ax = axes[1, 0]
    for i, (exp, label) in enumerate(zip(experiments, exp_labels)):
        if exp in metrics:
            ax.bar(x + i*width - width, metrics[exp]['mape'], width,
                   label=label, color=exp_colors[i], edgecolor='black', lw=0.5)
    ax.set_ylabel('MAPE (%)', fontsize=13)
    ax.set_title('(c) Mean Absolute Percentage Error', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f'$r={r}$' for r in ASPECT_RATIOS])
    ax.axhline(y=20, color='green', linestyle='--', alpha=0.5, label='20% threshold')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Mean Ratio
    ax = axes[1, 1]
    for i, (exp, label) in enumerate(zip(experiments, exp_labels)):
        if exp in metrics:
            ax.bar(x + i*width - width, metrics[exp]['ratio'], width,
                   label=label, color=exp_colors[i], edgecolor='black', lw=0.5)
    ax.set_ylabel('Mean Ratio (Volterra/SGD)', fontsize=13)
    ax.set_title('(d) Prediction Bias (ideal = 1.0)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f'$r={r}$' for r in ASPECT_RATIOS])
    ax.axhline(y=1.0, color='green', linestyle='--', lw=2, alpha=0.7)
    ax.set_ylim(0.7, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Volterra Theory Prediction Quality Across Experiments',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_DIR, 'fig_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PAPER_DIR, 'fig_metrics_comparison.pdf'), bbox_inches='tight')
    plt.close()
    print("  Saved fig_metrics_comparison")


def figure4_eigenvalue_spectrum(all_results):
    """
    Figure: Eigenvalue distribution comparison across experiments.
    """
    experiments = ['part2_base', 'part2_nonlinear', 'part2_whitened']
    exp_labels = ['Base', 'Nonlinear', 'Whitened']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for idx, r in enumerate(ASPECT_RATIOS):
        ax = axes[idx]

        for i, (exp, label) in enumerate(zip(experiments, exp_labels)):
            if exp not in all_results:
                continue
            eigvals = all_results[exp][r]['eigvals']

            ax.hist(eigvals, bins=50, density=True, alpha=0.5,
                    label=f'{label} (SR={np.max(eigvals)/np.mean(eigvals):.0f})')

        ax.set_xlabel(r'Eigenvalue $\lambda$', fontsize=13)
        if idx == 0:
            ax.set_ylabel('Density', fontsize=13)
        ax.set_title(f'$r = {r}$', fontsize=13)
        ax.set_xlim(0, 15)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Eigenvalue Distribution of Feature Matrix $(1/n)A^TA$',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_DIR, 'fig_eigenvalue_spectrum.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PAPER_DIR, 'fig_eigenvalue_spectrum.pdf'), bbox_inches='tight')
    plt.close()
    print("  Saved fig_eigenvalue_spectrum")


def figure5_summary_table(all_results):
    """
    Create a summary table as a figure for the paper.
    """
    experiments = ['part2_base', 'part2_nonlinear', 'part2_whitened']
    exp_labels = ['Base (Linear)', 'Nonlinear (ReLU)', 'Whitened']

    # Collect data
    data_rows = []
    for exp, label in zip(experiments, exp_labels):
        if exp not in all_results:
            continue
        results = all_results[exp]

        for r in ASPECT_RATIOS:
            res = results[r]
            all_volt, all_sgd = [], []

            for data in res['results_safe'].values():
                loss_every = data.get('loss_every', 1)
                t_sgd = np.arange(len(data['sgd_mean'])) * loss_every / N_SAMPLES
                volt_interp = np.interp(t_sgd, data['t_theory'], data['psi'])
                all_volt.extend(volt_interp)
                all_sgd.extend(data['sgd_mean'])

            volt = np.array(all_volt)
            sgd = np.array(all_sgd)
            valid = np.isfinite(volt) & np.isfinite(sgd) & (sgd > 0)
            volt, sgd = volt[valid], sgd[valid]

            pearson_r, _ = stats.pearsonr(volt, sgd)
            ss_res = np.sum((sgd - volt) ** 2)
            ss_tot = np.sum((sgd - np.mean(sgd)) ** 2)
            r2 = 1 - ss_res / ss_tot
            mape = np.mean(np.abs(volt - sgd) / sgd) * 100
            ratio = np.mean(volt / sgd)
            sr = res['spectral_info']['spectral_ratio']

            data_rows.append([label, f'{r}', f'{sr:.0f}', f'{pearson_r:.3f}',
                            f'{r2:.3f}', f'{mape:.1f}%', f'{ratio:.2f}'])

    # Create table figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    columns = ['Experiment', '$r$', 'SR', 'Pearson $R$', '$R^2$', 'MAPE', 'Ratio']

    table = ax.table(cellText=data_rows, colLabels=columns,
                     loc='center', cellLoc='center',
                     colColours=['#f0f0f0']*7)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_text_props(fontweight='bold')

    plt.title('Summary of Volterra Theory Prediction Quality', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(os.path.join(PAPER_DIR, 'fig_summary_table.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PAPER_DIR, 'fig_summary_table.pdf'), bbox_inches='tight')
    plt.close()
    print("  Saved fig_summary_table")


def main():
    print("="*70)
    print("GENERATING PUBLICATION-QUALITY FIGURES")
    print("="*70)

    # Load results
    print("\nLoading results...")
    all_results = load_results()

    if not all_results:
        print("ERROR: No results found!")
        return

    # Generate figures
    print("\nGenerating figures...")

    # Figure 1: Loss curves for each experiment
    if 'part2_base' in all_results:
        figure1_loss_curves(all_results['part2_base'], 'part2_base',
                          'SGD vs Volterra Theory: Base Experiment (Linear Target)',
                          'fig1_base_loss_curves')

    if 'part2_nonlinear' in all_results:
        figure1_loss_curves(all_results['part2_nonlinear'], 'part2_nonlinear',
                          'SGD vs Volterra Theory: Nonlinear Experiment (ReLU Target)',
                          'fig1_nonlinear_loss_curves')

    if 'part2_whitened' in all_results:
        figure1_loss_curves(all_results['part2_whitened'], 'part2_whitened',
                          'SGD vs Volterra Theory: Whitened Experiment',
                          'fig1_whitened_loss_curves')

    # Figure 2: Scatter plots for each experiment
    if 'part2_base' in all_results:
        figure2_scatter_plots(all_results['part2_base'], 'part2_base',
                            'Volterra Prediction Accuracy: Base Experiment',
                            'fig2_base_scatter')

    if 'part2_nonlinear' in all_results:
        figure2_scatter_plots(all_results['part2_nonlinear'], 'part2_nonlinear',
                            'Volterra Prediction Accuracy: Nonlinear Experiment',
                            'fig2_nonlinear_scatter')

    if 'part2_whitened' in all_results:
        figure2_scatter_plots(all_results['part2_whitened'], 'part2_whitened',
                            'Volterra Prediction Accuracy: Whitened Experiment',
                            'fig2_whitened_scatter')

    # Figure 3: Metrics comparison across all experiments
    figure3_metrics_comparison(all_results)

    # Figure 4: Eigenvalue spectrum
    figure4_eigenvalue_spectrum(all_results)

    # Figure 5: Summary table
    figure5_summary_table(all_results)

    print(f"\nAll figures saved to: {PAPER_DIR}/")
    print("="*70)


if __name__ == '__main__':
    main()
