"""
Unified Experiment Runner for Volterra SGD Analysis
====================================================
Runs all Part 2 and Part 3 experiments with integrated plotting.

Part 2 Experiments:
- Base: Linear target regression on MNIST random features
- Nonlinear: ReLU target regression
- Whitened: Whitened MNIST data

Part 3 Experiment:
- Step-size criticality analysis

All results are saved to ./results/ and plots to ./plots/
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from tqdm import tqdm

from src.data_loader import load_mnist, whiten_data
from src.random_features import generate_random_weights, compute_features
from src.spectral import compute_eigenvalues, marchenko_pastur_density
from src.sgd import LeastSquaresSGD
from src.volterra import VolterraSolver

# =============================================================================
# CONFIGURATION
# =============================================================================
# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Experiment parameters
N_SAMPLES = 3000
NUM_EPOCHS = 20
NUM_RUNS = 5
ASPECT_RATIOS = [0.5, 1.0, 1.5]

# Output directories
RESULTS_DIR = './results'
PLOTS_DIR = './plots'

# =============================================================================
# PLOT STYLE CONFIGURATION
# =============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'mathtext.fontset': 'cm',
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'axes.axisbelow': True,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'legend.fontsize': 10,
    'legend.framealpha': 0.95,
    'legend.edgecolor': '0.8',
    'lines.linewidth': 2.0,
    'lines.markersize': 6,
    'grid.alpha': 0.4,
    'grid.linewidth': 0.6,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

COLORS = {
    'volterra': '#2ca02c',
    'sgd': '#d62728',
    'streaming': '#1f77b4',
    'lr1': '#d62728',
    'lr2': '#1f77b4',
    'lr3': '#2ca02c',
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_device():
    """Setup and return the compute device."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def build_random_features(X: torch.Tensor, r: float, device: torch.device):
    """Builds random features A = sigma(XW) with per-column centering."""
    n_samples = X.shape[0]
    d_in = X.shape[1]
    d_out = int(n_samples * r)

    W = generate_random_weights(d_in, d_out, device=device)
    A = compute_features(X, W, activation='shifted_relu', scale_by_sqrt_d=True)
    A = A - A.mean(dim=0, keepdim=True)

    return A, W


def prepare_linear_targets(y: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Mean-centered MNIST digit labels as regression targets."""
    b = y.to(device=device, dtype=torch.float32)
    return b - b.mean()


def prepare_nonlinear_targets(y: torch.Tensor, device: torch.device):
    """ReLU applied to centered labels - creates non-realizable target."""
    centered = y.to(device=device, dtype=torch.float32)
    centered = centered - centered.mean()
    b = torch.nn.functional.relu(centered)
    zero_fraction = (b == 0).float().mean().item()
    return b, zero_fraction


def estimate_target_stats(A: torch.Tensor, b: torch.Tensor) -> tuple:
    """Estimate R and noise variance for Volterra."""
    with torch.no_grad():
        solution = torch.linalg.lstsq(A, b.unsqueeze(1)).solution.squeeze(1)
        residuals = A @ solution - b
        R_val = torch.sum(solution**2).item()
        noise_var = torch.mean(residuals**2).item()
    return R_val, noise_var


def analyze_and_scale_spectrum(A: torch.Tensor, r: float, n: int):
    """Compute eigenvalues, scale A, and compute gamma_max values."""
    eigvals_initial = compute_eigenvalues(A)
    mean_eig_initial = np.mean(eigvals_initial)

    scale_factor = 1.0 / np.sqrt(mean_eig_initial)
    A_scaled = A * scale_factor

    eigvals = compute_eigenvalues(A_scaled)
    mean_eig = np.mean(eigvals)
    max_eig = np.max(eigvals)
    min_eig = np.min(eigvals[eigvals > 1e-10])

    gamma_max_theory = (2.0 / r) / mean_eig
    gamma_max_safe = 2.0 / max_eig
    spectral_ratio = max_eig / mean_eig

    spectral_info = {
        'mean_eig': mean_eig,
        'max_eig': max_eig,
        'min_eig': min_eig,
        'spectral_ratio': spectral_ratio,
        'scale_factor': scale_factor
    }

    return A_scaled, eigvals, gamma_max_theory, gamma_max_safe, spectral_info


def run_sgd_sweep(A: torch.Tensor, b: torch.Tensor, eigvals: np.ndarray,
                  gamma_max: float, r: float, n: int, num_epochs: int,
                  num_runs: int, R_val: float, noise_var: float,
                  sweep_name: str = ""):
    """Run SGD + Volterra comparison for multiple learning rates."""
    steps = n * num_epochs
    multipliers = [0.25, 0.5, 0.9]

    results = {}

    for mult in multipliers:
        gamma = gamma_max * mult
        key = f"{mult:.2f}"

        # Volterra Theory
        solver = VolterraSolver(eigvals, gamma, r, R=R_val, R_tilde=noise_var)
        psi, t_theory = solver.solve(t_max=num_epochs, dt=0.05)

        # Empirical SGD with progress bar
        sgd_runs = []
        for run in tqdm(range(num_runs), desc=f"  {sweep_name} gamma={mult:.0%}", leave=False):
            model = LeastSquaresSGD(A, b, learning_rate=gamma/n, batch_size=1)
            loss_hist = model.train(steps, show_progress=True, desc=f"    Run {run+1}")
            sgd_runs.append(loss_hist)

        sgd_mean = np.mean(sgd_runs, axis=0)
        sgd_std = np.std(sgd_runs, axis=0)

        results[key] = {
            'gamma': gamma,
            'mult': mult,
            't_theory': t_theory,
            'psi': psi,
            'sgd_runs': sgd_runs,
            'sgd_mean': sgd_mean,
            'sgd_std': sgd_std
        }

    return results


def is_stable(loss_hist: np.ndarray, initial_loss: float) -> bool:
    """Check if SGD run was stable."""
    if not np.all(np.isfinite(loss_hist)):
        return False
    if loss_hist[-1] > 10 * initial_loss:
        return False
    return True


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================
def plot_loss_curves(all_results: dict, n_samples: int, output_dir: str,
                     title_prefix: str = "", extra_info_key: str = None):
    """Generate loss curves figure with Volterra vs SGD comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors_lr = [COLORS['lr1'], COLORS['lr2'], COLORS['lr3']]

    for idx, r in enumerate(ASPECT_RATIOS):
        ax = axes[idx]
        res = all_results[r]

        for i, (key, data) in enumerate(res['results_safe'].items()):
            t_sgd = np.arange(len(data['sgd_mean'])) / n_samples
            ax.plot(t_sgd, data['sgd_mean'], color=colors_lr[i], lw=2, alpha=0.9)
            ax.fill_between(t_sgd,
                           data['sgd_mean'] - data['sgd_std'],
                           data['sgd_mean'] + data['sgd_std'],
                           color=colors_lr[i], alpha=0.15)
            ax.plot(data['t_theory'], data['psi'], '--', color=colors_lr[i], lw=1.8)

        ax.set_xlabel('Epochs')
        ax.set_ylabel('Training Loss' if idx == 0 else '')
        ax.set_yscale('log')

        subtitle = f'r = {r}'
        if extra_info_key and extra_info_key in res:
            subtitle += f' ({extra_info_key}: {res[extra_info_key]:.1%})'
        elif 'spectral_info' in res:
            subtitle += f' (SR: {res["spectral_info"]["spectral_ratio"]:.0f})'
        ax.set_title(subtitle)
        ax.grid(True, alpha=0.3)

    # Add legend to last subplot
    legend_elements = [
        Line2D([0], [0], color='gray', lw=2, label='SGD (solid)'),
        Line2D([0], [0], color='gray', lw=2, linestyle='--', label='Volterra (dashed)')
    ]
    axes[2].legend(handles=legend_elements, loc='upper right', fontsize=9)

    fig.suptitle(f'{title_prefix} Volterra vs SGD', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'loss_curves.pdf'))
    plt.close()


def plot_correlation_scatter(all_results: dict, n_samples: int, output_dir: str,
                             title_prefix: str = ""):
    """Generate Volterra vs SGD scatter plot showing prediction accuracy."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    correlations = []

    for idx, r in enumerate(ASPECT_RATIOS):
        ax = axes[idx]
        res = all_results[r]

        all_volterra = []
        all_sgd = []

        for key, data in res['results_safe'].items():
            t_sgd = np.arange(len(data['sgd_mean'])) / n_samples
            volterra_interp = np.interp(t_sgd, data['t_theory'], data['psi'])
            step = max(1, len(t_sgd) // 100)
            all_volterra.extend(volterra_interp[::step])
            all_sgd.extend(data['sgd_mean'][::step])

        all_volterra = np.array(all_volterra)
        all_sgd = np.array(all_sgd)

        valid = (all_volterra > 0) & (all_sgd > 0) & np.isfinite(all_volterra) & np.isfinite(all_sgd)
        all_volterra = all_volterra[valid]
        all_sgd = all_sgd[valid]

        ax.scatter(all_volterra, all_sgd, alpha=0.5, s=15, c=COLORS['streaming'], edgecolors='none')

        min_val = min(all_volterra.min(), all_sgd.min())
        max_val = max(all_volterra.max(), all_sgd.max())
        ax.plot([min_val, max_val], [min_val, max_val], '--', color=COLORS['sgd'], lw=2, label='y = x')

        corr, _ = stats.pearsonr(all_volterra, all_sgd)
        correlations.append(corr)

        ax.set_xlabel('Volterra Predicted Loss')
        ax.set_ylabel('Actual SGD Loss' if idx == 0 else '')
        ax.set_title(f'r = {r} (R = {corr:.4f})')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{title_prefix} Volterra Prediction Accuracy', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_scatter.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'correlation_scatter.pdf'))
    plt.close()

    return correlations


def plot_eigenvalue_spectrum(all_eigvals: dict, output_dir: str, title_prefix: str = ""):
    """Generate eigenvalue spectrum figure."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, r in enumerate(ASPECT_RATIOS):
        ax = axes[idx]
        eigvals = all_eigvals[r]

        ax.hist(eigvals, bins=50, density=True, alpha=0.7, color='steelblue',
                edgecolor='white', label='Empirical')

        x_mp = np.linspace(0.001, np.max(eigvals)*1.1, 500)
        try:
            mp_density = marchenko_pastur_density(x_mp, r)
            ax.plot(x_mp, mp_density, 'r-', linewidth=2, label='Marchenko-Pastur')
        except Exception:
            pass

        mean_eig = np.mean(eigvals)
        max_eig = np.max(eigvals)
        ax.axvline(mean_eig, color='green', linestyle='--', linewidth=2, label=f'Mean = {mean_eig:.2f}')
        ax.axvline(max_eig, color='orange', linestyle='--', linewidth=2, label=f'Max = {max_eig:.1f}')

        ax.set_xlabel(r'Eigenvalue $\lambda$')
        ax.set_ylabel('Density' if idx == 0 else '')
        ax.set_title(f'r = {r} (SR = {max_eig/mean_eig:.0f})')
        ax.legend(fontsize=8)
        ax.set_xlim(0, min(max_eig*1.2, 50))
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{title_prefix} Eigenvalue Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eigenvalue_spectrum.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'eigenvalue_spectrum.pdf'))
    plt.close()


def plot_summary_metrics(all_results: dict, output_dir: str, title_prefix: str = ""):
    """Generate summary metrics figure."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Spectral Ratio
    ax = axes[0]
    spectral_ratios = [all_results[r]['spectral_info']['spectral_ratio'] for r in ASPECT_RATIOS]
    bars = ax.bar(range(len(ASPECT_RATIOS)), spectral_ratios, color='steelblue', edgecolor='navy')
    ax.set_xticks(range(len(ASPECT_RATIOS)))
    ax.set_xticklabels([f'r={r}' for r in ASPECT_RATIOS])
    ax.set_ylabel(r'Spectral Ratio $\lambda_{max}/\bar{\lambda}$')
    ax.set_title('Spectral Ratio')
    for bar, val in zip(bars, spectral_ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val:.0f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Theory vs Safe gamma
    ax = axes[1]
    theory_gamma = [all_results[r]['gamma_max_theory'] for r in ASPECT_RATIOS]
    safe_gamma = [all_results[r]['gamma_max_safe'] for r in ASPECT_RATIOS]
    x = np.arange(len(ASPECT_RATIOS))
    width = 0.35
    ax.bar(x - width/2, theory_gamma, width, label=r'$\gamma_{theory}$', color='coral')
    ax.bar(x + width/2, safe_gamma, width, label=r'$\gamma_{safe}$', color='seagreen')
    ax.set_xticks(x)
    ax.set_xticklabels([f'r={r}' for r in ASPECT_RATIOS])
    ax.set_ylabel('Step Size')
    ax.set_title('Theory vs Safe Step Size')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Gap ratio
    ax = axes[2]
    gap = [t/s for t, s in zip(theory_gamma, safe_gamma)]
    bars = ax.bar(range(len(ASPECT_RATIOS)), gap, color='purple', edgecolor='darkviolet')
    ax.set_xticks(range(len(ASPECT_RATIOS)))
    ax.set_xticklabels([f'r={r}' for r in ASPECT_RATIOS])
    ax.set_ylabel(r'Gap Ratio $\gamma_{theory}/\gamma_{safe}$')
    ax.set_title('Theory-Practice Gap')
    for bar, val in zip(bars, gap):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.0f}x',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'{title_prefix} Summary Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_metrics.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'summary_metrics.pdf'))
    plt.close()


def plot_criticality_analysis(all_results: dict, output_dir: str):
    """Generate criticality analysis plots."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, r in enumerate(ASPECT_RATIOS):
        ax = axes[idx]
        res = all_results[r]
        data = res['sweep_results']

        multipliers = [d['multiplier'] for d in data]
        rel_losses = [min(d['relative_loss'], 100.0) for d in data]

        gamma_safe = res['spectral_info']['gamma_safe']
        gamma_theory = res['spectral_info']['gamma_theory']

        ax.plot(multipliers, rel_losses, 'o-', color=COLORS['sgd'], lw=2, ms=7, label='Final/Initial Loss')
        ax.axvline(1.0, color=COLORS['streaming'], linestyle='--', lw=2, label=r'$\gamma_{safe}$')

        ratio_theory = gamma_theory / gamma_safe
        ax.axvline(ratio_theory, color=COLORS['volterra'], linestyle='--', lw=2, label=r'$\gamma_{theory}$')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\gamma / \gamma_{safe}$')
        ax.set_ylabel('Final Loss / Initial Loss' if idx == 0 else '')

        lambda_ratio = res['spectral_info']['lambda_max'] / res['spectral_info']['mean_lambda']
        ax.set_title(f'r = {r} (SR = {lambda_ratio:.0f})')
        ax.grid(True, alpha=0.3)

        if idx == 2:
            ax.legend(loc='lower left', fontsize=9)

    fig.suptitle('Step-Size Criticality Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'criticality_analysis.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'criticality_analysis.pdf'))
    plt.close()


def plot_criticality_heatmap(all_results: dict, output_dir: str):
    """Generate stability heatmap."""
    fig, ax = plt.subplots(figsize=(12, 6))

    r_values = ASPECT_RATIOS
    k_values = range(-6, 2)
    multipliers = [2**k for k in k_values]

    stability_matrix = np.zeros((len(r_values), len(multipliers)))

    for i, r in enumerate(r_values):
        for j, mult in enumerate(multipliers):
            sweep = all_results[r]['sweep_results']
            matching = [s for s in sweep if abs(s['multiplier'] - mult) < 1e-6]
            if matching:
                stability_matrix[i, j] = matching[0]['stable_runs'] / matching[0]['total_runs']

    im = ax.imshow(stability_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(multipliers)))
    ax.set_xticklabels([f'{m:.3f}' for m in multipliers], rotation=45)
    ax.set_yticks(range(len(r_values)))
    ax.set_yticklabels([f'r={r}' for r in r_values])

    ax.set_xlabel(r'Multiplier $\gamma / \gamma_{safe}$')
    ax.set_ylabel('Aspect Ratio')
    ax.set_title('Stability Fraction (Green=Stable, Red=Diverged)')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Fraction Stable')

    for i in range(len(r_values)):
        for j in range(len(multipliers)):
            ax.text(j, i, f'{stability_matrix[i, j]:.1f}',
                   ha="center", va="center", color="black", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stability_heatmap.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'stability_heatmap.pdf'))
    plt.close()


# =============================================================================
# EXPERIMENT RUNNERS
# =============================================================================
def run_part2_base(X_mnist: torch.Tensor, y_mnist: torch.Tensor, device: torch.device):
    """Run Part 2 Base Experiment."""
    print("\n" + "="*70)
    print("PART 2: BASE EXPERIMENT (Linear Target)")
    print("="*70)

    plots_dir = os.path.join(PLOTS_DIR, 'part2_base')
    results_dir = os.path.join(RESULTS_DIR, 'part2_base')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    b_labels = prepare_linear_targets(y_mnist, device)

    all_results = {}
    all_eigvals = {}

    for r in tqdm(ASPECT_RATIOS, desc="Part 2 Base"):
        print(f"\n  Processing r = {r}...")

        A, W = build_random_features(X_mnist, r, device)
        A_scaled, eigvals, gamma_max_theory, gamma_max_safe, spectral_info = \
            analyze_and_scale_spectrum(A, r, N_SAMPLES)

        all_eigvals[r] = eigvals

        R_val, noise_var = estimate_target_stats(A_scaled, b_labels)

        results_safe = run_sgd_sweep(
            A_scaled, b_labels, eigvals,
            gamma_max_safe, r, N_SAMPLES, NUM_EPOCHS, NUM_RUNS,
            R_val, noise_var, sweep_name="Safe"
        )

        all_results[r] = {
            'spectral_info': spectral_info,
            'gamma_max_theory': gamma_max_theory,
            'gamma_max_safe': gamma_max_safe,
            'label_R': R_val,
            'label_noise': noise_var,
            'eigvals': eigvals,
            'results_safe': results_safe
        }

    # Save results
    np.save(os.path.join(results_dir, 'all_results.npy'), all_results)
    print(f"\n  Results saved to {results_dir}/all_results.npy")

    # Generate plots
    print("  Generating plots...")
    plot_loss_curves(all_results, N_SAMPLES, plots_dir, "Part 2 Base:")
    correlations = plot_correlation_scatter(all_results, N_SAMPLES, plots_dir, "Part 2 Base:")
    plot_eigenvalue_spectrum(all_eigvals, plots_dir, "Part 2 Base:")
    plot_summary_metrics(all_results, plots_dir, "Part 2 Base:")

    print(f"  Plots saved to {plots_dir}/")
    print(f"  Correlations: {[f'{c:.4f}' for c in correlations]}")

    return all_results, all_eigvals


def run_part2_nonlinear(X_mnist: torch.Tensor, y_mnist: torch.Tensor, device: torch.device):
    """Run Part 2 Nonlinear Experiment."""
    print("\n" + "="*70)
    print("PART 2: NONLINEAR EXPERIMENT (ReLU Target)")
    print("="*70)

    plots_dir = os.path.join(PLOTS_DIR, 'part2_nonlinear')
    results_dir = os.path.join(RESULTS_DIR, 'part2_nonlinear')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    b_labels, zero_fraction = prepare_nonlinear_targets(y_mnist, device)
    print(f"  Zero fraction (ReLU clipped): {zero_fraction:.2%}")

    all_results = {}
    all_eigvals = {}

    for r in tqdm(ASPECT_RATIOS, desc="Part 2 Nonlinear"):
        print(f"\n  Processing r = {r}...")

        A, W = build_random_features(X_mnist, r, device)
        A_scaled, eigvals, gamma_max_theory, gamma_max_safe, spectral_info = \
            analyze_and_scale_spectrum(A, r, N_SAMPLES)

        all_eigvals[r] = eigvals

        R_val, noise_var = estimate_target_stats(A_scaled, b_labels)

        results_safe = run_sgd_sweep(
            A_scaled, b_labels, eigvals,
            gamma_max_safe, r, N_SAMPLES, NUM_EPOCHS, NUM_RUNS,
            R_val, noise_var, sweep_name="Safe"
        )

        all_results[r] = {
            'spectral_info': spectral_info,
            'gamma_max_theory': gamma_max_theory,
            'gamma_max_safe': gamma_max_safe,
            'zero_fraction': zero_fraction,
            'label_R': R_val,
            'label_noise': noise_var,
            'eigvals': eigvals,
            'results_safe': results_safe
        }

    # Save results
    np.save(os.path.join(results_dir, 'all_results.npy'), all_results)
    print(f"\n  Results saved to {results_dir}/all_results.npy")

    # Generate plots
    print("  Generating plots...")
    plot_loss_curves(all_results, N_SAMPLES, plots_dir, "Part 2 Nonlinear:", "zero_fraction")
    correlations = plot_correlation_scatter(all_results, N_SAMPLES, plots_dir, "Part 2 Nonlinear:")
    plot_eigenvalue_spectrum(all_eigvals, plots_dir, "Part 2 Nonlinear:")
    plot_summary_metrics(all_results, plots_dir, "Part 2 Nonlinear:")

    print(f"  Plots saved to {plots_dir}/")
    print(f"  Correlations: {[f'{c:.4f}' for c in correlations]}")

    return all_results, all_eigvals


def run_part2_whitened(X_mnist: torch.Tensor, y_mnist: torch.Tensor, device: torch.device):
    """Run Part 2 Whitened Experiment."""
    print("\n" + "="*70)
    print("PART 2: WHITENED EXPERIMENT")
    print("="*70)

    plots_dir = os.path.join(PLOTS_DIR, 'part2_whitened')
    results_dir = os.path.join(RESULTS_DIR, 'part2_whitened')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Whiten data
    print("  Whitening MNIST data...")
    X_whitened, whiten_info = whiten_data(X_mnist)
    print(f"  Condition number: {whiten_info['condition_number']:.2f}")

    b_labels = prepare_linear_targets(y_mnist, device)

    all_results = {}
    all_eigvals = {}

    for r in tqdm(ASPECT_RATIOS, desc="Part 2 Whitened"):
        print(f"\n  Processing r = {r}...")

        A, W = build_random_features(X_whitened, r, device)
        A_scaled, eigvals, gamma_max_theory, gamma_max_safe, spectral_info = \
            analyze_and_scale_spectrum(A, r, N_SAMPLES)

        all_eigvals[r] = eigvals

        R_val, noise_var = estimate_target_stats(A_scaled, b_labels)

        results_safe = run_sgd_sweep(
            A_scaled, b_labels, eigvals,
            gamma_max_safe, r, N_SAMPLES, NUM_EPOCHS, NUM_RUNS,
            R_val, noise_var, sweep_name="Safe"
        )

        all_results[r] = {
            'spectral_info': spectral_info,
            'gamma_max_theory': gamma_max_theory,
            'gamma_max_safe': gamma_max_safe,
            'whiten_condition_number': whiten_info['condition_number'],
            'label_R': R_val,
            'label_noise': noise_var,
            'eigvals': eigvals,
            'results_safe': results_safe
        }

    # Save results
    np.save(os.path.join(results_dir, 'all_results.npy'), all_results)
    print(f"\n  Results saved to {results_dir}/all_results.npy")

    # Generate plots
    print("  Generating plots...")
    plot_loss_curves(all_results, N_SAMPLES, plots_dir, "Part 2 Whitened:")
    correlations = plot_correlation_scatter(all_results, N_SAMPLES, plots_dir, "Part 2 Whitened:")
    plot_eigenvalue_spectrum(all_eigvals, plots_dir, "Part 2 Whitened:")
    plot_summary_metrics(all_results, plots_dir, "Part 2 Whitened:")

    print(f"  Plots saved to {plots_dir}/")
    print(f"  Correlations: {[f'{c:.4f}' for c in correlations]}")

    return all_results, all_eigvals


def run_part3_criticality(X_mnist: torch.Tensor, y_mnist: torch.Tensor, device: torch.device):
    """Run Part 3 Criticality Experiment."""
    print("\n" + "="*70)
    print("PART 3: STEP-SIZE CRITICALITY ANALYSIS")
    print("="*70)

    plots_dir = os.path.join(PLOTS_DIR, 'part3_criticality')
    results_dir = os.path.join(RESULTS_DIR, 'part3_criticality')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    b_labels = prepare_linear_targets(y_mnist, device)

    all_results = {}

    for r in tqdm(ASPECT_RATIOS, desc="Part 3 Criticality"):
        print(f"\n  Processing r = {r}...")
        d = int(N_SAMPLES * r)

        # Build features
        W = generate_random_weights(X_mnist.shape[1], d, device=device)
        A = compute_features(X_mnist, W, activation='shifted_relu', scale_by_sqrt_d=True)
        A = A - A.mean(dim=0, keepdim=True)

        # Eigenvalue analysis
        eigvals = compute_eigenvalues(A)
        mean_eig = np.mean(eigvals)
        scale_factor = 1.0 / np.sqrt(mean_eig)
        A_scaled = A * scale_factor

        eigvals_scaled = compute_eigenvalues(A_scaled)
        mean_eig_scaled = np.mean(eigvals_scaled)
        max_eig_scaled = np.max(eigvals_scaled)

        gamma_safe = 2.0 / max_eig_scaled
        gamma_theory = (2.0 / r) / mean_eig_scaled

        initial_loss = (1.0 / (2 * N_SAMPLES)) * torch.sum(b_labels**2).item()

        # Learning rate sweep
        k_values = range(-6, 2)
        multipliers = [2**k for k in k_values]

        sweep_results = []

        for mult in multipliers:
            gamma = gamma_safe * mult
            lr = gamma / N_SAMPLES

            run_statuses = []
            final_losses = []
            steps = N_SAMPLES * NUM_EPOCHS

            for run in tqdm(range(NUM_RUNS), desc=f"    mult={mult:.3f}", leave=False):
                model = LeastSquaresSGD(A_scaled, b_labels, learning_rate=lr, batch_size=1)
                loss_hist = model.train(steps, show_progress=True, desc=f"      Run {run+1}")

                stable = is_stable(loss_hist, initial_loss)
                final_loss = loss_hist[-1] if np.isfinite(loss_hist[-1]) else float('inf')

                run_statuses.append(stable)
                final_losses.append(final_loss)

            num_stable = sum(run_statuses)
            avg_final_loss = np.mean([fl for fl in final_losses if fl != float('inf')]) \
                if any(np.isfinite(final_losses)) else float('inf')

            sweep_results.append({
                'gamma': gamma,
                'multiplier': mult,
                'stable_runs': num_stable,
                'total_runs': NUM_RUNS,
                'avg_final_loss': avg_final_loss,
                'relative_loss': avg_final_loss / initial_loss if avg_final_loss != float('inf') else float('inf')
            })

        all_results[r] = {
            'r': r,
            'spectral_info': {
                'lambda_max': max_eig_scaled,
                'mean_lambda': mean_eig_scaled,
                'gamma_safe': gamma_safe,
                'gamma_theory': gamma_theory
            },
            'initial_loss': initial_loss,
            'sweep_results': sweep_results
        }

    # Save results
    np.save(os.path.join(results_dir, 'all_results.npy'), all_results)
    print(f"\n  Results saved to {results_dir}/all_results.npy")

    # Generate plots
    print("  Generating plots...")
    plot_criticality_analysis(all_results, plots_dir)
    plot_criticality_heatmap(all_results, plots_dir)

    print(f"  Plots saved to {plots_dir}/")

    return all_results


def generate_combined_summary(base_results, nonlinear_results, whitened_results, crit_results):
    """Generate combined summary plots across all experiments."""
    print("\n" + "="*70)
    print("GENERATING COMBINED SUMMARY PLOTS")
    print("="*70)

    summary_dir = os.path.join(PLOTS_DIR, 'summary')
    os.makedirs(summary_dir, exist_ok=True)

    # Combined spectral ratio comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(ASPECT_RATIOS))
    width = 0.25

    sr_base = [base_results[r]['spectral_info']['spectral_ratio'] for r in ASPECT_RATIOS]
    sr_nonlinear = [nonlinear_results[r]['spectral_info']['spectral_ratio'] for r in ASPECT_RATIOS]
    sr_whitened = [whitened_results[r]['spectral_info']['spectral_ratio'] for r in ASPECT_RATIOS]

    ax.bar(x - width, sr_base, width, label='Base', color=COLORS['sgd'], edgecolor='black', lw=0.8)
    ax.bar(x, sr_nonlinear, width, label='Nonlinear', color=COLORS['streaming'], edgecolor='black', lw=0.8)
    ax.bar(x + width, sr_whitened, width, label='Whitened', color=COLORS['volterra'], edgecolor='black', lw=0.8)

    ax.set_xlabel('Aspect Ratio r')
    ax.set_ylabel(r'Spectral Ratio $\lambda_{max}/\bar{\lambda}$')
    ax.set_title('Spectral Ratio Comparison Across Experiments')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{r}' for r in ASPECT_RATIOS])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for bars, vals in [(x - width, sr_base), (x, sr_nonlinear), (x + width, sr_whitened)]:
        for bar_x, val in zip(bars, vals):
            ax.text(bar_x, val + 3, f'{val:.0f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'spectral_ratio_comparison.png'), dpi=300)
    plt.savefig(os.path.join(summary_dir, 'spectral_ratio_comparison.pdf'))
    plt.close()

    print(f"  Summary plots saved to {summary_dir}/")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main():
    """Main entry point - runs all experiments sequentially."""
    total_start = time.time()

    print("="*70)
    print("VOLTERRA SGD - UNIFIED EXPERIMENT RUNNER")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  N_SAMPLES: {N_SAMPLES}")
    print(f"  NUM_EPOCHS: {NUM_EPOCHS}")
    print(f"  NUM_RUNS: {NUM_RUNS}")
    print(f"  ASPECT_RATIOS: {ASPECT_RATIOS}")

    # Setup
    device = get_device()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Load MNIST
    print("\nLoading MNIST dataset...")
    X_mnist, y_mnist = load_mnist(root='./data', train=True, flatten=True,
                                   subset_size=N_SAMPLES, download=True)
    X_mnist = X_mnist.to(device)
    y_mnist = y_mnist.to(device)
    print(f"  X shape: {X_mnist.shape}")
    print(f"  y shape: {y_mnist.shape}")

    # Run experiments
    base_results, base_eigvals = run_part2_base(X_mnist, y_mnist, device)
    nonlinear_results, nonlinear_eigvals = run_part2_nonlinear(X_mnist, y_mnist, device)
    whitened_results, whitened_eigvals = run_part2_whitened(X_mnist, y_mnist, device)
    crit_results = run_part3_criticality(X_mnist, y_mnist, device)

    # Generate combined summary
    generate_combined_summary(base_results, nonlinear_results, whitened_results, crit_results)

    # Final summary
    total_time = time.time() - total_start
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*70)
    print(f"\nTotal runtime: {total_time/60:.2f} minutes")
    print(f"\nResults saved to: {RESULTS_DIR}/")
    print(f"Plots saved to: {PLOTS_DIR}/")


if __name__ == '__main__':
    main()
