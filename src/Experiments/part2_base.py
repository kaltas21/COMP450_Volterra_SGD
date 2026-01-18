"""
Part 2: Base Experiment - Volterra Theory on MNIST Random Features

Tests Volterra theory predictions against empirical SGD on random features
derived from MNIST data when regressing directly on the digit labels (centered).

Parameters: n=3000, epochs=20, runs=5, aspect_ratios=[0.5, 1.0, 1.5]
"""

import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from tqdm import tqdm
from src.data_loader import load_mnist
from src.random_features import generate_random_weights, compute_features
from src.spectral import compute_eigenvalues, marchenko_pastur_density
from src.sgd import LeastSquaresSGD
from src.volterra import VolterraSolver

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Full run parameters
N_SAMPLES = 3000
NUM_EPOCHS = 20
NUM_RUNS = 5
ASPECT_RATIOS = [0.5, 1.0, 1.5]


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

    print(f"\n=== Building Random Features (r={r}) ===")
    print(f"Input: X in R^({n_samples} x {d_in})")
    print(f"Output: A in R^({n_samples} x {d_out})")

    W = generate_random_weights(d_in, d_out, device=device)
    A = compute_features(X, W, activation='shifted_relu', scale_by_sqrt_d=True)
    A = A - A.mean(dim=0, keepdim=True)

    return A, W


def prepare_label_targets(y: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Use MNIST digit labels as regression targets.

    Labels are centered to remove the intercept term, which keeps the loss scale
    comparable to the planted-target setup and makes the Volterra forcing term
    depend on the feature covariance instead of a constant offset.
    """
    b = y.to(device=device, dtype=torch.float32)
    return b - b.mean()


def estimate_target_stats(A: torch.Tensor, b: torch.Tensor) -> tuple:
    """Estimate R and noise variance for Volterra using truncated SVD.

    Uses SVD with adaptive regularization based on condition number.
    This properly handles ill-conditioned matrices (r ≈ 1.0) where
    standard least-squares fails.
    """
    with torch.no_grad():
        n, d = A.shape

        # Use SVD for stable computation
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)

        # Condition number and adaptive threshold
        max_sv = S[0].item()
        min_sv = S[-1].item()
        condition_number = max_sv / (min_sv + 1e-10)

        # Adaptive regularization: stronger for ill-conditioned matrices
        if condition_number > 1000:
            threshold = max_sv * 0.01
        elif condition_number > 100:
            threshold = max_sv * 0.001
        else:
            threshold = max_sv * 1e-6

        # Truncated pseudo-inverse: zero out small singular values
        S_inv = torch.zeros_like(S)
        mask = S > threshold
        S_inv[mask] = 1.0 / S[mask]

        # Compute minimum-norm solution: x = V @ S_inv @ U^T @ b
        Utb = U.T @ b
        solution = Vh.T @ (S_inv * Utb)

        residuals = A @ solution - b
        R_val = torch.sum(solution**2).item()
        noise_var = torch.mean(residuals**2).item()

        # Additional sanity check based on data scale
        b_var = torch.var(b).item()
        max_reasonable_R = 100 * b_var

        if R_val > max_reasonable_R:
            print(f"  Warning: R_val={R_val:.2e} exceeds {max_reasonable_R:.2e}, clamping")
            R_val = max_reasonable_R

        print(f"    Condition number: {condition_number:.1f}, threshold: {threshold:.2e}, R: {R_val:.4f}, noise: {noise_var:.6f}")

    return R_val, noise_var


def analyze_and_scale_spectrum(A: torch.Tensor, r: float, n: int):
    """Compute eigenvalues, scale A, and compute gamma_max values."""
    print(f"Computing eigenvalues...")
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

    print(f"  mean(lambda) = {mean_eig:.4f}, max(lambda) = {max_eig:.4f}")
    print(f"  Spectral ratio = {spectral_ratio:.2f}")
    print(f"  gamma_theory = {gamma_max_theory:.4f}, gamma_safe = {gamma_max_safe:.4f}")

    spectral_info = {
        'mean_eig': mean_eig,
        'max_eig': max_eig,
        'min_eig': min_eig,
        'spectral_ratio': spectral_ratio,
        'scale_factor': scale_factor
    }

    return A_scaled, eigvals, gamma_max_theory, gamma_max_safe, spectral_info


def run_experiment_sweep(A: torch.Tensor, b: torch.Tensor, eigvals: np.ndarray,
                         gamma_max: float, r: float, n: int, num_epochs: int,
                         num_runs: int, R_val: float, noise_var: float,
                         sweep_name: str = ""):
    """Run SGD + Volterra comparison for multiple learning rates."""
    steps = n * num_epochs
    multipliers = [0.25, 0.5, 0.9]

    print(f"\n=== {sweep_name} Sweep (gamma_max = {gamma_max:.4f}) ===")

    results = {}

    for mult in multipliers:
        gamma = gamma_max * mult
        key = f"{mult:.2f}"
        print(f"  gamma = {gamma:.4f} ({mult*100:.0f}% of gamma_max)...")

        # Volterra Theory driven by label-induced R and residual variance
        solver = VolterraSolver(eigvals, gamma, r, R=R_val, R_tilde=noise_var)
        psi, t_theory = solver.solve(t_max=num_epochs, dt=0.05)

        # Empirical SGD with progress bar
        sgd_runs = []
        for run in tqdm(range(num_runs), desc=f"    Runs (gamma={mult:.0%})", leave=False):
            model = LeastSquaresSGD(A, b, learning_rate=gamma/n, batch_size=1)
            loss_hist = model.train(steps, show_progress=True, desc=f"      SGD Run {run+1}")
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


def plot_loss_curves(all_results: dict, output_dir: str):
    """Generate loss curves figure."""
    print("\nGenerating Figure 1: Loss Curves...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, r in enumerate(ASPECT_RATIOS):
        ax = axes[idx]
        res = all_results[r]

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        for i, (key, data) in enumerate(res['results_safe'].items()):
            t_sgd = np.arange(len(data['sgd_mean'])) / N_SAMPLES

            ax.plot(t_sgd, data['sgd_mean'], color=colors[i], alpha=0.8,
                    label=f"SGD gamma={data['gamma']:.4f}")
            ax.fill_between(t_sgd,
                           data['sgd_mean'] - data['sgd_std'],
                           data['sgd_mean'] + data['sgd_std'],
                           color=colors[i], alpha=0.2)
            ax.plot(data['t_theory'], data['psi'], '--', color=colors[i], alpha=0.8,
                    label=f"Volterra")

        ax.set_xlabel('Epochs (t)', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'r = {r} (d = {int(N_SAMPLES*r)})\nSpectral ratio = {res["spectral_info"]["spectral_ratio"]:.1f}',
                    fontsize=12)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')

    plt.suptitle('Volterra Prediction vs SGD (Safe Learning Rates)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/loss_curves.png")


def plot_eigenvalue_spectrum(all_eigvals: dict, output_dir: str):
    """Generate eigenvalue spectrum figure."""
    print("Generating Figure 2: Eigenvalue Spectrum...")

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

        ax.set_xlabel('Eigenvalue lambda', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'r = {r}\nSpectral ratio = {max_eig/mean_eig:.1f}', fontsize=12)
        ax.legend(fontsize=9)
        ax.set_xlim(0, min(max_eig*1.2, 50))
        ax.grid(True, alpha=0.3)

    plt.suptitle('Eigenvalue Distribution of H = (1/n)A^TA', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eigenvalue_spectrum.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/eigenvalue_spectrum.png")


def plot_volterra_scatter(all_results: dict, output_dir: str):
    """Generate Volterra vs SGD scatter plot."""
    print("Generating Figure 3: Volterra vs SGD Scatter...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, r in enumerate(ASPECT_RATIOS):
        ax = axes[idx]
        res = all_results[r]

        all_volterra = []
        all_sgd = []

        for key, data in res['results_safe'].items():
            t_sgd = np.arange(len(data['sgd_mean'])) / N_SAMPLES
            volterra_interp = np.interp(t_sgd, data['t_theory'], data['psi'])

            step = max(1, len(t_sgd) // 100)
            all_volterra.extend(volterra_interp[::step])
            all_sgd.extend(data['sgd_mean'][::step])

        all_volterra = np.array(all_volterra)
        all_sgd = np.array(all_sgd)

        valid = (all_volterra > 0) & (all_sgd > 0) & np.isfinite(all_volterra) & np.isfinite(all_sgd)
        all_volterra = all_volterra[valid]
        all_sgd = all_sgd[valid]

        ax.scatter(all_volterra, all_sgd, alpha=0.5, s=10, c='steelblue')

        min_val = min(all_volterra.min(), all_sgd.min())
        max_val = max(all_volterra.max(), all_sgd.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')

        corr, _ = stats.pearsonr(all_volterra, all_sgd)

        ax.set_xlabel('Volterra Predicted Loss', fontsize=12)
        ax.set_ylabel('Actual SGD Loss', fontsize=12)
        ax.set_title(f'r = {r}\nCorrelation R = {corr:.4f}', fontsize=12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Volterra Prediction Accuracy', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'volterra_vs_sgd_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/volterra_vs_sgd_scatter.png")


def plot_summary_metrics(all_results: dict, output_dir: str):
    """Generate summary metrics figure."""
    print("Generating Figure 4: Summary Metrics...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Spectral Ratio by r
    ax = axes[0]
    spectral_ratios = [all_results[r]['spectral_info']['spectral_ratio'] for r in ASPECT_RATIOS]
    bars = ax.bar(range(len(ASPECT_RATIOS)), spectral_ratios, color='steelblue', edgecolor='navy')
    ax.set_xticks(range(len(ASPECT_RATIOS)))
    ax.set_xticklabels([f'r={r}' for r in ASPECT_RATIOS])
    ax.set_ylabel('Spectral Ratio (lambda_max / lambda_mean)', fontsize=12)
    ax.set_title('Spectral Ratio by Aspect Ratio', fontsize=12)
    for bar, val in zip(bars, spectral_ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Step Size Gap (theory vs safe)
    ax = axes[1]
    theory_gamma = [all_results[r]['gamma_max_theory'] for r in ASPECT_RATIOS]
    safe_gamma = [all_results[r]['gamma_max_safe'] for r in ASPECT_RATIOS]

    x = np.arange(len(ASPECT_RATIOS))
    width = 0.35
    ax.bar(x - width/2, theory_gamma, width, label='gamma_theory', color='coral')
    ax.bar(x + width/2, safe_gamma, width, label='gamma_safe', color='seagreen')
    ax.set_xticks(x)
    ax.set_xticklabels([f'r={r}' for r in ASPECT_RATIOS])
    ax.set_ylabel('Step Size (gamma)', fontsize=12)
    ax.set_title('Theory vs Safe Step Size', fontsize=12)
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Gap ratio
    ax = axes[2]
    gap = [t/s for t, s in zip(theory_gamma, safe_gamma)]
    bars = ax.bar(range(len(ASPECT_RATIOS)), gap, color='purple', edgecolor='darkviolet')
    ax.set_xticks(range(len(ASPECT_RATIOS)))
    ax.set_xticklabels([f'r={r}' for r in ASPECT_RATIOS])
    ax.set_ylabel('Gap Ratio (gamma_theory / gamma_safe)', fontsize=12)
    ax.set_title('Theory-Practice Gap', fontsize=12)
    for bar, val in zip(bars, gap):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val:.0f}x',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Summary Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/summary_metrics.png")


def main():
    """Main entry point for the experiment."""
    device = get_device()

    # Output directories - separate plots and results
    plots_dir = './plots/part2_base'
    results_dir = './results/part2_base'
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("PART 2: BASE EXPERIMENT")
    print(f"{'='*60}")
    print(f"Parameters: n={N_SAMPLES}, epochs={NUM_EPOCHS}, runs={NUM_RUNS}")
    print(f"Aspect ratios: {ASPECT_RATIOS}")
    print(f"Plots directory: {plots_dir}")
    print(f"Results directory: {results_dir}")

    # Load MNIST
    print("\nLoading MNIST...")
    X_mnist, y_mnist = load_mnist(root='./data', train=True, flatten=True,
                            subset_size=N_SAMPLES, download=True)
    X_mnist = X_mnist.to(device)
    y_mnist = y_mnist.to(device)
    b_labels = prepare_label_targets(y_mnist, device)
    print(f"X_mnist shape: {X_mnist.shape}")
    print(f"Labels shape: {y_mnist.shape} (mean-centered)")

    # Main experiment loop
    print("\n" + "="*60)
    print("STARTING FULL EXPERIMENTS")
    print("="*60)

    start_time = time.time()
    all_results = {}
    all_eigvals = {}

    for idx, r in enumerate(ASPECT_RATIOS):
        exp_start = time.time()
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {idx+1}/{len(ASPECT_RATIOS)}: r = {r}")
        print(f"{'='*60}")

        A, W = build_random_features(X_mnist, r, device)
        A_scaled, eigvals, gamma_max_theory, gamma_max_safe, spectral_info = \
            analyze_and_scale_spectrum(A, r, N_SAMPLES)

        all_eigvals[r] = eigvals

        initial_loss = (1.0 / (2 * N_SAMPLES)) * torch.sum(b_labels**2).item()
        R_val, noise_var = estimate_target_stats(A_scaled, b_labels)

        results_safe = run_experiment_sweep(
            A_scaled, b_labels, eigvals,
            gamma_max_safe, r, N_SAMPLES, NUM_EPOCHS, NUM_RUNS,
            R_val, noise_var,
            sweep_name="Safe"
        )

        results_theory = run_experiment_sweep(
            A_scaled, b_labels, eigvals,
            gamma_max_theory, r, N_SAMPLES, NUM_EPOCHS, NUM_RUNS,
            R_val, noise_var,
            sweep_name="Theory"
        )

        all_results[r] = {
            'spectral_info': spectral_info,
            'gamma_max_theory': gamma_max_theory,
            'gamma_max_safe': gamma_max_safe,
            'initial_loss': initial_loss,
            'label_R': R_val,
            'label_noise': noise_var,
            'eigvals': eigvals,
            'results_safe': results_safe,
            'results_theory': results_theory
        }

        exp_time = time.time() - exp_start
        print(f"\nCompleted r = {r} in {exp_time/60:.2f} minutes")

    total_time = time.time() - start_time

    # Save results
    np.save(os.path.join(results_dir, 'all_results.npy'), all_results)
    print(f"\nResults saved to {results_dir}/all_results.npy")

    # Generate figures
    plot_loss_curves(all_results, plots_dir)
    plot_eigenvalue_spectrum(all_eigvals, plots_dir)
    plot_volterra_scatter(all_results, plots_dir)
    plot_summary_metrics(all_results, plots_dir)

    # Summary
    print("\n" + "="*60)
    print("PART 2 BASE EXPERIMENT COMPLETE")
    print("="*60)

    print(f"\nTotal runtime: {total_time/60:.2f} minutes")
    print(f"\nResults summary:")
    for r in ASPECT_RATIOS:
        res = all_results[r]
        print(f"\n  r = {r}:")
        print(f"    Spectral ratio: {res['spectral_info']['spectral_ratio']:.2f}")
        print(f"    gamma_theory: {res['gamma_max_theory']:.4f}")
        print(f"    gamma_safe: {res['gamma_max_safe']:.4f}")
        print(f"    Gap: {res['gamma_max_theory']/res['gamma_max_safe']:.1f}x")

    print(f"\nFigures saved to {plots_dir}/")


if __name__ == '__main__':
    main()
