"""
Part 2: Non-Linear Target Experiment (ReLU on labels)

Key difference: b is built directly from MNIST labels instead of a planted
target. We center the digit labels and apply ReLU:

    b = ReLU(y - mean(y))

This keeps the non-realizable, half-zero structure from the original setup
while ensuring the regression task still uses the real MNIST targets.

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


def prepare_nonlinear_label_targets(y: torch.Tensor, device: torch.device):
    """
    Build non-linear regression targets from MNIST digits.

    We center the labels and apply ReLU so roughly half of the entries are
    clamped to zero, mirroring the non-realizable ReLU target from the
    original design while still using real labels.
    """
    centered = y.to(device=device, dtype=torch.float32)
    centered = centered - centered.mean()
    b = torch.nn.functional.relu(centered)
    zero_fraction = (b == 0).float().mean().item()

    print(f"  Zero fraction (ReLU clipped): {zero_fraction:.2%}")

    return b, zero_fraction


def estimate_target_stats(A: torch.Tensor, b: torch.Tensor, r: float, eigvals: np.ndarray) -> tuple:
    """Estimate R and noise variance for Volterra, calibrated to match SGD initial loss.

    psi(0) = (R/2)*h1(0) + R_tilde/2, where h1(0) = 1 for r<=1, or 1/r for r>1.
    Calibrate R so psi(0) = L(0) = (1/2n)||b||^2.
    """
    with torch.no_grad():
        n, d = A.shape
        initial_loss = (1.0 / (2.0 * n)) * torch.sum(b**2).item()

        h1_0 = np.mean(eigvals) if r <= 1.0 else 1.0 / r

        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        max_sv, min_sv = S[0].item(), S[-1].item()
        condition_number = max_sv / (min_sv + 1e-10)

        threshold = max_sv * (0.01 if condition_number > 1000 else 0.001 if condition_number > 100 else 1e-6)
        S_inv = torch.where(S > threshold, 1.0 / S, torch.zeros_like(S))

        solution = Vh.T @ (S_inv * (U.T @ b))
        noise_var = torch.mean((A @ solution - b)**2).item()

        R_val = (2.0 / h1_0) * (initial_loss - noise_var / 2.0)
        if R_val < 0:
            R_val = 0.0

        psi_0 = (R_val / 2.0) * h1_0 + noise_var / 2.0
        print(f"    h1(0)={h1_0:.4f}, R={R_val:.4f}, noise={noise_var:.6f}, psi(0)={psi_0:.4f}")

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

        # Volterra Theory with label-derived forcing parameters
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
            ax.plot(data['t_theory'], data['psi'], '--', color=colors[i], alpha=0.8)

        # Mark final loss (non-zero due to non-realizability)
        final_loss = res['results_safe']['0.90']['sgd_mean'][-1]
        ax.axhline(y=final_loss, color='red', linestyle=':', alpha=0.5,
                   label=f'Final loss={final_loss:.3f}')

        ax.set_xlabel('Epochs (t)', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'r = {r} (d = {int(N_SAMPLES*r)})\nZero fraction = {res["zero_fraction"]:.1%}',
                    fontsize=12)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')

    plt.suptitle('Non-Linear Target: Volterra vs SGD (Safe Learning Rates)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/loss_curves.png")


def plot_zero_fraction(all_results: dict, output_dir: str):
    """Generate zero fraction analysis figure."""
    print("Generating Figure 2: Zero Fraction Analysis...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, r in enumerate(ASPECT_RATIOS):
        ax = axes[idx]
        res = all_results[r]

        zero_frac = res['zero_fraction']
        nonzero_frac = 1 - zero_frac

        sizes = [zero_frac, nonzero_frac]
        labels = [f'Zero\n({zero_frac:.1%})', f'Non-zero\n({nonzero_frac:.1%})']
        colors_pie = ['#ff6b6b', '#4ecdc4']
        explode = (0.05, 0)

        ax.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
               autopct='', startangle=90, textprops={'fontsize': 11})
        ax.set_title(f'r = {r}\nTargets after ReLU', fontsize=12)

    plt.suptitle('Zero Fraction: ~50% of targets clipped by ReLU', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'zero_fraction.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/zero_fraction.png")


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

    plt.suptitle('Non-Linear Target: Volterra Prediction Accuracy', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'volterra_vs_sgd_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/volterra_vs_sgd_scatter.png")


def plot_final_loss_comparison(all_results: dict, output_dir: str):
    """Generate final loss comparison figure."""
    print("Generating Figure 4: Final Loss Comparison...")

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(ASPECT_RATIOS))
    width = 0.35

    final_losses = [all_results[r]['results_safe']['0.90']['sgd_mean'][-1] for r in ASPECT_RATIOS]
    zero_fractions = [all_results[r]['zero_fraction'] for r in ASPECT_RATIOS]

    bars = ax.bar(x, final_losses, width, label='Final Loss (non-realizable)', color='coral')

    for i, (bar, zf) in enumerate(zip(bars, zero_fractions)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'Zero: {zf:.1%}', ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Aspect Ratio', fontsize=12)
    ax.set_ylabel('Final Loss', fontsize=12)
    ax.set_title('Non-Linear Target: Irreducible Error from ReLU\n(Loss cannot reach zero)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'r={r}' for r in ASPECT_RATIOS])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_loss_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/final_loss_comparison.png")


def main():
    """Main entry point for the experiment."""
    device = get_device()

    # Output directories - separate plots and results
    plots_dir = './plots/part2_nonlinear'
    results_dir = './results/part2_nonlinear'
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("PART 2: NON-LINEAR TARGET EXPERIMENT")
    print("Key change: b = ReLU(y - mean(y)) using real labels")
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
    b_labels, zero_fraction_base = prepare_nonlinear_label_targets(y_mnist, device)
    print(f"X_mnist shape: {X_mnist.shape}")
    print(f"Labels shape: {y_mnist.shape} (centered then ReLUed)")

    # Main experiment loop
    print("\n" + "="*60)
    print("STARTING FULL NON-LINEAR TARGET EXPERIMENTS")
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

        R_val, noise_var = estimate_target_stats(A_scaled, b_labels, r, eigvals)

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
            'zero_fraction': zero_fraction_base,
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
    plot_zero_fraction(all_results, plots_dir)
    plot_volterra_scatter(all_results, plots_dir)
    plot_final_loss_comparison(all_results, plots_dir)

    # Summary
    print("\n" + "="*60)
    print("PART 2 NON-LINEAR TARGET EXPERIMENT COMPLETE")
    print("="*60)

    print(f"\nTotal runtime: {total_time/60:.2f} minutes")
    print(f"\nResults summary:")
    for r in ASPECT_RATIOS:
        res = all_results[r]
        final_loss = res['results_safe']['0.90']['sgd_mean'][-1]
        print(f"\n  r = {r}:")
        print(f"    Zero fraction: {res['zero_fraction']:.2%}")
        print(f"    Final loss: {final_loss:.4f} (non-zero due to ReLU)")
        print(f"    Spectral ratio: {res['spectral_info']['spectral_ratio']:.2f}")

    print(f"\nFigures saved to {plots_dir}/")


if __name__ == '__main__':
    main()
