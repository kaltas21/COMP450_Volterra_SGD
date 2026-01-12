"""
Full Run: Non-Linear Target Experiment (ReLU on targets)
Key difference: b = ReLU(A @ x_star) instead of b = A @ x_star
Parameters: n=3000, epochs=20, runs=5, aspect_ratios=[0.5, 1.0, 1.5]
"""

import sys
import os
import time

sys.path.append(os.path.abspath('.'))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from src.data_loader import load_mnist
from src.random_features import generate_random_weights, compute_features
from src.spectral import compute_eigenvalues, marchenko_pastur_density
from src.sgd import LeastSquaresSGD
from src.volterra import VolterraSolver

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# =============================================================================
# FULL RUN PARAMETERS
# =============================================================================
n = 3000
num_epochs = 20
num_runs = 5
aspect_ratios = [0.5, 1.0, 1.5]

# Output directory
output_dir = './results/full_run_nonlinear'
os.makedirs(output_dir, exist_ok=True)

print(f"\n{'='*60}")
print("FULL RUN: NON-LINEAR TARGET EXPERIMENT")
print("Key change: b = ReLU(A @ x_star)")
print(f"{'='*60}")
print(f"Parameters: n={n}, epochs={num_epochs}, runs={num_runs}")
print(f"Aspect ratios: {aspect_ratios}")
print(f"Output directory: {output_dir}")

# Load MNIST
print("\nLoading MNIST...")
X_mnist, y_mnist = load_mnist(root='./data', train=True, flatten=True,
                               subset_size=n, download=True)
X_mnist = X_mnist.to(device)
print(f"X_mnist shape: {X_mnist.shape}")


def build_random_features(X, r, device):
    """Builds random features A = sigma(XW) with per-column centering."""
    n_samples = X.shape[0]
    d_in = X.shape[1]
    d_out = int(n_samples * r)

    print(f"\n=== Building Random Features (r={r}) ===")
    print(f"Input: X in R^({n_samples} x {d_in})")
    print(f"Output: A in R^({n_samples} x {d_out})")

    W = generate_random_weights(d_in, d_out).to(device)
    A = compute_features(X, W, activation='shifted_relu', scale_by_sqrt_d=True)
    A = A - A.mean(dim=0, keepdim=True)

    return A, W


def generate_nonlinear_planted_targets(A, device):
    """
    Generate NON-LINEAR planted targets: b = ReLU(A @ x_star)
    This creates a non-realizable problem.
    """
    n, d = A.shape
    x_star = torch.randn(d, device=device)
    x_star = x_star / torch.norm(x_star)

    # Linear target (before ReLU)
    linear_target = A @ x_star

    # NON-LINEAR target (after ReLU) - KEY DIFFERENCE
    b = torch.nn.functional.relu(linear_target)

    # Statistics
    zero_fraction = (b == 0).float().mean().item()

    x0 = torch.zeros(d, device=device)
    initial_loss = (1.0 / (2 * n)) * torch.sum(b**2).item()

    print(f"  Zero fraction (ReLU clipped): {zero_fraction:.2%}")
    print(f"  Initial loss f(x0) = {initial_loss:.4f}")

    return x_star, b, x0, initial_loss, zero_fraction


def analyze_and_scale_spectrum(A, r, n):
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


def run_experiment_sweep(A, b, x_star, eigvals, gamma_max, r, n, num_epochs, num_runs, sweep_name=""):
    """Run SGD + Volterra comparison for multiple learning rates."""
    d = A.shape[1]
    steps = n * num_epochs
    multipliers = [0.25, 0.5, 0.9]

    print(f"\n=== {sweep_name} Sweep (gamma_max = {gamma_max:.4f}) ===")

    results = {}

    for mult in multipliers:
        gamma = gamma_max * mult
        key = f"{mult:.2f}"
        print(f"  gamma = {gamma:.4f} ({mult*100:.0f}% of gamma_max)...")

        # Volterra Theory (R=1 as approximation for non-realizable case)
        R_val = 1.0
        solver = VolterraSolver(eigvals, gamma, r, R=R_val, R_tilde=0.0)
        psi, t_theory = solver.solve(t_max=num_epochs, dt=0.05)

        # Empirical SGD
        sgd_runs = []
        for run in range(num_runs):
            model = LeastSquaresSGD(A, b, learning_rate=gamma/n, batch_size=1)
            loss_hist = model.train(steps)
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


# =============================================================================
# MAIN EXPERIMENT LOOP
# =============================================================================
print("\n" + "="*60)
print("STARTING FULL NON-LINEAR TARGET EXPERIMENTS")
print("="*60)

start_time = time.time()
all_results = {}
all_eigvals = {}

for idx, r in enumerate(aspect_ratios):
    exp_start = time.time()
    print(f"\n{'='*60}")
    print(f"EXPERIMENT {idx+1}/{len(aspect_ratios)}: r = {r}")
    print(f"{'='*60}")

    # Build random features
    A, W = build_random_features(X_mnist, r, device)

    # Eigenvalue analysis and scaling FIRST
    A_scaled, eigvals, gamma_max_theory, gamma_max_safe, spectral_info = \
        analyze_and_scale_spectrum(A, r, n)

    # Store eigenvalues
    all_eigvals[r] = eigvals

    # Generate NON-LINEAR planted targets (after scaling)
    x_star, b_scaled, x0, initial_loss, zero_fraction = \
        generate_nonlinear_planted_targets(A_scaled, device)

    # Run sweeps
    results_safe = run_experiment_sweep(
        A_scaled, b_scaled, x_star, eigvals,
        gamma_max_safe, r, n, num_epochs, num_runs,
        sweep_name="Safe"
    )

    results_theory = run_experiment_sweep(
        A_scaled, b_scaled, x_star, eigvals,
        gamma_max_theory, r, n, num_epochs, num_runs,
        sweep_name="Theory"
    )

    # Store results
    all_results[r] = {
        'spectral_info': spectral_info,
        'gamma_max_theory': gamma_max_theory,
        'gamma_max_safe': gamma_max_safe,
        'initial_loss': initial_loss,
        'zero_fraction': zero_fraction,
        'eigvals': eigvals,
        'results_safe': results_safe,
        'results_theory': results_theory
    }

    exp_time = time.time() - exp_start
    print(f"\nCompleted r = {r} in {exp_time/60:.2f} minutes")

total_time = time.time() - start_time

# Save results
np.save(os.path.join(output_dir, 'all_results.npy'), all_results)
print(f"\nResults saved to {output_dir}/all_results.npy")

# =============================================================================
# FIGURE 1: LOSS CURVES COMPARISON
# =============================================================================
print("\nGenerating Figure 1: Loss Curves...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, r in enumerate(aspect_ratios):
    ax = axes[idx]
    res = all_results[r]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, (key, data) in enumerate(res['results_safe'].items()):
        t_sgd = np.arange(len(data['sgd_mean'])) / n

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
    ax.set_title(f'r = {r} (d = {int(n*r)})\nZero fraction = {res["zero_fraction"]:.1%}',
                fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')

plt.suptitle('Non-Linear Target: Volterra vs SGD (Safe Learning Rates)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/loss_curves.png")

# =============================================================================
# FIGURE 2: ZERO FRACTION ANALYSIS
# =============================================================================
print("Generating Figure 2: Zero Fraction Analysis...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, r in enumerate(aspect_ratios):
    ax = axes[idx]
    res = all_results[r]

    # Get the target vector statistics
    zero_frac = res['zero_fraction']
    nonzero_frac = 1 - zero_frac

    # Pie chart
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

# =============================================================================
# FIGURE 3: VOLTERRA VS SGD SCATTER
# =============================================================================
print("Generating Figure 3: Volterra vs SGD Scatter...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, r in enumerate(aspect_ratios):
    ax = axes[idx]
    res = all_results[r]

    all_volterra = []
    all_sgd = []

    for key, data in res['results_safe'].items():
        t_sgd = np.arange(len(data['sgd_mean'])) / n
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

# =============================================================================
# FIGURE 4: COMPARISON WITH BASE (Final Loss)
# =============================================================================
print("Generating Figure 4: Final Loss Comparison...")

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(aspect_ratios))
width = 0.35

# Final losses from non-linear experiment
final_losses = [all_results[r]['results_safe']['0.90']['sgd_mean'][-1] for r in aspect_ratios]
zero_fractions = [all_results[r]['zero_fraction'] for r in aspect_ratios]

bars = ax.bar(x, final_losses, width, label='Final Loss (non-realizable)', color='coral')

# Annotate with zero fraction
for i, (bar, zf) in enumerate(zip(bars, zero_fractions)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'Zero: {zf:.1%}', ha='center', va='bottom', fontsize=10)

ax.set_xlabel('Aspect Ratio', fontsize=12)
ax.set_ylabel('Final Loss', fontsize=12)
ax.set_title('Non-Linear Target: Irreducible Error from ReLU\n(Loss cannot reach zero)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'r={r}' for r in aspect_ratios])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'final_loss_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/final_loss_comparison.png")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*60)
print("FULL NON-LINEAR TARGET RUN COMPLETE")
print("="*60)

print(f"\nTotal runtime: {total_time/60:.2f} minutes")
print(f"\nResults summary:")
for r in aspect_ratios:
    res = all_results[r]
    final_loss = res['results_safe']['0.90']['sgd_mean'][-1]
    print(f"\n  r = {r}:")
    print(f"    Zero fraction: {res['zero_fraction']:.2%}")
    print(f"    Final loss: {final_loss:.4f} (non-zero due to ReLU)")
    print(f"    Spectral ratio: {res['spectral_info']['spectral_ratio']:.2f}")

print(f"\nFigures saved to {output_dir}/:")
print("  - loss_curves.png")
print("  - zero_fraction.png")
print("  - volterra_vs_sgd_scatter.png")
print("  - final_loss_comparison.png")
