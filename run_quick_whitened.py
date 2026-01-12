"""
Quick Run: Whitened MNIST Experiment
Parameters: n=2000, epochs=10, runs=3 (~15 min runtime)
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

from src.data_loader import load_mnist, whiten_data
from src.random_features import generate_random_weights, compute_features
from src.spectral import compute_eigenvalues, marchenko_pastur_density
from src.sgd import LeastSquaresSGD
from src.volterra import VolterraSolver

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Quick run parameters
n = 2000
num_epochs = 10
num_runs = 3
aspect_ratios = [0.5, 1.0, 1.5]

output_dir = './results/quick_whitened'
os.makedirs(output_dir, exist_ok=True)

print(f"\n{'='*60}")
print("QUICK RUN: WHITENED MNIST EXPERIMENT")
print(f"{'='*60}")
print(f"Parameters: n={n}, epochs={num_epochs}, runs={num_runs}")
print(f"Aspect ratios: {aspect_ratios}")

# Load and whiten MNIST
print("\nLoading MNIST...")
X_mnist, y_mnist = load_mnist(root='./data', train=True, flatten=True,
                               subset_size=n, download=True)
X_mnist = X_mnist.to(device)
print(f"X_mnist shape: {X_mnist.shape}")

print("\n=== Whitening MNIST Data ===")
X_whitened, whiten_info = whiten_data(X_mnist)
print(f"Condition number of original covariance: {whiten_info['condition_number']:.2f}")

# Check spectral properties before/after whitening
cov_before = (X_mnist.T @ X_mnist) / n
eigvals_before = torch.linalg.eigvalsh(cov_before).cpu().numpy()
cov_after = (X_whitened.T @ X_whitened) / n
eigvals_after = torch.linalg.eigvalsh(cov_after).cpu().numpy()

print(f"\nBefore whitening: eigenvalue range [{eigvals_before.min():.4f}, {eigvals_before.max():.4f}]")
print(f"After whitening: eigenvalue range [{eigvals_after.min():.4f}, {eigvals_after.max():.4f}]")

X_data = X_whitened

def build_random_features(X, r, device):
    n_samples = X.shape[0]
    d_in = X.shape[1]
    d_out = int(n_samples * r)

    print(f"\n=== Building Random Features (r={r}) ===")
    print(f"Input: X_whitened in R^({n_samples} x {d_in})")
    print(f"Output: A in R^({n_samples} x {d_out})")

    W = generate_random_weights(d_in, d_out).to(device)
    A = compute_features(X, W, activation='shifted_relu', scale_by_sqrt_d=True)
    A = A - A.mean(dim=0, keepdim=True)

    return A, W

def run_sweep(A, b, eigvals, gamma_max, r, n, num_epochs, num_runs, sweep_name):
    steps = n * num_epochs
    multipliers = [0.25, 0.5, 0.9]
    results = {}

    print(f"\n=== {sweep_name} Sweep (gamma_max = {gamma_max:.4f}) ===")

    for mult in multipliers:
        gamma = gamma_max * mult
        print(f"  gamma = {gamma:.4f} ({int(mult*100)}% of gamma_max)...", flush=True)

        solver = VolterraSolver(eigvals, gamma, r, R=1.0, R_tilde=0.0)
        psi, t_theory = solver.solve(t_max=num_epochs, dt=0.1)

        sgd_runs = []
        for run in range(num_runs):
            model = LeastSquaresSGD(A, b, learning_rate=gamma/n, batch_size=1)
            sgd_runs.append(model.train(steps))

        results[mult] = {
            'gamma': gamma,
            'psi': psi,
            't_theory': t_theory,
            'sgd_mean': np.mean(sgd_runs, axis=0),
            'sgd_std': np.std(sgd_runs, axis=0)
        }

    return results

# Main experiment loop
print("\n" + "="*60)
print("STARTING WHITENED MNIST EXPERIMENTS")
print("="*60)

start_time = time.time()
all_results = {}

for idx, r in enumerate(aspect_ratios):
    exp_start = time.time()
    print(f"\n{'='*60}")
    print(f"EXPERIMENT {idx+1}/{len(aspect_ratios)}: r = {r}")
    print(f"{'='*60}")

    A, W = build_random_features(X_data, r, device)

    # Eigenvalue analysis
    print("Computing eigenvalues...")
    eigvals = compute_eigenvalues(A)
    mean_eig = np.mean(eigvals)

    scale_factor = 1.0 / np.sqrt(mean_eig)
    A = A * scale_factor
    eigvals = compute_eigenvalues(A)

    mean_eig = np.mean(eigvals)
    max_eig = np.max(eigvals)
    spectral_ratio = max_eig / mean_eig

    gamma_theory = 2.0 / (r * mean_eig)
    gamma_safe = 2.0 / max_eig

    print(f"  mean(lambda) = {mean_eig:.4f}, max(lambda) = {max_eig:.4f}")
    print(f"  Spectral ratio = {spectral_ratio:.2f}")
    print(f"  gamma_theory = {gamma_theory:.4f}, gamma_safe = {gamma_safe:.4f}")

    # Generate planted targets
    d = A.shape[1]
    x_star = torch.randn(d, device=device)
    x_star = x_star / torch.norm(x_star)
    b = A @ x_star

    # Run sweeps
    results_safe = run_sweep(A, b, eigvals, gamma_safe, r, n, num_epochs, num_runs, "Safe")
    results_theory = run_sweep(A, b, eigvals, gamma_theory, r, n, num_epochs, num_runs, "Theory")

    all_results[r] = {
        'eigvals': eigvals,
        'spectral_ratio': spectral_ratio,
        'gamma_theory': gamma_theory,
        'gamma_safe': gamma_safe,
        'results_safe': results_safe,
        'results_theory': results_theory
    }

    print(f"\nCompleted r = {r} in {(time.time()-exp_start)/60:.2f} minutes")

np.save(os.path.join(output_dir, 'all_results.npy'), all_results)

# Generate figures
print("\nGenerating Figure 1: Loss Curves...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, r in enumerate(aspect_ratios):
    ax = axes[idx]
    res = all_results[r]

    for mult, data in res['results_safe'].items():
        epochs_sgd = np.linspace(0, num_epochs, len(data['sgd_mean']))
        ax.plot(epochs_sgd, data['sgd_mean'], label=f'SGD (gamma={data["gamma"]:.4f})')
        ax.plot(data['t_theory'], data['psi'], '--', label=f'Volterra')

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title(f'r = {r} (Whitened)')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=150)
plt.close()
print(f"  Saved: {output_dir}/loss_curves.png")

# Figure 2: Spectral comparison (whitened vs expected)
print("Generating Figure 2: Spectral Analysis...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, r in enumerate(aspect_ratios):
    ax = axes[idx]
    eigvals = all_results[r]['eigvals']

    ax.hist(eigvals, bins=50, density=True, alpha=0.7, label='Empirical')

    # MP density for comparison
    lambda_plus = (1 + np.sqrt(r))**2
    lambda_minus = (1 - np.sqrt(r))**2
    x = np.linspace(max(0.01, lambda_minus*0.9), lambda_plus*1.1, 200)
    mp = marchenko_pastur_density(x, r)
    ax.plot(x, mp, 'r-', lw=2, label='Marchenko-Pastur')

    ax.axvline(all_results[r]['spectral_ratio'], color='g', linestyle='--',
               label=f'Spectral ratio: {all_results[r]["spectral_ratio"]:.1f}')
    ax.set_xlabel('Eigenvalue')
    ax.set_ylabel('Density')
    ax.set_title(f'r = {r} (Whitened MNIST)')
    ax.legend(fontsize=8)
    ax.set_xlim(0, min(20, all_results[r]['eigvals'].max()*1.1))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'eigenvalue_spectrum.png'), dpi=150)
plt.close()
print(f"  Saved: {output_dir}/eigenvalue_spectrum.png")

# Figure 3: Correlation scatter
print("Generating Figure 3: Volterra vs SGD Scatter...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
correlations = []
for idx, r in enumerate(aspect_ratios):
    ax = axes[idx]
    res = all_results[r]

    all_volterra = []
    all_sgd = []

    for sweep in [res['results_safe'], res['results_theory']]:
        for mult, data in sweep.items():
            t_theory = data['t_theory']
            psi = data['psi']
            sgd_mean = data['sgd_mean']
            epochs_sgd = np.linspace(0, num_epochs, len(sgd_mean))

            psi_interp = np.interp(epochs_sgd, t_theory, psi)
            all_volterra.extend(psi_interp)
            all_sgd.extend(sgd_mean)

    all_volterra = np.array(all_volterra)
    all_sgd = np.array(all_sgd)

    corr, _ = stats.pearsonr(all_volterra, all_sgd)
    correlations.append(corr)

    ax.scatter(all_volterra, all_sgd, alpha=0.3, s=10)
    ax.plot([0, max(all_sgd)], [0, max(all_sgd)], 'r--', label='y=x')
    ax.set_xlabel('Volterra Prediction')
    ax.set_ylabel('SGD Loss')
    ax.set_title(f'r = {r}, Corr = {corr:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'volterra_vs_sgd_scatter.png'), dpi=150)
plt.close()
print(f"  Saved: {output_dir}/volterra_vs_sgd_scatter.png")

# Figure 4: Summary comparison with non-whitened
print("Generating Figure 4: Summary Metrics...")
fig, ax = plt.subplots(figsize=(10, 6))
x_pos = np.arange(len(aspect_ratios))
width = 0.35

spectral_ratios = [all_results[r]['spectral_ratio'] for r in aspect_ratios]
# Compare with base experiment spectral ratios (from FINDINGS.md)
base_spectral_ratios = [133.2, 270.6, 272.4]

bars1 = ax.bar(x_pos - width/2, spectral_ratios, width, label='Whitened', color='blue', alpha=0.7)
bars2 = ax.bar(x_pos + width/2, base_spectral_ratios, width, label='Non-whitened', color='red', alpha=0.7)

ax.set_xlabel('Aspect Ratio')
ax.set_ylabel('Spectral Ratio (lambda_max / lambda_mean)')
ax.set_title('Effect of Whitening on Spectral Ratio')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'r={r}' for r in aspect_ratios])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars1, spectral_ratios):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}',
            ha='center', va='bottom', fontsize=9)
for bar, val in zip(bars2, base_spectral_ratios):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}',
            ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'summary_comparison.png'), dpi=150)
plt.close()
print(f"  Saved: {output_dir}/summary_comparison.png")

total_time = time.time() - start_time

print(f"\n{'='*60}")
print("WHITENED MNIST EXPERIMENT COMPLETE")
print(f"{'='*60}")
print(f"\nTotal runtime: {total_time/60:.2f} minutes")
print(f"\nResults summary:")
for r in aspect_ratios:
    res = all_results[r]
    print(f"\n  r = {r}:")
    print(f"    Spectral ratio: {res['spectral_ratio']:.2f}")
    print(f"    Correlation: {correlations[aspect_ratios.index(r)]:.4f}")
    print(f"    gamma_theory: {res['gamma_theory']:.4f}")
    print(f"    gamma_safe: {res['gamma_safe']:.4f}")

print(f"\nFigures saved to {output_dir}/")
