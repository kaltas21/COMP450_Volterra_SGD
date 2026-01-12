"""
Quick Run: Real MNIST Labels Experiment (Classification as Regression)
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

from src.data_loader import load_mnist, labels_to_onehot
from src.random_features import generate_random_weights, compute_features
from src.spectral import compute_eigenvalues, marchenko_pastur_density
from src.sgd import MultiOutputLeastSquaresSGD
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

output_dir = './results/quick_real_labels'
os.makedirs(output_dir, exist_ok=True)

print(f"\n{'='*60}")
print("QUICK RUN: REAL MNIST LABELS EXPERIMENT")
print("(Classification as 10-output Regression)")
print(f"{'='*60}")
print(f"Parameters: n={n}, epochs={num_epochs}, runs={num_runs}")
print(f"Aspect ratios: {aspect_ratios}")

# Load MNIST with labels
print("\nLoading MNIST...")
X_mnist, y_mnist = load_mnist(root='./data', train=True, flatten=True,
                               subset_size=n, download=True)
X_mnist = X_mnist.to(device)
y_mnist = y_mnist.to(device)
print(f"X_mnist shape: {X_mnist.shape}")
print(f"y_mnist shape: {y_mnist.shape}")

# One-hot encode labels
Y_onehot = labels_to_onehot(y_mnist, num_classes=10)
print(f"\n=== One-Hot Encoded Labels ===")
print(f"Y_onehot shape: {Y_onehot.shape}")
print(f"Label distribution: {torch.bincount(y_mnist).cpu().numpy()}")

def build_random_features(X, r, device):
    n_samples = X.shape[0]
    d_in = X.shape[1]
    d_out = int(n_samples * r)

    print(f"\n=== Building Random Features (r={r}) ===")
    print(f"Output: A in R^({n_samples} x {d_out})")

    W = generate_random_weights(d_in, d_out).to(device)
    A = compute_features(X, W, activation='shifted_relu', scale_by_sqrt_d=True)
    A = A - A.mean(dim=0, keepdim=True)

    return A, W

def compute_optimal_R(A, Y):
    """Compute R for Volterra solver from optimal solution."""
    X_opt = torch.linalg.lstsq(A, Y).solution
    R_val = torch.sum(X_opt**2).item() / Y.shape[1]
    return R_val, X_opt

def run_sweep(A, Y, R_val, eigvals, gamma_max, r, n, num_epochs, num_runs, sweep_name):
    steps = n * num_epochs
    multipliers = [0.25, 0.5, 0.9]
    results = {}

    print(f"\n=== {sweep_name} Sweep (gamma_max = {gamma_max:.4f}) ===")

    for mult in multipliers:
        gamma = gamma_max * mult
        print(f"  gamma = {gamma:.4f} ({int(mult*100)}% of gamma_max)...", flush=True)

        solver = VolterraSolver(eigvals, gamma, r, R=R_val, R_tilde=0.0)
        psi, t_theory = solver.solve(t_max=num_epochs, dt=0.1)

        sgd_runs = []
        for run in range(num_runs):
            model = MultiOutputLeastSquaresSGD(A, Y, learning_rate=gamma/n, batch_size=1)
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
print("STARTING REAL MNIST LABELS EXPERIMENTS")
print("="*60)

start_time = time.time()
all_results = {}

for idx, r in enumerate(aspect_ratios):
    exp_start = time.time()
    print(f"\n{'='*60}")
    print(f"EXPERIMENT {idx+1}/{len(aspect_ratios)}: r = {r}")
    print(f"{'='*60}")

    A, W = build_random_features(X_mnist, r, device)

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

    # Compute R for real labels
    R_val, X_opt = compute_optimal_R(A, Y_onehot)
    print(f"  R (optimal solution norm) = {R_val:.4f}")

    # Initial loss
    initial_loss = (1.0 / (2 * n)) * torch.sum(Y_onehot**2).item()
    print(f"  Initial loss f(X=0) = {initial_loss:.4f}")

    # Run sweeps
    results_safe = run_sweep(A, Y_onehot, R_val, eigvals, gamma_safe, r, n, num_epochs, num_runs, "Safe")
    results_theory = run_sweep(A, Y_onehot, R_val, eigvals, gamma_theory, r, n, num_epochs, num_runs, "Theory")

    # Compute classification accuracy at end
    final_X = torch.linalg.lstsq(A, Y_onehot).solution
    predictions = A @ final_X
    predicted_labels = predictions.argmax(dim=1)
    accuracy = (predicted_labels == y_mnist).float().mean().item()
    print(f"  Final classification accuracy: {accuracy*100:.2f}%")

    all_results[r] = {
        'eigvals': eigvals,
        'spectral_ratio': spectral_ratio,
        'gamma_theory': gamma_theory,
        'gamma_safe': gamma_safe,
        'R_val': R_val,
        'initial_loss': initial_loss,
        'accuracy': accuracy,
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
    ax.set_title(f'r = {r}, Acc = {res["accuracy"]*100:.1f}%')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=150)
plt.close()
print(f"  Saved: {output_dir}/loss_curves.png")

# Figure 2: Per-class loss breakdown
print("Generating Figure 2: Per-Class Analysis...")
fig, ax = plt.subplots(figsize=(10, 6))
class_counts = torch.bincount(y_mnist).cpu().numpy()
x_pos = np.arange(10)
ax.bar(x_pos, class_counts, color='steelblue', alpha=0.7)
ax.set_xlabel('Digit Class')
ax.set_ylabel('Count')
ax.set_title('MNIST Label Distribution in Training Set')
ax.set_xticks(x_pos)
ax.set_xticklabels([str(i) for i in range(10)])
ax.grid(True, alpha=0.3, axis='y')

for i, v in enumerate(class_counts):
    ax.text(i, v + 5, str(v), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=150)
plt.close()
print(f"  Saved: {output_dir}/class_distribution.png")

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

# Figure 4: Accuracy vs Aspect Ratio
print("Generating Figure 4: Accuracy Summary...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy
ax1 = axes[0]
accuracies = [all_results[r]['accuracy']*100 for r in aspect_ratios]
ax1.bar(range(len(aspect_ratios)), accuracies, color='green', alpha=0.7)
ax1.set_xlabel('Aspect Ratio')
ax1.set_ylabel('Classification Accuracy (%)')
ax1.set_title('Classification Accuracy vs Aspect Ratio')
ax1.set_xticks(range(len(aspect_ratios)))
ax1.set_xticklabels([f'r={r}' for r in aspect_ratios])
ax1.set_ylim(0, 100)
ax1.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(accuracies):
    ax1.text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=10)

# R values
ax2 = axes[1]
R_vals = [all_results[r]['R_val'] for r in aspect_ratios]
ax2.bar(range(len(aspect_ratios)), R_vals, color='orange', alpha=0.7)
ax2.set_xlabel('Aspect Ratio')
ax2.set_ylabel('R (Optimal Solution Norm)')
ax2.set_title('Optimal Solution Norm vs Aspect Ratio')
ax2.set_xticks(range(len(aspect_ratios)))
ax2.set_xticklabels([f'r={r}' for r in aspect_ratios])
ax2.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(R_vals):
    ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'accuracy_summary.png'), dpi=150)
plt.close()
print(f"  Saved: {output_dir}/accuracy_summary.png")

total_time = time.time() - start_time

print(f"\n{'='*60}")
print("REAL MNIST LABELS EXPERIMENT COMPLETE")
print(f"{'='*60}")
print(f"\nTotal runtime: {total_time/60:.2f} minutes")
print(f"\nResults summary:")
for r in aspect_ratios:
    res = all_results[r]
    print(f"\n  r = {r}:")
    print(f"    Spectral ratio: {res['spectral_ratio']:.2f}")
    print(f"    Classification accuracy: {res['accuracy']*100:.2f}%")
    print(f"    Correlation: {correlations[aspect_ratios.index(r)]:.4f}")
    print(f"    R value: {res['R_val']:.4f}")

print(f"\nFigures saved to {output_dir}/")
