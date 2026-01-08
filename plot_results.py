"""
Quick visualization script - run after run_part2.py
Or run standalone (will re-run experiments quickly)
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import load_mnist
from src.random_features import generate_random_weights, compute_features
from src.spectral import compute_eigenvalues
from src.sgd import LeastSquaresSGD
from src.volterra import VolterraSolver

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Parameters (even smaller for quick plotting)
n = 1000
num_epochs = 10
num_runs = 2

print(f"\nRunning quick experiment for plotting: n={n}, epochs={num_epochs}")

# Load MNIST
X_mnist, _ = load_mnist(root='./data', train=True, flatten=True,
                         subset_size=n, download=True)
X_mnist = X_mnist.to(device)

def run_single_experiment(X, r, n, num_epochs, num_runs):
    """Run experiment for one aspect ratio."""
    d = int(n * r)

    # Build features
    W = generate_random_weights(X.shape[1], d).to(device)
    A = compute_features(X, W, activation='shifted_relu', scale_by_sqrt_d=True)
    A = A - A.mean(dim=0, keepdim=True)

    # Generate target
    x_star = torch.randn(d, device=device)
    x_star = x_star / torch.norm(x_star)
    b = A @ x_star

    # Compute and scale eigenvalues
    eigvals = compute_eigenvalues(A)
    mean_eig = np.mean(eigvals)
    scale_factor = 1.0 / np.sqrt(mean_eig)
    A_scaled = A * scale_factor
    b_scaled = A_scaled @ x_star
    eigvals_scaled = compute_eigenvalues(A_scaled)

    mean_eig_scaled = np.mean(eigvals_scaled)
    max_eig_scaled = np.max(eigvals_scaled)

    # Compute gamma_max
    gamma_max_theory = (2.0 / r) / mean_eig_scaled
    gamma = gamma_max_theory * 0.5  # Use 50%

    print(f"  r={r}: gamma_max={gamma_max_theory:.4f}, using gamma={gamma:.4f}")

    # Volterra
    solver = VolterraSolver(eigvals_scaled, gamma, r, R=1.0, R_tilde=0.0)
    psi, t_theory = solver.solve(t_max=num_epochs, dt=0.1)

    # SGD
    steps = n * num_epochs
    sgd_runs = []
    for run in range(num_runs):
        model = LeastSquaresSGD(A_scaled, b_scaled, learning_rate=gamma/n, batch_size=1)
        loss_hist = model.train(steps)
        sgd_runs.append(loss_hist)

    sgd_mean = np.mean(sgd_runs, axis=0)
    sgd_std = np.std(sgd_runs, axis=0)

    return {
        'gamma': gamma,
        't_theory': t_theory,
        'psi': psi,
        'sgd_mean': sgd_mean,
        'sgd_std': sgd_std,
        'steps': steps
    }

# Run experiments
print("\nRunning experiments...")
aspect_ratios = [0.5, 1.0, 1.2]
results = {}

for r in aspect_ratios:
    print(f"Running r={r}...")
    results[r] = run_single_experiment(X_mnist, r, n, num_epochs, num_runs)

# Create plots
print("\nCreating plots...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, r in enumerate(aspect_ratios):
    ax = axes[idx]
    res = results[r]

    # Epochs for SGD
    sgd_epochs = np.arange(len(res['sgd_mean'])) / n

    # Plot Volterra
    ax.plot(res['t_theory'], res['psi'], 'b-', linewidth=2, label='Volterra Theory')

    # Plot SGD
    ax.plot(sgd_epochs, res['sgd_mean'], 'r-', linewidth=1.5, label='Empirical SGD', alpha=0.8)
    ax.fill_between(sgd_epochs,
                     res['sgd_mean'] - res['sgd_std'],
                     res['sgd_mean'] + res['sgd_std'],
                     color='red', alpha=0.2)

    ax.set_xlabel('Epochs', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title(f'r = {r}, γ = {res["gamma"]:.3f}', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig('volterra_results.png', dpi=150, bbox_inches='tight')
print("\nSaved: volterra_results.png")

# Create eigenvalue distribution plot
print("Creating eigenvalue plot...")
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))

for idx, r in enumerate(aspect_ratios):
    ax = axes2[idx]
    d = int(n * r)

    # Recompute for eigenvalue plot
    W = generate_random_weights(X_mnist.shape[1], d).to(device)
    A = compute_features(X_mnist, W, activation='shifted_relu', scale_by_sqrt_d=True)
    A = A - A.mean(dim=0, keepdim=True)
    eigvals = compute_eigenvalues(A)

    # Scale
    scale_factor = 1.0 / np.sqrt(np.mean(eigvals))
    eigvals_scaled = eigvals * (scale_factor ** 2)

    ax.hist(eigvals_scaled[eigvals_scaled > 1e-10], bins=50, density=True, alpha=0.7, color='steelblue')
    ax.axvline(np.mean(eigvals_scaled), color='red', linestyle='--', linewidth=2, label=f'Mean = {np.mean(eigvals_scaled):.2f}')
    ax.set_xlabel('Eigenvalue', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'Eigenvalue Distribution (r = {r})', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eigenvalue_distributions.png', dpi=150, bbox_inches='tight')
print("Saved: eigenvalue_distributions.png")

print("\nDone! Check:")
print("  - volterra_results.png")
print("  - eigenvalue_distributions.png")
