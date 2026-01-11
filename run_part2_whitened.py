"""
Experiment: Whitened MNIST Data

This experiment tests Volterra theory when MNIST data is whitened BEFORE
applying random features.

Key difference from run_part2.py: X_whitened = Sigma^{-1/2}(X - mu) is used
instead of raw MNIST X. This makes E[x]=0 and E[xx^T]=I (identity covariance).

The goal is to test whether Volterra theory works better when the Gaussian
assumption (which the theory relies on) is more closely satisfied.
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
from tqdm import tqdm

from src.data_loader import load_mnist, whiten_data
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

# Global parameters
n = 2000
num_epochs = 15
num_runs = 2

print(f"\nParameters: n={n}, epochs={num_epochs}, runs={num_runs}")
print("\n*** EXPERIMENT: Whitened MNIST Data ***")

# Load MNIST
print("\nLoading MNIST...")
X_mnist, y_mnist = load_mnist(root='./data', train=True, flatten=True,
                               subset_size=n, download=True)
X_mnist = X_mnist.to(device)
print(f"X_mnist shape: {X_mnist.shape}")

# WHITENING STEP - KEY DIFFERENCE FROM run_part2.py
print("\n=== Whitening MNIST Data ===")
print("Goal: E[x] = 0, E[xx^T] = I")

# Check statistics before whitening
print("\nBefore whitening:")
print(f"  Mean: {X_mnist.mean().item():.4f}")
cov_before = (X_mnist.T @ X_mnist) / n
print(f"  Cov diagonal mean: {cov_before.diag().mean().item():.4f}")
print(f"  Cov off-diagonal mean: {(cov_before - torch.diag(cov_before.diag())).abs().mean().item():.6f}")

# Apply whitening
X_whitened, whiten_info = whiten_data(X_mnist)
print(f"\nWhitening complete!")
print(f"  Condition number of original covariance: {whiten_info['condition_number']:.2f}")

# Check statistics after whitening
print("\nAfter whitening:")
print(f"  Mean: {X_whitened.mean().item():.6f}")
cov_after = (X_whitened.T @ X_whitened) / n
print(f"  Cov diagonal mean: {cov_after.diag().mean().item():.4f}")
print(f"  Cov off-diagonal mean: {(cov_after - torch.diag(cov_after.diag())).abs().mean().item():.6f}")

# Use whitened data for the rest of the experiment
X_data = X_whitened


def build_random_features(X, r, device):
    """Builds random features A = sigma(XW) with per-column centering."""
    n_samples = X.shape[0]
    d_in = X.shape[1]
    d_out = int(n_samples * r)

    print(f"\n=== Building Random Features (r={r}) ===")
    print(f"Input: X_whitened in R^({n_samples} x {d_in})")
    print(f"Output: A in R^({n_samples} x {d_out})")

    W = generate_random_weights(d_in, d_out).to(device)
    A = compute_features(X, W, activation='shifted_relu', scale_by_sqrt_d=True)

    # Per-column centering
    col_means_before = A.mean(dim=0)
    max_col_mean_before = col_means_before.abs().max().item()
    print(f"Before centering: max |column mean| = {max_col_mean_before:.4f}")

    A = A - A.mean(dim=0, keepdim=True)

    max_col_mean_after = A.mean(dim=0).abs().max().item()
    print(f"After centering: max |column mean| = {max_col_mean_after:.6f}")

    diagnostics = {
        'max_col_mean_before': max_col_mean_before,
        'max_col_mean_after': max_col_mean_after,
    }

    return A, W, diagnostics


def generate_planted_targets(A, device):
    """Generate planted targets b = A @ x* with ||x*|| = 1."""
    n, d = A.shape

    x_star = torch.randn(d, device=device)
    x_star = x_star / torch.norm(x_star)

    b = A @ x_star

    x0 = torch.zeros(d, device=device)
    initial_loss = (1.0 / (2 * n)) * torch.sum(b**2).item()

    print(f"\n=== Planted Targets ===")
    print(f"||x*||^2 = {torch.norm(x_star).item()**2:.4f}")
    print(f"Initial loss f(x0) = {initial_loss:.4f}")

    return x_star, b, x0, initial_loss


def analyze_and_scale_spectrum(A, r, n):
    """Compute eigenvalues, scale A, and compute both gamma_max values."""
    print(f"\n=== Eigenvalue Analysis ===")

    print("Computing eigenvalues...", flush=True)
    eigvals_initial = compute_eigenvalues(A)
    mean_eig_initial = np.mean(eigvals_initial)

    print(f"Before scaling:")
    print(f"  mean(lambda) = {mean_eig_initial:.4f}")

    scale_factor = 1.0 / np.sqrt(mean_eig_initial)
    A_scaled = A * scale_factor
    print(f"\nScaling A by {scale_factor:.4f}")

    print("Recomputing eigenvalues after scaling...", flush=True)
    eigvals = compute_eigenvalues(A_scaled)
    mean_eig = np.mean(eigvals)
    max_eig = np.max(eigvals)
    min_eig = np.min(eigvals[eigvals > 1e-10])

    print(f"\nAfter scaling:")
    print(f"  mean(lambda) = {mean_eig:.4f}")
    print(f"  lambda_max = {max_eig:.4f}")
    print(f"  lambda_min (non-zero) = {min_eig:.6f}")

    gamma_max_theory = (2.0 / r) / mean_eig
    gamma_max_safe = 2.0 / max_eig

    spectral_ratio = max_eig / mean_eig

    print(f"\n=== Critical Step Sizes ===")
    print(f"  gamma_max_theory = {gamma_max_theory:.4f}")
    print(f"  gamma_max_safe = {gamma_max_safe:.4f}")
    print(f"  lambda_max / mean(lambda) = {spectral_ratio:.2f}")
    print(f"  (Compare to non-whitened: typically 100-200x)")

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
    multipliers = [1/4, 1/2, 0.9]

    print(f"\n=== Running {sweep_name} Sweep (gamma_max = {gamma_max:.4f}) ===")

    results = {}

    for i, mult in enumerate(multipliers):
        gamma = gamma_max * mult
        key = f"{mult:.4f}"
        print(f"\n  [{i+1}/{len(multipliers)}] gamma = {gamma:.4f}")

        # Volterra Theory
        print(f"      Computing Volterra...", end=" ", flush=True)
        R_val = 1.0
        solver = VolterraSolver(eigvals, gamma, r, R=R_val, R_tilde=0.0)
        psi, t_theory = solver.solve(t_max=num_epochs, dt=0.1)
        print(f"Done!")

        # Empirical SGD
        print(f"      Running SGD ({num_runs} runs)...")
        sgd_runs = []
        for run in range(num_runs):
            model = LeastSquaresSGD(A, b, learning_rate=gamma/n, batch_size=1)
            loss_hist = model.train(steps)
            sgd_runs.append(loss_hist)
            print(f"        Run {run+1}/{num_runs}: final loss = {loss_hist[-1]:.4f}")

        sgd_mean = np.mean(sgd_runs, axis=0)
        sgd_std = np.std(sgd_runs, axis=0)

        results[key] = {
            'gamma': gamma,
            'mult': mult,
            't_theory': t_theory,
            'psi': psi,
            'sgd_mean': sgd_mean,
            'sgd_std': sgd_std
        }

    return results


# Main Experiment Loop
print("\n" + "="*60)
print("STARTING WHITENED MNIST EXPERIMENTS")
print("="*60)

start_time = time.time()
all_results = {}
aspect_ratios = [0.5, 1.0, 1.2]

for idx, r in enumerate(aspect_ratios):
    exp_start = time.time()
    print(f"\n{'='*60}")
    print(f"EXPERIMENT {idx+1}/3: r = {r}")
    print(f"{'='*60}")

    # Build random features from WHITENED data
    A, W, feat_diag = build_random_features(X_data, r, device)

    # Generate planted targets
    x_star, b, x0, initial_loss = generate_planted_targets(A, device)

    # Eigenvalue analysis and scaling
    A_scaled, eigvals, gamma_max_theory, gamma_max_safe, spectral_info = \
        analyze_and_scale_spectrum(A, r, n)

    # Recompute b after scaling A
    b_scaled = A_scaled @ x_star
    initial_loss_scaled = (1.0 / (2 * n)) * torch.sum(b_scaled**2).item()
    print(f"\nAfter scaling: Initial loss f(x0) = {initial_loss_scaled:.4f}")

    # Run both sweeps
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
        'feat_diag': feat_diag,
        'spectral_info': spectral_info,
        'gamma_max_theory': gamma_max_theory,
        'gamma_max_safe': gamma_max_safe,
        'initial_loss': initial_loss_scaled,
        'whiten_condition_number': whiten_info['condition_number'],  # Extra info
        'results_safe': results_safe,
        'results_theory': results_theory
    }

    exp_time = time.time() - exp_start
    print(f"\nCompleted r = {r} in {exp_time/60:.2f} minutes")

total_time = time.time() - start_time

# Save results
results_dir = './results/part2_whitened'
os.makedirs(results_dir, exist_ok=True)
np.save(os.path.join(results_dir, 'all_results.npy'), all_results)

# Summary
print("\n" + "="*60)
print("SUMMARY - WHITENED MNIST EXPERIMENT")
print("="*60)
print(f"\nWhitening condition number: {whiten_info['condition_number']:.2f}")

for r in aspect_ratios:
    res = all_results[r]
    print(f"\nr = {r}:")
    print(f"  d = {int(n * r)}")
    print(f"  Spectral ratio = {res['spectral_info']['spectral_ratio']:.2f}")
    print(f"  gamma_max_theory = {res['gamma_max_theory']:.4f}")
    print(f"  gamma_max_safe = {res['gamma_max_safe']:.4f}")
    print(f"  Initial loss = {res['initial_loss']:.4f}")

print(f"\n" + "="*60)
print(f"TOTAL RUNTIME: {total_time/60:.2f} minutes")
print(f"Results saved to: {results_dir}")
print("="*60)
print("\nNote: Compare spectral ratios to non-whitened experiment.")
print("Whitening should reduce spectral ratio, making theory more applicable.")
