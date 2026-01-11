"""
Experiment: Real MNIST Labels (Classification as Regression)

This experiment tests Volterra theory on actual MNIST classification task.
Key difference from run_part2.py: Uses one-hot encoded MNIST labels Y (n x 10)
as targets instead of synthetic planted targets.

This is a 10-dimensional regression problem:
    min (1/2n) ||AX - Y||_F^2
where X is (d x 10) and Y is one-hot encoded labels.

The Volterra theory should predict the average loss dynamics across
all 10 output dimensions.
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

from src.data_loader import load_mnist, labels_to_onehot
from src.random_features import generate_random_weights, compute_features
from src.spectral import compute_eigenvalues, marchenko_pastur_density
from src.sgd import MultiOutputLeastSquaresSGD
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
print("\n*** EXPERIMENT: Real MNIST Labels (One-Hot Classification as Regression) ***")

# Load MNIST with LABELS
print("\nLoading MNIST...")
X_mnist, y_mnist = load_mnist(root='./data', train=True, flatten=True,
                               subset_size=n, download=True)
X_mnist = X_mnist.to(device)
y_mnist = y_mnist.to(device)
print(f"X_mnist shape: {X_mnist.shape}")
print(f"y_mnist shape: {y_mnist.shape}")

# Convert labels to one-hot encoding - KEY DIFFERENCE
Y_onehot = labels_to_onehot(y_mnist, num_classes=10)
print(f"\n=== One-Hot Encoded Labels ===")
print(f"Y_onehot shape: {Y_onehot.shape} (n x num_classes)")
print(f"Label distribution: {torch.bincount(y_mnist).cpu().numpy()}")


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


def compute_optimal_R(A, Y):
    """
    Compute R parameter for Volterra solver with real labels.

    For multi-output regression starting from X=0:
    R = (1/k) * ||X_opt||_F^2 where X_opt = A^+ @ Y (least squares solution)

    This represents the average squared norm of the optimal solution per output.
    """
    # Compute least squares solution: X_opt = (A^T A)^{-1} A^T Y
    X_opt = torch.linalg.lstsq(A, Y).solution  # (d, k)

    # R = average squared Frobenius norm per output dimension
    R_val = torch.sum(X_opt**2).item() / Y.shape[1]

    print(f"\n=== Optimal Solution Statistics ===")
    print(f"X_opt shape: {X_opt.shape}")
    print(f"||X_opt||_F^2 = {torch.sum(X_opt**2).item():.4f}")
    print(f"R (per-output average) = {R_val:.4f}")

    return R_val, X_opt


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

    spectral_info = {
        'mean_eig': mean_eig,
        'max_eig': max_eig,
        'min_eig': min_eig,
        'spectral_ratio': spectral_ratio,
        'scale_factor': scale_factor
    }

    return A_scaled, eigvals, gamma_max_theory, gamma_max_safe, spectral_info


def run_experiment_sweep(A, Y, R_val, eigvals, gamma_max, r, n, num_epochs, num_runs, sweep_name=""):
    """Run multi-output SGD + Volterra comparison for multiple learning rates."""
    d = A.shape[1]
    k = Y.shape[1]  # number of outputs (10 for MNIST)
    steps = n * num_epochs
    multipliers = [1/4, 1/2, 0.9]

    print(f"\n=== Running {sweep_name} Sweep (gamma_max = {gamma_max:.4f}) ===")
    print(f"    Multi-output regression: {k} output dimensions")

    results = {}

    for i, mult in enumerate(multipliers):
        gamma = gamma_max * mult
        key = f"{mult:.4f}"
        print(f"\n  [{i+1}/{len(multipliers)}] gamma = {gamma:.4f}")

        # Volterra Theory (using computed R for real labels)
        print(f"      Computing Volterra (R={R_val:.4f})...", end=" ", flush=True)
        solver = VolterraSolver(eigvals, gamma, r, R=R_val, R_tilde=0.0)
        psi, t_theory = solver.solve(t_max=num_epochs, dt=0.1)
        print(f"Done!")

        # Empirical Multi-Output SGD
        print(f"      Running Multi-Output SGD ({num_runs} runs)...")
        sgd_runs = []
        for run in range(num_runs):
            model = MultiOutputLeastSquaresSGD(A, Y, learning_rate=gamma/n, batch_size=1)
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
print("STARTING REAL MNIST LABELS EXPERIMENTS")
print("="*60)

start_time = time.time()
all_results = {}
aspect_ratios = [0.5, 1.0, 1.2]

for idx, r in enumerate(aspect_ratios):
    exp_start = time.time()
    print(f"\n{'='*60}")
    print(f"EXPERIMENT {idx+1}/3: r = {r}")
    print(f"{'='*60}")

    # Build random features
    A, W, feat_diag = build_random_features(X_mnist, r, device)

    # Eigenvalue analysis and scaling
    A_scaled, eigvals, gamma_max_theory, gamma_max_safe, spectral_info = \
        analyze_and_scale_spectrum(A, r, n)

    # Compute R for the real labels problem (after scaling A)
    R_val, X_opt = compute_optimal_R(A_scaled, Y_onehot)

    # Compute initial loss with one-hot targets
    initial_loss = (1.0 / (2 * n)) * torch.sum(Y_onehot**2).item()
    print(f"\nInitial loss f(X=0) = {initial_loss:.4f}")
    print(f"(This is (1/2n)||Y||_F^2 since X starts at 0)")

    # Run both sweeps
    results_safe = run_experiment_sweep(
        A_scaled, Y_onehot, R_val, eigvals,
        gamma_max_safe, r, n, num_epochs, num_runs,
        sweep_name="Safe"
    )

    results_theory = run_experiment_sweep(
        A_scaled, Y_onehot, R_val, eigvals,
        gamma_max_theory, r, n, num_epochs, num_runs,
        sweep_name="Theory"
    )

    # Store results
    all_results[r] = {
        'feat_diag': feat_diag,
        'spectral_info': spectral_info,
        'gamma_max_theory': gamma_max_theory,
        'gamma_max_safe': gamma_max_safe,
        'initial_loss': initial_loss,
        'R_value': R_val,  # Extra info for this experiment
        'num_outputs': Y_onehot.shape[1],
        'results_safe': results_safe,
        'results_theory': results_theory
    }

    exp_time = time.time() - exp_start
    print(f"\nCompleted r = {r} in {exp_time/60:.2f} minutes")

total_time = time.time() - start_time

# Save results
results_dir = './results/part2_real_labels'
os.makedirs(results_dir, exist_ok=True)
np.save(os.path.join(results_dir, 'all_results.npy'), all_results)

# Summary
print("\n" + "="*60)
print("SUMMARY - REAL MNIST LABELS EXPERIMENT")
print("="*60)
print(f"\nTask: 10-class classification as 10-output regression")
print(f"Targets: One-hot encoded MNIST labels")

for r in aspect_ratios:
    res = all_results[r]
    print(f"\nr = {r}:")
    print(f"  d = {int(n * r)}")
    print(f"  Spectral ratio = {res['spectral_info']['spectral_ratio']:.2f}")
    print(f"  gamma_max_theory = {res['gamma_max_theory']:.4f}")
    print(f"  gamma_max_safe = {res['gamma_max_safe']:.4f}")
    print(f"  Initial loss = {res['initial_loss']:.4f}")
    print(f"  R (optimal solution norm) = {res['R_value']:.4f}")

print(f"\n" + "="*60)
print(f"TOTAL RUNTIME: {total_time/60:.2f} minutes")
print(f"Results saved to: {results_dir}")
print("="*60)
print("\nNote: This tests whether Volterra theory predicts real classification dynamics.")
