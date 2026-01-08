"""
Quick GPU test - minimal version of Part 2
Should run in < 1 minute
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import torch
import numpy as np
from src.data_loader import load_mnist
from src.random_features import generate_random_weights, compute_features
from src.spectral import compute_eigenvalues
from src.sgd import LeastSquaresSGD
from src.volterra import VolterraSolver

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# MINIMAL parameters for quick test
n = 100  # Very small
r = 1.0
d = int(n * r)
num_epochs = 2
num_runs = 1

print(f"\nTest Parameters: n={n}, d={d}, r={r}, epochs={num_epochs}")

# Load tiny MNIST subset
print("\nLoading MNIST subset...")
X_mnist, y_mnist = load_mnist(root='./data', train=True, flatten=True,
                               subset_size=n, download=True)
X_mnist = X_mnist.to(device)
print(f"X_mnist shape: {X_mnist.shape}, device: {X_mnist.device}")

# Build random features
print("\nBuilding random features...")
W = generate_random_weights(X_mnist.shape[1], d).to(device)
A = compute_features(X_mnist, W, activation='shifted_relu', scale_by_sqrt_d=True)
A = A - A.mean(dim=0, keepdim=True)  # Center
print(f"A shape: {A.shape}, device: {A.device}")

# Generate planted target
print("\nGenerating planted target...")
x_star = torch.randn(d, device=device)
x_star = x_star / torch.norm(x_star)
b = A @ x_star
initial_loss = (1.0 / (2 * n)) * torch.sum(b**2).item()
print(f"Initial loss: {initial_loss:.4f}")

# Eigenvalue analysis
print("\nComputing eigenvalues...")
eigvals = compute_eigenvalues(A)
mean_eig = np.mean(eigvals)
print(f"Mean eigenvalue: {mean_eig:.4f}")

# Scale A
scale_factor = 1.0 / np.sqrt(mean_eig)
A_scaled = A * scale_factor
b_scaled = A_scaled @ x_star
eigvals_scaled = compute_eigenvalues(A_scaled)
mean_eig_scaled = np.mean(eigvals_scaled)
max_eig_scaled = np.max(eigvals_scaled)
print(f"After scaling: mean={mean_eig_scaled:.4f}, max={max_eig_scaled:.4f}")

# Compute gamma_max
gamma_max = (2.0 / r) / mean_eig_scaled
gamma = gamma_max * 0.5  # Use 50% of max
print(f"\ngamma_max: {gamma_max:.4f}")
print(f"Testing gamma: {gamma:.4f}")

# Volterra theory
print("\nSolving Volterra...")
solver = VolterraSolver(eigvals_scaled, gamma, r, R=1.0, R_tilde=0.0)
psi, t_theory = solver.solve(t_max=num_epochs, dt=0.2)  # Large dt for speed
print(f"Volterra computed {len(psi)} points")
print(f"Final Volterra loss: {psi[-1]:.4f}")

# SGD
print("\nRunning SGD...")
steps = n * num_epochs
model = LeastSquaresSGD(A_scaled, b_scaled, learning_rate=gamma/n, batch_size=1)
loss_hist = model.train(steps)
print(f"Final SGD loss: {loss_hist[-1]:.4f}")

print("\n[SUCCESS] Test completed successfully!")
print(f"GPU test passed - all components working on {device}")
