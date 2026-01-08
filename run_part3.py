"""
Part 3: Step-size Criticality Analysis
Empirically map when SGD becomes unstable as learning rate increases.
"""

import sys
import os
import time
import json

# Add parent directory to path
sys.path.append(os.path.abspath('.'))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import project modules
from src.data_loader import load_mnist
from src.random_features import generate_random_weights, compute_features
from src.spectral import compute_eigenvalues
from src.sgd import LeastSquaresSGD

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Parameters
n = 2000  # Consistent with Part 2
num_epochs = 20 # Enough to see divergence
num_runs = 3    # As requested (3-5)

print(f"\nParameters: n={n}, epochs={num_epochs}, runs={num_runs}")

# Load MNIST
print("\nLoading MNIST...")
X_mnist, y_mnist = load_mnist(root='./data', train=True, flatten=True,
                               subset_size=n, download=True)
X_mnist = X_mnist.to(device)

def is_stable(loss_hist, initial_loss):
    """
    Stability criteria:
    1. Loss is finite (not NaN or Inf)
    2. Final loss <= 10 * initial_loss
    """
    if not np.all(np.isfinite(loss_hist)):
        return False
    if loss_hist[-1] > 10 * initial_loss:
        return False
    return True

def run_criticality_experiment(X, r, n, num_epochs, num_runs):
    """Run step-size criticality analysis for a specific aspect ratio."""
    d = int(n * r)
    print(f"\n=== Running Criticality Analysis for r={r} (d={d}) ===")

    # 1. Build Features
    W = generate_random_weights(X.shape[1], d).to(device)
    A = compute_features(X, W, activation='shifted_relu', scale_by_sqrt_d=True)
    
    # Center features
    A = A - A.mean(dim=0, keepdim=True)
    
    # 2. Planted Target
    x_star = torch.randn(d, device=device)
    x_star = x_star / torch.norm(x_star)
    b = A @ x_star
    
    # 3. Eigenvalue Analysis
    print("Computing eigenvalues...")
    eigvals = compute_eigenvalues(A)
    mean_eig = np.mean(eigvals)
    max_eig = np.max(eigvals)
    
    # Scale A so mean(lambda) = 1
    scale_factor = 1.0 / np.sqrt(mean_eig)
    A_scaled = A * scale_factor
    b_scaled = A_scaled @ x_star # Recompute b with scaled A
    
    # Recompute spectral properties after scaling
    eigvals_scaled = compute_eigenvalues(A_scaled)
    mean_eig_scaled = np.mean(eigvals_scaled) # Should be ~1.0
    max_eig_scaled = np.max(eigvals_scaled)
    
    gamma_safe = 2.0 / max_eig_scaled
    gamma_theory = (2.0 / r) / mean_eig_scaled
    
    print(f"  mean(lambda) = {mean_eig_scaled:.4f}")
    print(f"  lambda_max = {max_eig_scaled:.4f}")
    print(f"  gamma_safe = {gamma_safe:.4f}")
    print(f"  gamma_theory = {gamma_theory:.4f}")
    
    initial_loss = (1.0 / (2 * n)) * torch.sum(b_scaled**2).item()
    print(f"  Initial Loss: {initial_loss:.4f}")

    # 4. Learning Rate Sweep
    # gamma = gamma_safe * 2^k
    k_values = range(-6, 2) # -6, -5, ..., 0, 1
    multipliers = [2**k for k in k_values]
    
    sweep_results = []
    
    for mult in multipliers:
        gamma = gamma_safe * mult
        lr = gamma / n
        
        print(f"\n  Testing gamma = {gamma:.4f} (mult={mult:.4f})")
        
        run_statuses = []
        final_losses = []
        
        steps = n * num_epochs
        
        for run in range(num_runs):
            model = LeastSquaresSGD(A_scaled, b_scaled, learning_rate=lr, batch_size=1)
            loss_hist = model.train(steps)
            
            stable = is_stable(loss_hist, initial_loss)
            final_loss = loss_hist[-1] if np.isfinite(loss_hist[-1]) else float('inf')
            
            run_statuses.append(stable)
            final_losses.append(final_loss)
            
            status_str = "Stable" if stable else "DIVERGED"
            print(f"    Run {run+1}: {status_str} (Loss: {final_loss:.6f})")
            
        # Aggregate results for this gamma
        num_stable = sum(run_statuses)
        avg_final_loss = np.mean([fl for fl in final_losses if fl != float('inf')]) if any(np.isfinite(final_losses)) else float('inf')
        
        sweep_results.append({
            'gamma': gamma,
            'multiplier': mult,
            'stable_runs': num_stable,
            'total_runs': num_runs,
            'avg_final_loss': avg_final_loss,
            'relative_loss': avg_final_loss / initial_loss if avg_final_loss != float('inf') else float('inf')
        })

    return {
        'r': r,
        'spectral_info': {
            'lambda_max': max_eig_scaled,
            'mean_lambda': mean_eig_scaled,
            'gamma_safe': gamma_safe,
            'gamma_theory': gamma_theory
        },
        'sweep_results': sweep_results
    }

# Main Execution
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

aspect_ratios = [0.5, 1.0, 1.5] # Covering under, square, over regimes
all_results = {}

for r in aspect_ratios:
    all_results[r] = run_criticality_experiment(X_mnist, r, n, num_epochs, num_runs)

# Save raw results
np.save(os.path.join(results_dir, 'part3_results.npy'), all_results)

# Generate Plots
print("\nGenerating Stability Plots...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, r in enumerate(aspect_ratios):
    ax = axes[idx]
    res = all_results[r]
    data = res['sweep_results']
    
    gammas = [d['gamma'] for d in data]
    multipliers = [d['multiplier'] for d in data]
    rel_losses = [d['relative_loss'] for d in data]
    
    gamma_safe = res['spectral_info']['gamma_safe']
    gamma_theory = res['spectral_info']['gamma_theory']
    
    # Handle infinite losses for plotting
    # Cap strictly for visualization
    plot_rel_losses = [min(l, 100.0) for l in rel_losses]
    
    # Plot stability curve
    ax.plot(multipliers, plot_rel_losses, 'bo-', linewidth=2, label='Final Loss')
    
    # Mark gamma_safe (x=1)
    ax.axvline(1.0, color='g', linestyle='--', label=r'$\gamma_{safe}$')

    # Mark gamma_theory
    ratio_theory = gamma_theory / gamma_safe
    ax.axvline(ratio_theory, color='r', linestyle='--', label=r'$\gamma_{theory}$')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\gamma / \gamma_{safe}$')
    ax.set_ylabel('Final Loss / Initial Loss')
    ax.set_title(f'Stability Analysis (r={r})')
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend()

    # Text annotation
    lambda_ratio = res['spectral_info']['lambda_max'] / res['spectral_info']['mean_lambda']
    ax.text(0.05, 0.05, rf'$\lambda_{{max}}/\overline{{\lambda}} = {lambda_ratio:.1f}$',
            transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plot_path = os.path.join(results_dir, 'stability_analysis.png')
plt.savefig(plot_path, dpi=150)
print(f"Plot saved to {plot_path}")

# Interpretation Hook (Markdown Report)
report_path = os.path.join(results_dir, 'PART3_REPORT.md')
with open(report_path, 'w') as f:
    f.write("# Part 3: Step-Size Criticality Analysis\n\n")
    f.write("## Experiment Summary\n")
    f.write(f"- Dataset: MNIST (n={n})\n")
    f.write(f"- Features: Random Features (ReLU)\n")
    f.write(f"- Definition of Critical Step Size: Largest $\\gamma$ where loss remains finite and $\\le 10 \\times$ initial loss.\n\n")
    
    f.write("## Results by Regime\n\n")
    
    for r in aspect_ratios:
        res = all_results[r]
        info = res['spectral_info']
        f.write(f"### Aspect Ratio r = {r}\n")
        f.write(f"- **Spectral Ratio** ($\\lambda_{{max}} / \\overline{{\\lambda}}$): {info['lambda_max']/info['mean_lambda']:.2f}\n")
        f.write(f"- **Theoretical Limit** ($\\gamma_{{theory}}$): {info['gamma_theory']:.4f}\n")
        f.write(f"- **Safe Limit** ($\\gamma_{{safe}}$): {info['gamma_safe']:.4f}\n")
        
        # Estimate critical gamma from data
        # Last stable gamma
        stable_gammas = [d['gamma'] for d in res['sweep_results'] if d['stable_runs'] >= num_runs/2]
        crit_gamma = max(stable_gammas) if stable_gammas else 0.0

        f.write(f"- **Empirical Critical Limit**: {crit_gamma:.4f}\n")
        if crit_gamma > 0:
            f.write(f"- **Gap**: Theory predicts step size {info['gamma_theory']/crit_gamma:.1f}x larger than possible.\n\n")
        else:
            f.write(f"- **Gap**: No stable step sizes found in sweep range.\n\n")
        
    f.write("## Interpretation\n\n")
    f.write("1. **Spectral Ratio**: The large ratio between $\\lambda_{max}$ and mean($\\lambda$) explains early divergence. Volterra theory (based on mean) overestimates stability for non-isotropic data.\n")
    f.write("2. **Theory vs Practice**: $\\gamma_{theory}$ assumes isotropic features. Real data has 'outlier' eigenvalues that dictate the true stability limit ($\\gamma_{safe} \\approx 2/\\lambda_{max}$).\n")
    f.write("3. **Mismatch**: The mismatch is not a model failure but a violation of the asymptotic assumptions (specifically, that spectrum is compact).\n")

print(f"Report saved to {report_path}")
