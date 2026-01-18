"""
Part 3: Step-size Criticality Analysis

Empirically maps when SGD becomes unstable as learning rate increases.
Tests the gap between theoretical critical step size (gamma_theory)
and the safe step size (gamma_safe = 2/lambda_max).

Targets are the mean-centered MNIST digit labels (no planted x*), to keep the
criticality analysis tied to the actual dataset.

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

from tqdm import tqdm
from src.data_loader import load_mnist
from src.random_features import generate_random_weights, compute_features
from src.spectral import compute_eigenvalues
from src.sgd import LeastSquaresSGD

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


def prepare_label_targets(y: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Use MNIST digit labels as regression targets (mean-centered).
    """
    b = y.to(device=device, dtype=torch.float32)
    return b - b.mean()


def is_stable(loss_hist: np.ndarray, initial_loss: float) -> bool:
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


def run_criticality_experiment(X: torch.Tensor, y: torch.Tensor, r: float, n: int,
                               num_epochs: int, num_runs: int, device: torch.device):
    """Run step-size criticality analysis for a specific aspect ratio."""
    d = int(n * r)
    print(f"\n=== Running Criticality Analysis for r={r} (d={d}) ===")

    # Build Features
    W = generate_random_weights(X.shape[1], d, device=device)
    A = compute_features(X, W, activation='shifted_relu', scale_by_sqrt_d=True)
    A = A - A.mean(dim=0, keepdim=True)

    # Label-based target (mean-centered digits)
    b = prepare_label_targets(y, device)

    # Eigenvalue Analysis
    print("Computing eigenvalues...")
    eigvals = compute_eigenvalues(A)
    mean_eig = np.mean(eigvals)
    max_eig = np.max(eigvals)

    # Scale A so mean(lambda) = 1
    scale_factor = 1.0 / np.sqrt(mean_eig)
    A_scaled = A * scale_factor

    # Recompute spectral properties after scaling
    eigvals_scaled = compute_eigenvalues(A_scaled)
    mean_eig_scaled = np.mean(eigvals_scaled)
    max_eig_scaled = np.max(eigvals_scaled)

    gamma_safe = 2.0 / max_eig_scaled
    gamma_theory = (2.0 / r) / mean_eig_scaled

    print(f"  mean(lambda) = {mean_eig_scaled:.4f}")
    print(f"  lambda_max = {max_eig_scaled:.4f}")
    print(f"  gamma_safe = {gamma_safe:.4f}")
    print(f"  gamma_theory = {gamma_theory:.4f}")

    initial_loss = (1.0 / (2 * n)) * torch.sum(b**2).item()
    print(f"  Initial Loss: {initial_loss:.4f}")

    # Learning Rate Sweep: gamma = gamma_safe * 2^k
    k_values = range(-6, 2)  # -6, -5, ..., 0, 1
    multipliers = [2**k for k in k_values]

    sweep_results = []

    for mult in multipliers:
        gamma = gamma_safe * mult
        lr = gamma / n

        print(f"\n  Testing gamma = {gamma:.4f} (mult={mult:.4f})")

        run_statuses = []
        final_losses = []

        steps = n * num_epochs

        for run in tqdm(range(num_runs), desc=f"    Runs (mult={mult:.3f})", leave=False):
            model = LeastSquaresSGD(A_scaled, b, learning_rate=lr, batch_size=1)
            loss_hist = model.train(steps, show_progress=True, desc=f"      SGD Run {run+1}")

            stable = is_stable(loss_hist, initial_loss)
            final_loss = loss_hist[-1] if np.isfinite(loss_hist[-1]) else float('inf')

            run_statuses.append(stable)
            final_losses.append(final_loss)

            status_str = "Stable" if stable else "DIVERGED"
            tqdm.write(f"    Run {run+1}: {status_str} (Loss: {final_loss:.6f})")

        # Aggregate results for this gamma
        num_stable = sum(run_statuses)
        avg_final_loss = np.mean([fl for fl in final_losses if fl != float('inf')]) \
            if any(np.isfinite(final_losses)) else float('inf')

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
        'initial_loss': initial_loss,
        'sweep_results': sweep_results
    }


def plot_stability_analysis(all_results: dict, output_dir: str):
    """Generate stability analysis plots."""
    print("\nGenerating Stability Plots...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, r in enumerate(ASPECT_RATIOS):
        ax = axes[idx]
        res = all_results[r]
        data = res['sweep_results']

        gammas = [d['gamma'] for d in data]
        multipliers = [d['multiplier'] for d in data]
        rel_losses = [d['relative_loss'] for d in data]

        gamma_safe = res['spectral_info']['gamma_safe']
        gamma_theory = res['spectral_info']['gamma_theory']

        # Cap infinite losses for visualization
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
        ax.set_xlabel(r'$\gamma / \gamma_{safe}$', fontsize=12)
        ax.set_ylabel('Final Loss / Initial Loss', fontsize=12)
        ax.set_title(f'Stability Analysis (r={r})', fontsize=12)
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend()

        # Text annotation
        lambda_ratio = res['spectral_info']['lambda_max'] / res['spectral_info']['mean_lambda']
        ax.text(0.05, 0.05, rf'$\lambda_{{max}}/\overline{{\lambda}} = {lambda_ratio:.1f}$',
                transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'stability_analysis.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {plot_path}")


def plot_summary_heatmap(all_results: dict, output_dir: str):
    """Generate summary heatmap of stability regions."""
    print("Generating Summary Heatmap...")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Collect data for heatmap
    r_values = ASPECT_RATIOS
    k_values = range(-6, 2)
    multipliers = [2**k for k in k_values]

    stability_matrix = np.zeros((len(r_values), len(multipliers)))

    for i, r in enumerate(r_values):
        for j, mult in enumerate(multipliers):
            sweep = all_results[r]['sweep_results']
            matching = [s for s in sweep if abs(s['multiplier'] - mult) < 1e-6]
            if matching:
                stability_matrix[i, j] = matching[0]['stable_runs'] / matching[0]['total_runs']

    im = ax.imshow(stability_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(multipliers)))
    ax.set_xticklabels([f'{m:.3f}' for m in multipliers], rotation=45)
    ax.set_yticks(range(len(r_values)))
    ax.set_yticklabels([f'r={r}' for r in r_values])

    ax.set_xlabel('Multiplier (gamma / gamma_safe)', fontsize=12)
    ax.set_ylabel('Aspect Ratio', fontsize=12)
    ax.set_title('Stability Fraction (Green = All Stable, Red = All Diverged)', fontsize=12)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Fraction of Stable Runs', fontsize=10)

    # Add text annotations
    for i in range(len(r_values)):
        for j in range(len(multipliers)):
            text = ax.text(j, i, f'{stability_matrix[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stability_heatmap.png'), dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/stability_heatmap.png")


def plot_critical_gamma_comparison(all_results: dict, output_dir: str):
    """Generate comparison of critical step sizes."""
    print("Generating Critical Gamma Comparison...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Theory vs Safe vs Empirical critical gamma
    ax = axes[0]
    x = np.arange(len(ASPECT_RATIOS))
    width = 0.25

    theory_gamma = [all_results[r]['spectral_info']['gamma_theory'] for r in ASPECT_RATIOS]
    safe_gamma = [all_results[r]['spectral_info']['gamma_safe'] for r in ASPECT_RATIOS]

    # Estimate empirical critical gamma (last stable)
    empirical_gamma = []
    for r in ASPECT_RATIOS:
        stable_gammas = [d['gamma'] for d in all_results[r]['sweep_results']
                        if d['stable_runs'] >= NUM_RUNS / 2]
        empirical_gamma.append(max(stable_gammas) if stable_gammas else 0.0)

    ax.bar(x - width, theory_gamma, width, label='gamma_theory', color='coral')
    ax.bar(x, safe_gamma, width, label='gamma_safe', color='seagreen')
    ax.bar(x + width, empirical_gamma, width, label='gamma_empirical', color='steelblue')

    ax.set_xticks(x)
    ax.set_xticklabels([f'r={r}' for r in ASPECT_RATIOS])
    ax.set_ylabel('Critical Step Size', fontsize=12)
    ax.set_title('Critical Step Size Comparison', fontsize=12)
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Gap ratio
    ax = axes[1]
    gap_theory_safe = [t/s for t, s in zip(theory_gamma, safe_gamma)]
    gap_theory_emp = [t/e if e > 0 else float('inf') for t, e in zip(theory_gamma, empirical_gamma)]

    ax.bar(x - width/2, gap_theory_safe, width, label='theory/safe', color='purple')
    ax.bar(x + width/2, gap_theory_emp, width, label='theory/empirical', color='orange')

    ax.set_xticks(x)
    ax.set_xticklabels([f'r={r}' for r in ASPECT_RATIOS])
    ax.set_ylabel('Gap Ratio', fontsize=12)
    ax.set_title('Theory Overestimation Factor', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for i, (gs, ge) in enumerate(zip(gap_theory_safe, gap_theory_emp)):
        ax.text(i - width/2, gs + 5, f'{gs:.0f}x', ha='center', va='bottom', fontsize=9)
        if ge < 1000:
            ax.text(i + width/2, ge + 5, f'{ge:.0f}x', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Step-Size Criticality: Theory vs Practice', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'critical_gamma_comparison.png'), dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/critical_gamma_comparison.png")


def generate_report(all_results: dict, output_dir: str):
    """Generate markdown report."""
    print("Generating Report...")

    report_path = os.path.join(output_dir, 'PART3_REPORT.md')
    with open(report_path, 'w') as f:
        f.write("# Part 3: Step-Size Criticality Analysis\n\n")
        f.write("## Experiment Summary\n")
        f.write(f"- Dataset: MNIST (n={N_SAMPLES})\n")
        f.write(f"- Features: Random Features (shifted ReLU)\n")
        f.write(f"- Runs per step size: {NUM_RUNS}\n")
        f.write(f"- Definition of Critical Step Size: Largest gamma where loss remains finite and <= 10x initial loss.\n\n")

        f.write("## Results by Regime\n\n")

        for r in ASPECT_RATIOS:
            res = all_results[r]
            info = res['spectral_info']
            f.write(f"### Aspect Ratio r = {r}\n")
            f.write(f"- **Spectral Ratio** (lambda_max / mean_lambda): {info['lambda_max']/info['mean_lambda']:.2f}\n")
            f.write(f"- **Theoretical Limit** (gamma_theory): {info['gamma_theory']:.4f}\n")
            f.write(f"- **Safe Limit** (gamma_safe): {info['gamma_safe']:.4f}\n")

            # Estimate critical gamma from data
            stable_gammas = [d['gamma'] for d in res['sweep_results']
                           if d['stable_runs'] >= NUM_RUNS / 2]
            crit_gamma = max(stable_gammas) if stable_gammas else 0.0

            f.write(f"- **Empirical Critical Limit**: {crit_gamma:.4f}\n")
            if crit_gamma > 0:
                f.write(f"- **Gap**: Theory predicts step size {info['gamma_theory']/crit_gamma:.1f}x larger than possible.\n\n")
            else:
                f.write(f"- **Gap**: No stable step sizes found in sweep range.\n\n")

        f.write("## Interpretation\n\n")
        f.write("1. **Spectral Ratio**: The large ratio between lambda_max and mean(lambda) explains early divergence. ")
        f.write("Volterra theory (based on mean) overestimates stability for non-isotropic data.\n\n")
        f.write("2. **Theory vs Practice**: gamma_theory assumes isotropic features. Real data has 'outlier' eigenvalues ")
        f.write("that dictate the true stability limit (gamma_safe ~ 2/lambda_max).\n\n")
        f.write("3. **Mismatch**: The mismatch is not a model failure but a violation of the asymptotic assumptions ")
        f.write("(specifically, that spectrum is compact).\n")

    print(f"  Saved: {report_path}")


def main():
    """Main entry point for the experiment."""
    device = get_device()

    # Output directories - separate plots and results
    plots_dir = './plots/part3_criticality'
    results_dir = './results/part3_criticality'
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("PART 3: STEP-SIZE CRITICALITY ANALYSIS")
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
    print(f"X_mnist shape: {X_mnist.shape}")

    # Main experiment loop
    start_time = time.time()
    all_results = {}

    for r in ASPECT_RATIOS:
        all_results[r] = run_criticality_experiment(
            X_mnist, y_mnist, r, N_SAMPLES, NUM_EPOCHS, NUM_RUNS, device
        )

    total_time = time.time() - start_time

    # Save raw results
    np.save(os.path.join(results_dir, 'all_results.npy'), all_results)
    print(f"\nResults saved to {results_dir}/all_results.npy")

    # Generate plots
    plot_stability_analysis(all_results, plots_dir)
    plot_summary_heatmap(all_results, plots_dir)
    plot_critical_gamma_comparison(all_results, plots_dir)

    # Generate report
    generate_report(all_results, results_dir)

    # Summary
    print("\n" + "="*60)
    print("PART 3 CRITICALITY ANALYSIS COMPLETE")
    print("="*60)

    print(f"\nTotal runtime: {total_time/60:.2f} minutes")
    print(f"\nResults summary:")
    for r in ASPECT_RATIOS:
        res = all_results[r]
        info = res['spectral_info']
        stable_gammas = [d['gamma'] for d in res['sweep_results']
                        if d['stable_runs'] >= NUM_RUNS / 2]
        crit_gamma = max(stable_gammas) if stable_gammas else 0.0

        print(f"\n  r = {r}:")
        print(f"    Spectral ratio: {info['lambda_max']/info['mean_lambda']:.2f}")
        print(f"    gamma_theory: {info['gamma_theory']:.4f}")
        print(f"    gamma_safe: {info['gamma_safe']:.4f}")
        print(f"    gamma_empirical: {crit_gamma:.4f}")
        if crit_gamma > 0:
            print(f"    Theory overestimation: {info['gamma_theory']/crit_gamma:.1f}x")

    print(f"\nFigures saved to {plots_dir}/")


if __name__ == '__main__':
    main()
