"""
Figure 4 Reproduction: Comparison of SGD Models

Reproduces Figure 4 from the paper:
"SGD in the Large: Average-case Analysis, Asymptotics, and Stepsize Criticality" (Paquette et al., 2021)

Problem Setting:
    min_{x in R^d} f(x) = (1/2n) sum_i (a_i^T x - b_i)^2 = (1/2n) ||Ax - b||^2

We compare Empirical SGD against:
    - Volterra: Deterministic prediction from eigenvalues
    - Streaming SGD: One-pass SGD with fresh data (infinite data limit)
    - SME: Stochastic Modified Equation with true gradient covariance
    - SDE: Stochastic Differential Equation with isotropic noise
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from src.sgd import LeastSquaresSGD, StreamingSGD
from src.volterra import VolterraSolver
from src.spectral import compute_eigenvalues
from src.random_features import generate_random_weights, compute_features, generate_isotropic_matrix

# Global Settings
N_SAMPLES = 1000
ASPECT_RATIO = 1.2
D_FEATURES = int(N_SAMPLES * ASPECT_RATIO)
NUM_EPOCHS = 50
STEPS = N_SAMPLES * NUM_EPOCHS
NUM_RUNS = 5


def run_sde_simulation(A: torch.Tensor, b: torch.Tensor, x0: torch.Tensor,
                       gamma: float, steps: int, sigma_val: float = 0.1) -> np.ndarray:
    """
    Simulates the SDE model with isotropic noise.
    Paper uses Sigma = 0.01 * I, implying std = sqrt(0.01) = 0.1.
    """
    n = A.shape[0]
    x = x0.clone()
    losses = []
    eta = gamma / n
    noise_std = sigma_val

    for _ in range(steps):
        residuals = A @ x - b
        full_grad = (A.T @ residuals) / n
        loss = (1.0 / (2 * n)) * torch.sum(residuals**2).item()
        losses.append(loss)
        noise = torch.randn_like(x) * noise_std
        x = x - eta * full_grad - eta * noise

    return np.array(losses)


def run_sme_simulation(A: torch.Tensor, b: torch.Tensor, x0: torch.Tensor,
                       gamma: float, steps: int) -> np.ndarray:
    """
    Simulates the SME (Diagonal Approximation).
    Noise scales with the empirical variance of the gradients.
    """
    n = A.shape[0]
    x = x0.clone()
    losses = []
    eta = gamma / n

    for _ in range(steps):
        residuals = A @ x - b
        full_grad = (A.T @ residuals) / n
        loss = (1.0 / (2 * n)) * torch.sum(residuals**2).item()
        losses.append(loss)

        # Diagonal SME Variance: Var(g) = E[g^2] - (E[g])^2
        grads_sq_mean = ((A**2) * (residuals.unsqueeze(1)**2)).mean(dim=0)
        grad_mean_sq = full_grad**2
        grad_var = grads_sq_mean - grad_mean_sq
        grad_var = torch.clamp(grad_var, min=1e-12)
        grad_std = torch.sqrt(grad_var)

        noise = torch.randn_like(x) * grad_std
        x = x - eta * full_grad - eta * noise

    return np.array(losses)


def get_data(mode: str, n: int, d: int, device: torch.device):
    """
    Generates Data (A, b) with NORMALIZED target x_star (||x_star||^2 = 1).

    Args:
        mode: 'isotropic' or 'one_hidden'
        n: number of samples
        d: number of features
        device: torch device

    Returns:
        A, b, x0, x_star, W_fixed
    """
    if mode == 'isotropic':
        A = generate_isotropic_matrix(n, d, device=device)
        W = None
    elif mode == 'one_hidden':
        m = d
        X_raw = torch.randn(n, m, device=device)
        W = generate_random_weights(m, d, device=device)
        A = compute_features(X_raw, W, activation='shifted_relu')
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Target x* - normalize to have norm 1 (R=1)
    raw_x = torch.randn(d, device=device)
    x_star = raw_x / torch.norm(raw_x)
    b = A @ x_star
    x0 = torch.zeros(d, device=device)

    return A, b, x0, x_star, W


def run_experiment(data_mode: str, device: torch.device) -> dict:
    """
    Runs the full experiment for a given data mode.

    Args:
        data_mode: 'isotropic' or 'one_hidden'
        device: torch device

    Returns:
        Dictionary of results for each step size
    """
    # Prepare Data (already on device)
    A, b, x0, x_star, W_fixed = get_data(data_mode, N_SAMPLES, D_FEATURES, device)

    # Spectral Analysis
    print(f"[{data_mode}] Computing Eigenvalues...")
    eigvals = compute_eigenvalues(A)

    # Gamma Max Calculation (Thm 1.2): gamma_max = 2/r * (mean(eigenvalues))^-1
    tr_H = np.sum(eigvals)
    mean_eig = tr_H / D_FEATURES
    gamma_max = (2.0 / ASPECT_RATIO) * (1.0 / mean_eig)

    print(f"[{data_mode}] Mean Eval: {mean_eig:.4f}, Gamma Max: {gamma_max:.4f}")

    results = {}
    multipliers = [1/8, 1/4, 1/2, 1/1.2]
    R_val = (torch.norm(x0 - x_star)**2).item()

    for mult in multipliers:
        gamma = gamma_max * mult
        key = f"{mult}"
        print(f"  Running gamma = {gamma:.4f} ({mult:.4f}x)")

        # Volterra Theory
        solver = VolterraSolver(eigvals, gamma, ASPECT_RATIO, R=R_val, R_tilde=0.0)
        psi, t_theory = solver.solve(t_max=NUM_EPOCHS, dt=0.05)

        # Empirical SGD with progress bar
        sgd_runs = []
        for run_idx in tqdm(range(NUM_RUNS), desc=f"    SGD Runs ({mult:.3f}x)", leave=False):
            model = LeastSquaresSGD(A, b, learning_rate=gamma/N_SAMPLES, batch_size=1)
            loss_hist = model.train(STEPS, show_progress=True, desc=f"      Run {run_idx+1}")
            sgd_runs.append(loss_hist)
        sgd_mean = np.mean(sgd_runs, axis=0)
        sgd_std = np.std(sgd_runs, axis=0)

        # Streaming SGD
        def stream_gen(bs):
            if data_mode == 'isotropic':
                A_s = generate_isotropic_matrix(bs, D_FEATURES, device=device)
            else:
                m = D_FEATURES
                X_s = torch.randn(bs, m, device=device)
                A_s = compute_features(X_s, W_fixed, activation='shifted_relu')
            b_s = (A_s @ x_star).squeeze()
            return A_s, b_s

        streamer = StreamingSGD(D_FEATURES, gamma/N_SAMPLES, stream_gen)
        streaming_loss = streamer.train(STEPS, A, b, show_progress=True, desc=f"    Streaming ({mult:.3f}x)")

        # SME & SDE
        sme_loss = run_sme_simulation(A, b, x0, gamma, STEPS)
        sde_loss = run_sde_simulation(A, b, x0, gamma, STEPS, sigma_val=0.1)

        results[key] = {
            't_theory': t_theory,
            'psi': psi,
            'sgd_mean': sgd_mean,
            'sgd_std': sgd_std,
            'streaming': streaming_loss,
            'sme': sme_loss,
            'sde': sde_loss
        }

    return results


def plot_figure_4(results_iso: dict, results_ohl: dict, save_path: str = None):
    """
    Plots Figure 4: comparison of all models.

    Args:
        results_iso: Results from isotropic features experiment
        results_ohl: Results from one-hidden layer experiment
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 4, figsize=(22, 11), sharex=True, sharey=True)

    rows_data = [('isotropic features', results_iso), ('one-hidden layer', results_ohl)]
    mult_keys = ['0.125', '0.25', '0.5', str(1/1.2)]
    mult_labels = [8, 4, 2, 1.2]

    for r_idx, (name, res) in enumerate(rows_data):
        for c_idx, k in enumerate(mult_keys):
            ax = axes[r_idx, c_idx]
            if k not in res:
                continue
            data = res[k]

            t_steps = np.linspace(0, NUM_EPOCHS, len(data['sgd_mean']))

            # SME (Teal)
            ax.plot(t_steps, data['sme'], color='teal', lw=1.5, alpha=0.5, label='SME')

            # Volterra (Green)
            ax.plot(data['t_theory'], data['psi'], color='#77DD77', lw=3, label='Volterra', alpha=1.0)
            ax.plot(data['t_theory'][::80], data['psi'][::80], '^', color='#77DD77', ms=8, markeredgecolor='none')

            # Streaming (Blue)
            ax.plot(t_steps, data['streaming'], color='#8DA0CB', lw=2, alpha=0.8, label='Streaming')
            ax.plot(t_steps[::5000], data['streaming'][::5000], '<', color='#8DA0CB', ms=8)

            # SDE (Pink)
            ax.plot(t_steps, data['sde'], color='#F781BF', lw=2, alpha=0.7, label='SDE')
            ax.plot(t_steps[::5000], data['sde'][::5000], 'd', color='#F781BF', ms=8)

            # Empirical SGD (Orange)
            ax.plot(t_steps, data['sgd_mean'], color='#FC8D62', lw=2, label='Empirical SGD')
            ax.fill_between(t_steps,
                            data['sgd_mean'] - data['sgd_std'],
                            data['sgd_mean'] + data['sgd_std'],
                            color='#FC8D62', alpha=0.3)
            ax.plot(t_steps[::5000], data['sgd_mean'][::5000], 's', color='#FC8D62', ms=8)

            ax.set_yscale('log')
            ax.set_ylim(1e-4, 1.2)
            ax.grid(True, which='both', alpha=0.2)

            title_str = f"{name}, $\\gamma=\\gamma_{{\\max}}/{mult_labels[c_idx]}$"
            ax.set_title(title_str, fontsize=12)
            if r_idx == 1:
                ax.set_xlabel('epochs', fontsize=12)

    # Legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ordered_labels = ['SME', 'SDE', 'Empirical SGD', 'Streaming', 'Volterra']
    ordered_handles = [by_label[l] for l in ordered_labels if l in by_label]

    fig.legend(ordered_handles, ordered_labels, loc='lower center', ncol=5,
               bbox_to_anchor=(0.5, 0.02), fontsize=12)
    plt.subplots_adjust(bottom=0.12, wspace=0.1, hspace=0.2)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def main():
    """Main entry point for the experiment."""
    import os
    torch.manual_seed(42)
    np.random.seed(42)

    # Output directories
    plots_dir = './plots/figure4'
    results_dir = './results/figure4'
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Simulation: n={N_SAMPLES}, d={D_FEATURES} (r={ASPECT_RATIO}), "
          f"Epochs={NUM_EPOCHS}, Device={device}")
    print(f"Plots directory: {plots_dir}")
    print(f"Results directory: {results_dir}")

    print("\nRunning Isotropic Features Experiments...")
    results_iso = run_experiment('isotropic', device)

    print("\nRunning One-Hidden Layer Experiments...")
    results_ohl = run_experiment('one_hidden', device)

    # Save results
    np.save(os.path.join(results_dir, 'results_isotropic.npy'), results_iso)
    np.save(os.path.join(results_dir, 'results_one_hidden.npy'), results_ohl)
    print(f"\nResults saved to {results_dir}/")

    # Plot and save figure
    plot_figure_4(results_iso, results_ohl, save_path=os.path.join(plots_dir, 'figure4.png'))
    print(f"Figure saved to {plots_dir}/figure4.png")


if __name__ == '__main__':
    main()
