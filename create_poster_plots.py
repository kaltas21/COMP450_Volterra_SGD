"""
Create Publication-Quality Plots for Poster
Clean, readable fonts with proper LaTeX math rendering
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
import torch

from src.sgd import LeastSquaresSGD, StreamingSGD
from src.volterra import VolterraSolver
from src.spectral import compute_eigenvalues
from src.random_features import generate_random_weights, compute_features

torch.manual_seed(42)
np.random.seed(42)

# =============================================================================
# CLEAN STYLE CONFIGURATION
# =============================================================================
plt.rcParams.update({
    # Font settings - clean and readable
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'mathtext.fontset': 'cm',  # Computer Modern for math

    # Axes
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'axes.axisbelow': True,

    # Ticks
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',

    # Legend
    'legend.fontsize': 10,
    'legend.framealpha': 0.95,
    'legend.edgecolor': '0.8',

    # Lines
    'lines.linewidth': 2.0,
    'lines.markersize': 6,

    # Grid
    'grid.alpha': 0.4,
    'grid.linewidth': 0.6,

    # Figure
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color palette - high contrast, colorblind-friendly
COLORS = {
    'volterra': '#2ca02c',    # Green
    'sgd': '#d62728',         # Red
    'streaming': '#1f77b4',   # Blue
    'sme': '#9467bd',         # Purple
    'sde': '#ff7f0e',         # Orange
}

output_dir = './' \
'_figures'
os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("GENERATING POSTER PLOTS")
print("="*70)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def run_sde_simulation(A, b, x0, gamma, steps, sigma_val=0.1):
    n, d = A.shape
    x = x0.clone()
    losses = []
    eta = gamma / n
    for _ in range(steps):
        residuals = A @ x - b
        full_grad = (A.T @ residuals) / n
        loss = (1.0 / (2 * n)) * torch.sum(residuals**2).item()
        losses.append(loss)
        noise = torch.randn_like(x) * sigma_val
        x = x - eta * full_grad - eta * noise
    return np.array(losses)

def run_sme_simulation(A, b, x0, gamma, steps):
    n, d = A.shape
    x = x0.clone()
    losses = []
    eta = gamma / n
    for _ in range(steps):
        residuals = A @ x - b
        full_grad = (A.T @ residuals) / n
        loss = (1.0 / (2 * n)) * torch.sum(residuals**2).item()
        losses.append(loss)
        grads_sq_mean = ((A**2) * (residuals.unsqueeze(1)**2)).mean(dim=0)
        grad_mean_sq = full_grad**2
        grad_var = torch.clamp(grads_sq_mean - grad_mean_sq, min=1e-12)
        grad_std = torch.sqrt(grad_var)
        noise = torch.randn_like(x) * grad_std
        x = x - eta * full_grad - eta * noise
    return np.array(losses)

# =============================================================================
# FIGURE 1: PAPER REPLICATION (One-Hidden Layer, gamma = gamma_max/8)
# =============================================================================
print("\n[1/6] Creating Paper Replication Plot...")

n_rep, r_rep = 1000, 1.2
d_rep = int(n_rep * r_rep)
num_epochs_rep, steps_rep = 50, n_rep * 50
num_runs_rep = 3

m = d_rep
X_raw = torch.randn(n_rep, m)
W_fixed = generate_random_weights(m, d_rep)
A = compute_features(X_raw, W_fixed, activation='shifted_relu')

raw_x = torch.randn(d_rep)
x_star = raw_x / torch.norm(raw_x)
b = A @ x_star
x0 = torch.zeros(d_rep)

eigvals = compute_eigenvalues(A)
mean_eig = np.sum(eigvals) / d_rep
gamma_max = (2.0 / r_rep) * (1.0 / mean_eig)
gamma = gamma_max / 8
R_val = 1.0

print(f"  gamma_max = {gamma_max:.4f}, gamma = {gamma:.4f}")

# Run methods
print("  Running all methods...")
solver = VolterraSolver(eigvals, gamma, r_rep, R=R_val, R_tilde=0.0)
psi, t_theory = solver.solve(t_max=num_epochs_rep, dt=0.05)

sgd_runs = []
for _ in range(num_runs_rep):
    model = LeastSquaresSGD(A, b, learning_rate=gamma/n_rep, batch_size=1)
    sgd_runs.append(model.train(steps_rep))
sgd_mean, sgd_std = np.mean(sgd_runs, axis=0), np.std(sgd_runs, axis=0)

def stream_gen(bs):
    X_s = torch.randn(bs, m)
    A_s = compute_features(X_s, W_fixed, activation='shifted_relu')
    return A_s, (A_s @ x_star).squeeze()

streamer = StreamingSGD(d_rep, gamma/n_rep, stream_gen)
streaming_loss = streamer.train(steps_rep, A, b)
sme_loss = run_sme_simulation(A, b, x0, gamma, steps_rep)
sde_loss = run_sde_simulation(A, b, x0, gamma, steps_rep, sigma_val=0.1)

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
t_steps = np.linspace(0, num_epochs_rep, len(sgd_mean))
marker_step = len(t_steps) // 6

ax.plot(t_steps, sme_loss, color=COLORS['sme'], lw=1.8, alpha=0.7, label='SME')
ax.plot(t_steps[::marker_step], sme_loss[::marker_step], 'o', color=COLORS['sme'], ms=5, mew=0)

ax.plot(t_steps, sde_loss, color=COLORS['sde'], lw=1.8, alpha=0.7, label='SDE')
ax.plot(t_steps[::marker_step], sde_loss[::marker_step], 'D', color=COLORS['sde'], ms=5, mew=0)

ax.plot(t_steps, streaming_loss, color=COLORS['streaming'], lw=2, label='Streaming')
ax.plot(t_steps[::marker_step], streaming_loss[::marker_step], '<', color=COLORS['streaming'], ms=6, mew=0)

ax.plot(t_steps, sgd_mean, color=COLORS['sgd'], lw=2, label='Empirical SGD')
ax.fill_between(t_steps, sgd_mean - sgd_std, sgd_mean + sgd_std, color=COLORS['sgd'], alpha=0.15)
ax.plot(t_steps[::marker_step], sgd_mean[::marker_step], 's', color=COLORS['sgd'], ms=5, mew=0)

ax.plot(t_theory, psi, color=COLORS['volterra'], lw=2.5, label='Volterra')
ax.plot(t_theory[::60], psi[::60], '^', color=COLORS['volterra'], ms=6, mew=0)

ax.set_xlabel('Epochs')
ax.set_ylabel('Training Loss')
ax.set_yscale('log')
ax.set_ylim(1e-4, 1.0)
ax.set_title(r'One-Hidden Layer, $\gamma = \gamma_{\mathrm{max}}/8$')
ax.legend(loc='upper right', ncol=2, frameon=True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig1_replication.pdf'))
plt.savefig(os.path.join(output_dir, 'fig1_replication.png'))
plt.close()
print(f"  Saved: fig1_replication")

# =============================================================================
# LOAD EXPERIMENT RESULTS
# =============================================================================
print("\n[2/6] Loading experiment results...")
base_results = np.load('./results/full_run_base/all_results.npy', allow_pickle=True).item()
nonlinear_results = np.load('./results/full_run_nonlinear/all_results.npy', allow_pickle=True).item()
whitened_results = np.load('./results/quick_whitened/all_results.npy', allow_pickle=True).item()

aspect_ratios = [0.5, 1.0, 1.5]
n_exp = 3000
n_whitened = 2000

# =============================================================================
# FIGURE 2: BASE MNIST (Loss Curves + Correlation)
# =============================================================================
print("\n[3/6] Creating Base MNIST plots...")

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
plt.subplots_adjust(hspace=0.25, wspace=0.25)

colors_lr = ['#d62728', '#1f77b4', '#2ca02c']
lr_labels = [r'$\gamma = 0.25\gamma_{\mathrm{safe}}$',
             r'$\gamma = 0.50\gamma_{\mathrm{safe}}$',
             r'$\gamma = 0.90\gamma_{\mathrm{safe}}$']

# Row 1: Loss curves
for idx, r in enumerate(aspect_ratios):
    ax = axes[0, idx]
    res = base_results[r]

    for i, (key, data) in enumerate(res['results_safe'].items()):
        t_sgd = np.arange(len(data['sgd_mean'])) / n_exp
        ax.plot(t_sgd, data['sgd_mean'], color=colors_lr[i], lw=2, label='SGD' if idx == 0 and i == 0 else None)
        ax.fill_between(t_sgd, data['sgd_mean'] - data['sgd_std'],
                       data['sgd_mean'] + data['sgd_std'], color=colors_lr[i], alpha=0.12)
        ax.plot(data['t_theory'], data['psi'], '--', color=colors_lr[i], lw=1.8,
                label='Volterra' if idx == 0 and i == 0 else None)

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Training Loss' if idx == 0 else '')
    ax.set_yscale('log')
    ax.set_title(f'$r = {r}$')

# Add legend to first subplot
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='gray', lw=2, label='SGD (solid)'),
    Line2D([0], [0], color='gray', lw=2, linestyle='--', label='Volterra (dashed)')
]
axes[0, 0].legend(handles=legend_elements, loc='upper right', fontsize=9)

# Row 2: Correlation scatter
correlations_base = []
for idx, r in enumerate(aspect_ratios):
    ax = axes[1, idx]
    res = base_results[r]

    all_volterra, all_sgd = [], []
    for key, data in res['results_safe'].items():
        t_sgd = np.arange(len(data['sgd_mean'])) / n_exp
        volterra_interp = np.interp(t_sgd, data['t_theory'], data['psi'])
        step = max(1, len(t_sgd) // 100)
        all_volterra.extend(volterra_interp[::step])
        all_sgd.extend(data['sgd_mean'][::step])

    all_volterra, all_sgd = np.array(all_volterra), np.array(all_sgd)
    valid = (all_volterra > 0) & (all_sgd > 0) & np.isfinite(all_volterra) & np.isfinite(all_sgd)
    all_volterra, all_sgd = all_volterra[valid], all_sgd[valid]

    ax.scatter(all_volterra, all_sgd, alpha=0.5, s=20, c=COLORS['streaming'], edgecolors='none')

    min_val, max_val = min(all_volterra.min(), all_sgd.min()), max(all_volterra.max(), all_sgd.max())
    ax.plot([min_val, max_val], [min_val, max_val], '--', color=COLORS['sgd'], lw=2, label='$y = x$')

    corr, _ = stats.pearsonr(all_volterra, all_sgd)
    correlations_base.append(corr)

    ax.set_xlabel('Volterra Predicted Loss')
    ax.set_ylabel('Actual SGD Loss' if idx == 0 else '')
    ax.set_title(f'$R = {corr:.3f}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='lower right', fontsize=9)

fig.suptitle(r'Base MNIST: Linear Target $\mathbf{b} = A\mathbf{x}^*$', fontsize=15, y=0.98)
plt.savefig(os.path.join(output_dir, 'fig2_base_mnist.pdf'))
plt.savefig(os.path.join(output_dir, 'fig2_base_mnist.png'))
plt.close()
print(f"  Saved: fig2_base_mnist, Correlations: {[f'{c:.4f}' for c in correlations_base]}")

# =============================================================================
# FIGURE 3: NONLINEAR MNIST (Loss Curves + Correlation)
# =============================================================================
print("\n[4/6] Creating Nonlinear MNIST plots...")

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
plt.subplots_adjust(hspace=0.25, wspace=0.25)

# Row 1: Loss curves
for idx, r in enumerate(aspect_ratios):
    ax = axes[0, idx]
    res = nonlinear_results[r]

    for i, (key, data) in enumerate(res['results_safe'].items()):
        t_sgd = np.arange(len(data['sgd_mean'])) / n_exp
        ax.plot(t_sgd, data['sgd_mean'], color=colors_lr[i], lw=2)
        ax.fill_between(t_sgd, data['sgd_mean'] - data['sgd_std'],
                       data['sgd_mean'] + data['sgd_std'], color=colors_lr[i], alpha=0.12)
        ax.plot(data['t_theory'], data['psi'], '--', color=colors_lr[i], lw=1.8)

    final_loss = res['results_safe']['0.90']['sgd_mean'][-1]
    ax.axhline(y=final_loss, color='gray', linestyle=':', lw=1.5, alpha=0.7, label='Loss floor')

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Training Loss' if idx == 0 else '')
    ax.set_yscale('log')
    zf = res['zero_fraction']
    ax.set_title(f'$r = {r}$ (Zero: {zf:.0%})')

legend_elements = [
    Line2D([0], [0], color='gray', lw=2, label='SGD (solid)'),
    Line2D([0], [0], color='gray', lw=2, linestyle='--', label='Volterra (dashed)'),
    Line2D([0], [0], color='gray', lw=1.5, linestyle=':', label='Loss floor')
]
axes[0, 2].legend(handles=legend_elements, loc='upper right', fontsize=9)

# Row 2: Correlation scatter
correlations_nonlinear = []
for idx, r in enumerate(aspect_ratios):
    ax = axes[1, idx]
    res = nonlinear_results[r]

    all_volterra, all_sgd = [], []
    for key, data in res['results_safe'].items():
        t_sgd = np.arange(len(data['sgd_mean'])) / n_exp
        volterra_interp = np.interp(t_sgd, data['t_theory'], data['psi'])
        step = max(1, len(t_sgd) // 100)
        all_volterra.extend(volterra_interp[::step])
        all_sgd.extend(data['sgd_mean'][::step])

    all_volterra, all_sgd = np.array(all_volterra), np.array(all_sgd)
    valid = (all_volterra > 0) & (all_sgd > 0) & np.isfinite(all_volterra) & np.isfinite(all_sgd)
    all_volterra, all_sgd = all_volterra[valid], all_sgd[valid]

    ax.scatter(all_volterra, all_sgd, alpha=0.5, s=20, c=COLORS['streaming'], edgecolors='none')
    min_val, max_val = min(all_volterra.min(), all_sgd.min()), max(all_volterra.max(), all_sgd.max())
    ax.plot([min_val, max_val], [min_val, max_val], '--', color=COLORS['sgd'], lw=2, label='$y = x$')

    corr, _ = stats.pearsonr(all_volterra, all_sgd)
    correlations_nonlinear.append(corr)

    ax.set_xlabel('Volterra Predicted Loss')
    ax.set_ylabel('Actual SGD Loss' if idx == 0 else '')
    ax.set_title(f'$R = {corr:.3f}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='lower right', fontsize=9)

fig.suptitle(r'Nonlinear MNIST: $\mathbf{b} = \mathrm{ReLU}(A\mathbf{x}^*)$', fontsize=15, y=0.98)
plt.savefig(os.path.join(output_dir, 'fig3_nonlinear_mnist.pdf'))
plt.savefig(os.path.join(output_dir, 'fig3_nonlinear_mnist.png'))
plt.close()
print(f"  Saved: fig3_nonlinear_mnist, Correlations: {[f'{c:.4f}' for c in correlations_nonlinear]}")

# =============================================================================
# FIGURE 4: WHITENED MNIST (Loss Curves + Correlation)
# =============================================================================
print("\n[5/6] Creating Whitened MNIST plots...")

aspect_ratios_w = list(whitened_results.keys())

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
plt.subplots_adjust(hspace=0.25, wspace=0.25)

# Row 1: Loss curves
for idx, r in enumerate(aspect_ratios_w):
    ax = axes[0, idx]
    res = whitened_results[r]

    for i, (key, data) in enumerate(res['results_safe'].items()):
        t_sgd = np.arange(len(data['sgd_mean'])) / n_whitened
        ax.plot(t_sgd, data['sgd_mean'], color=colors_lr[i], lw=2)
        ax.fill_between(t_sgd, data['sgd_mean'] - data['sgd_std'],
                       data['sgd_mean'] + data['sgd_std'], color=colors_lr[i], alpha=0.12)
        ax.plot(data['t_theory'], data['psi'], '--', color=colors_lr[i], lw=1.8)

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Training Loss' if idx == 0 else '')
    ax.set_yscale('log')
    sr = res.get('spectral_ratio', res.get('spectral_info', {}).get('spectral_ratio', 0))
    ax.set_title(f'$r = {r}$ (SR: {sr:.0f})')

legend_elements = [
    Line2D([0], [0], color='gray', lw=2, label='SGD (solid)'),
    Line2D([0], [0], color='gray', lw=2, linestyle='--', label='Volterra (dashed)')
]
axes[0, 2].legend(handles=legend_elements, loc='upper right', fontsize=9)

# Row 2: Correlation scatter
correlations_whitened = []
for idx, r in enumerate(aspect_ratios_w):
    ax = axes[1, idx]
    res = whitened_results[r]

    all_volterra, all_sgd = [], []
    for key, data in res['results_safe'].items():
        t_sgd = np.arange(len(data['sgd_mean'])) / n_whitened
        volterra_interp = np.interp(t_sgd, data['t_theory'], data['psi'])
        step = max(1, len(t_sgd) // 100)
        all_volterra.extend(volterra_interp[::step])
        all_sgd.extend(data['sgd_mean'][::step])

    all_volterra, all_sgd = np.array(all_volterra), np.array(all_sgd)
    valid = (all_volterra > 0) & (all_sgd > 0) & np.isfinite(all_volterra) & np.isfinite(all_sgd)
    all_volterra, all_sgd = all_volterra[valid], all_sgd[valid]

    ax.scatter(all_volterra, all_sgd, alpha=0.5, s=20, c=COLORS['streaming'], edgecolors='none')
    min_val, max_val = min(all_volterra.min(), all_sgd.min()), max(all_volterra.max(), all_sgd.max())
    ax.plot([min_val, max_val], [min_val, max_val], '--', color=COLORS['sgd'], lw=2, label='$y = x$')

    corr, _ = stats.pearsonr(all_volterra, all_sgd)
    correlations_whitened.append(corr)

    ax.set_xlabel('Volterra Predicted Loss')
    ax.set_ylabel('Actual SGD Loss' if idx == 0 else '')
    ax.set_title(f'$R = {corr:.3f}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='lower right', fontsize=9)

fig.suptitle(r'Whitened MNIST: $\mathbb{E}[\mathbf{x}]=0$, $\mathbb{E}[\mathbf{x}\mathbf{x}^\top]=I$', fontsize=15, y=0.98)
plt.savefig(os.path.join(output_dir, 'fig4_whitened_mnist.pdf'))
plt.savefig(os.path.join(output_dir, 'fig4_whitened_mnist.png'))
plt.close()
print(f"  Saved: fig4_whitened_mnist, Correlations: {[f'{c:.4f}' for c in correlations_whitened]}")

# =============================================================================
# FIGURE 5: CRITICAL STEP SIZE ANALYSIS
# =============================================================================
print("\n[6/6] Creating Critical Step Size Analysis plot...")

part3 = np.load('./results/part3_results.npy', allow_pickle=True).item()

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
plt.subplots_adjust(wspace=0.25)

for idx, r in enumerate(aspect_ratios):
    ax = axes[idx]
    res = part3[r]
    data = res['sweep_results']

    multipliers = [d['multiplier'] for d in data]
    rel_losses = [min(d['relative_loss'], 100.0) for d in data]

    gamma_safe = res['spectral_info']['gamma_safe']
    gamma_theory = res['spectral_info']['gamma_theory']

    ax.plot(multipliers, rel_losses, 'o-', color=COLORS['sgd'], lw=2, ms=7, label='Final/Initial Loss')
    ax.axvline(1.0, color=COLORS['streaming'], linestyle='--', lw=2, label=r'$\gamma_{\mathrm{safe}}$')

    ratio_theory = gamma_theory / gamma_safe
    ax.axvline(ratio_theory, color=COLORS['volterra'], linestyle='--', lw=2, label=r'$\gamma_{\mathrm{theory}}$')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\gamma / \gamma_{\mathrm{safe}}$')
    ax.set_ylabel('Final Loss / Initial Loss' if idx == 0 else '')

    lambda_ratio = res['spectral_info']['lambda_max'] / res['spectral_info']['mean_lambda']
    ax.set_title(f'$r = {r}$ ($\\lambda_{{\\max}}/\\bar{{\\lambda}} = {lambda_ratio:.0f}$)')

    if idx == 2:
        ax.legend(loc='lower left', fontsize=10)

fig.suptitle('Step-Size Criticality Analysis', fontsize=15, y=0.98)
plt.savefig(os.path.join(output_dir, 'fig5_criticality.pdf'))
plt.savefig(os.path.join(output_dir, 'fig5_criticality.png'))
plt.close()
print(f"  Saved: fig5_criticality")

# =============================================================================
# FIGURE 6: SPECTRAL RATIO SUMMARY
# =============================================================================
print("\n[BONUS] Creating Spectral Ratio Summary...")

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(3)
width = 0.25

sr_base = [base_results[r]['spectral_info']['spectral_ratio'] for r in aspect_ratios]
sr_nonlinear = [nonlinear_results[r]['spectral_info']['spectral_ratio'] for r in aspect_ratios]
sr_whitened = [whitened_results[r].get('spectral_ratio', whitened_results[r].get('spectral_info', {}).get('spectral_ratio', 0)) for r in aspect_ratios_w]

bars1 = ax.bar(x - width, sr_base, width, label='Base', color=COLORS['sgd'], edgecolor='black', lw=0.8)
bars2 = ax.bar(x, sr_nonlinear, width, label='Nonlinear', color=COLORS['streaming'], edgecolor='black', lw=0.8)
bars3 = ax.bar(x + width, sr_whitened, width, label='Whitened', color=COLORS['volterra'], edgecolor='black', lw=0.8)

ax.set_xlabel('Aspect Ratio $r$')
ax.set_ylabel(r'Spectral Ratio $\lambda_{\max}/\bar{\lambda}$')
ax.set_title('Spectral Ratio Across Experiments')
ax.set_xticks(x)
ax.set_xticklabels([f'{r}' for r in aspect_ratios])
ax.legend(loc='upper right')

for bars, vals in [(bars1, sr_base), (bars2, sr_nonlinear), (bars3, sr_whitened)]:
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f'{val:.0f}',
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig6_spectral_summary.pdf'))
plt.savefig(os.path.join(output_dir, 'fig6_spectral_summary.png'))
plt.close()
print(f"  Saved: fig6_spectral_summary")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("ALL PLOTS COMPLETE")
print("="*70)
print(f"\nSaved to: {output_dir}/")
print("\nCorrelation Summary:")
print(f"  Base:      {[f'{c:.4f}' for c in correlations_base]}")
print(f"  Nonlinear: {[f'{c:.4f}' for c in correlations_nonlinear]}")
print(f"  Whitened:  {[f'{c:.4f}' for c in correlations_whitened]}")
