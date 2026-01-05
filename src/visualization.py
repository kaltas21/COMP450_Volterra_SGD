import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_theory_vs_empirical(
    t_theory: np.ndarray,
    loss_theory: np.ndarray,
    loss_empirical: np.ndarray,
    title: str = "Volterra vs SGD",
    log_scale: bool = True
):
    """
    Overlays the theoretical Volterra prediction and the empirical SGD run.
    """
    plt.figure(figsize=(10, 6))

    plt.plot(t_theory, loss_theory, label='Volterra Prediction (Theory)', linewidth=2, linestyle='--')

    # Scale empirical x-axis to match theory t_max (assuming steps mapped to continuous time t)
    t_empirical = np.linspace(0, t_theory[-1], len(loss_empirical))
    plt.plot(t_empirical, loss_empirical, label='SGD (Empirical)', alpha=0.6)

    if log_scale:
        plt.yscale('log')

    plt.xlabel('Time (t)')
    plt.ylabel('Training Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()

def plot_eigenvalue_distribution(
    eigenvalues: np.ndarray,
    r: float,
    title: str = "Spectral Density"
):
    """
    Plots histogram of eigenvalues vs Marchenko-Pastur theoretical density.
    """
    plt.figure(figsize=(10, 6))

    sns.histplot(eigenvalues, bins=50, stat="density", label='Empirical ESD', alpha=0.5)

    from src.spectral import marchenko_pastur_density
    x_grid = np.linspace(0, np.max(eigenvalues)*1.1, 1000)
    pdf = marchenko_pastur_density(x_grid, r=r)

    plt.plot(x_grid, pdf, 'r-', label='Marchenko-Pastur Law', linewidth=2)

    plt.xlabel('Eigenvalue $\lambda$')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_convergence_heatmap(
    learning_rates: list,
    ratios: list,
    convergence_speeds: np.ndarray
):
    """
    Heatmap for step size criticality analysis (Proposal Visual 4).
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        convergence_speeds,
        xticklabels=[f"{lr:.2f}" for lr in learning_rates],
        yticklabels=[f"{r:.2f}" for r in ratios],
        cmap="viridis",
        annot=True,
        fmt=".2f"
    )
    plt.xlabel('Learning Rate $\gamma$')
    plt.ylabel('Ratio $r = d/n$')
    plt.title('Convergence Rate Heatmap')
    plt.show()
