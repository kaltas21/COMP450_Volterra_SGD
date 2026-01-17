import numpy as np
import torch
from typing import Tuple


@torch.no_grad()
def compute_eigenvalues(A: torch.Tensor) -> np.ndarray:
    """
    GPU-friendly eigenvalues of H = (1/n) A^T A.
    Computes eigvals of the smaller matrix: (1/n) A A^T if d>n else (1/n) A^T A.
    Returns sorted eigenvalues as a NumPy array (on CPU) for downstream NumPy code.

    Args:
        A (torch.Tensor): Feature matrix of shape (n, d). Can be on CPU or GPU.

    Returns:
        eigvals (np.ndarray): Sorted eigenvalues (ascending) of H.
    """
    n, d = A.shape
    # Ensure float32 or float64 for stability
    A = A.detach()

    if d > n:
        M = (A @ A.T) / n          # shape (n, n)
    else:
        M = (A.T @ A) / n          # shape (d, d)

    # Symmetric eigvals - computed on same device as A
    eigvals = torch.linalg.eigvalsh(M)

    # Filter tiny negative numerical noise, then sort
    eigvals = torch.clamp(eigvals, min=0.0)
    eigvals, _ = torch.sort(eigvals)

    return eigvals.cpu().numpy()

def marchenko_pastur_density(x: np.ndarray, r: float, sigma_sq: float = 1.0) -> np.ndarray:
    """
    Computes the theoretical Marchenko-Pastur probability density function.

    Args:
        x (np.ndarray): Points at which to evaluate density.
        r (float): Ratio d/n.
        sigma_sq (float): Variance of entries (usually 1.0).

    Returns:
        pdf (np.ndarray): Density values.
    """
    lambda_plus = sigma_sq * (1 + np.sqrt(r))**2
    lambda_minus = sigma_sq * (1 - np.sqrt(r))**2

    pdf = np.zeros_like(x)
    mask = (x >= lambda_minus) & (x <= lambda_plus)

    if np.any(mask):
        root_term = np.sqrt((lambda_plus - x[mask]) * (x[mask] - lambda_minus))
        pdf[mask] = root_term / (2 * np.pi * sigma_sq * r * x[mask])

    return pdf

@torch.no_grad()
def compute_h_kernels(
    eigenvalues: np.ndarray,
    gamma: float,
    t_points,
    r: float,
    device: torch.device = None,
    return_torch: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    GPU-accelerated computation of kernels h_0(t), h_1(t), h_2(t).

    h_k(t) = Integral[ x^k * exp(-2 * gamma * t * x) dmu(x) ]

    Args:
        eigenvalues (np.ndarray): Computed eigenvalues (from n x n or d x d matrix).
        gamma (float): Step size.
        t_points: Time points to evaluate (np.ndarray or torch.Tensor).
        r (float): ratio d/n.
        device (torch.device): Device for computation. If None, uses CUDA if available.
        return_torch (bool): If True, returns torch tensors on device instead of numpy.

    Returns:
        h0, h1, h2: Values at t_points (torch.Tensor if return_torch=True, else np.ndarray).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert eigenvalues to torch tensor on device
    ev = torch.tensor(eigenvalues, dtype=torch.float64, device=device).unsqueeze(1)  # (M, 1)

    # Handle t_points - can be numpy array or torch tensor
    if isinstance(t_points, torch.Tensor):
        t = t_points.to(dtype=torch.float64, device=device).unsqueeze(0)  # (1, T)
    else:
        t = torch.tensor(t_points, dtype=torch.float64, device=device).unsqueeze(0)  # (1, T)

    # Compute exp(-2 * gamma * ev * t) - shape (M, T)
    exp_term = torch.exp(-2 * gamma * ev * t)

    # Sum over eigenvalues
    sum_h0 = torch.sum(exp_term, dim=0)           # (T,)
    sum_h1 = torch.sum(ev * exp_term, dim=0)      # (T,)
    sum_h2 = torch.sum(ev**2 * exp_term, dim=0)   # (T,)

    M = len(eigenvalues)

    # Normalization logic for spectral measure
    if r <= 1.0:
        d_est = M
        h0 = sum_h0 / d_est
        h1 = sum_h1 / d_est
        h2 = sum_h2 / d_est
    else:
        n_est = M
        d_est = int(r * n_est)
        n_zeros = d_est - n_est

        h0 = (sum_h0 + n_zeros) / d_est
        h1 = sum_h1 / d_est
        h2 = sum_h2 / d_est

    if return_torch:
        return h0, h1, h2
    else:
        return h0.cpu().numpy(), h1.cpu().numpy(), h2.cpu().numpy()
