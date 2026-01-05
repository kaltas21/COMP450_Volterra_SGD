import numpy as np
import torch
from typing import Tuple

def compute_eigenvalues(A: torch.Tensor) -> np.ndarray:
    """
    Computes the eigenvalues of the Hessian H = (1/n) * A^T * A.

    Optimization:
    If n < d, the non-zero eigenvalues of A^T A are the same as A A^T.
    We compute the eigenvalues of the smaller matrix for efficiency.

    Args:
        A (torch.Tensor): Feature matrix of shape (n, d).

    Returns:
        eigvals (np.ndarray): Sorted eigenvalues (ascending) of H.
    """
    n, d = A.shape
    A_np = A.detach().cpu().numpy()

    # Compute smaller Gram matrix to save memory/time
    if d > n:
        M = (A_np @ A_np.T) / n
    else:
        M = (A_np.T @ A_np) / n

    eigvals = np.linalg.eigvalsh(M)

    # Filter numerical noise
    eigvals = np.maximum(eigvals, 0.0)

    return np.sort(eigvals)

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

def compute_h_kernels(
    eigenvalues: np.ndarray,
    gamma: float,
    t_points: np.ndarray,
    r: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the kernels h_0(t), h_1(t), h_2(t) defined in Eq (2) / Eq (12).

    h_k(t) = Integral[ x^k * exp(-2 * gamma * t * x) dmu(x) ]

    Args:
        eigenvalues (np.ndarray): Computed eigenvalues (from n x n or d x d matrix).
        gamma (float): Step size.
        t_points (np.ndarray): Time points to evaluate.
        r (float): ratio d/n.

    Returns:
        h0, h1, h2 (np.ndarray): Values at t_points.
    """
    ev = eigenvalues[:, None]
    t = t_points[None, :]
    exp_term = np.exp(-2 * gamma * ev * t)

    # Sum over the computed non-zero eigenvalues
    sum_h0 = np.sum(exp_term, axis=0)
    sum_h1 = np.sum(ev * exp_term, axis=0)
    sum_h2 = np.sum(ev**2 * exp_term, axis=0)

    M = len(eigenvalues) # Assuming these are the M = min(n, d) eigenvalues

    # Normalization logic for spectral measure dmu(x) which sums to 1 over d dimensions
    if r <= 1.0:
        # d <= n. All eigenvalues are present. Normalize by d.
        d_est = M
        h0 = sum_h0 / d_est
        h1 = sum_h1 / d_est
        h2 = sum_h2 / d_est
    else:
        # r > 1 => d > n. We have n non-zero evals, plus (d-n) zeros.
        # h1 and h2 are unaffected by zeros (0^k = 0 for k>=1).
        # h0 includes contribution from zeros (0^0 = 1).
        n_est = M
        d_est = int(r * n_est)
        n_zeros = d_est - n_est

        h0 = (sum_h0 + n_zeros) / d_est
        h1 = sum_h1 / d_est
        h2 = sum_h2 / d_est

    return h0, h1, h2
