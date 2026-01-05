import torch
import math

def generate_random_weights(d_in: int, d_out: int, distribution: str = 'gaussian') -> torch.Tensor:
    """
    Generates a fixed random weight matrix W for feature projection.

    Args:
        d_in (int): Input dimension (raw features).
        d_out (int): Output dimension (random features).
        distribution (str): 'gaussian' or 'orthogonal'.

    Returns:
        W (torch.Tensor): Weight matrix of shape (d_in, d_out).
    """
    if distribution == 'gaussian':
        W = torch.randn(d_in, d_out)
    else:
        raise NotImplementedError(f"Distribution {distribution} not implemented.")
    return W

def shifted_relu(z: torch.Tensor) -> torch.Tensor:
    """
    Implements the shifted ReLU activation function used in the paper
    to ensure zero Gaussian mean (Appendix A.2.2, Eq 27, 141).

    g(z) = ReLU(z) - 1/sqrt(2*pi)
    """
    return torch.nn.functional.relu(z) - (1.0 / math.sqrt(2 * math.pi))

def compute_features(
    X: torch.Tensor,
    W: torch.Tensor,
    activation: str = 'shifted_relu',
    scale_by_sqrt_d: bool = True
) -> torch.Tensor:
    """
    Computes the Random Feature matrix A = sigma(X W).

    According to the paper (Eq 25), A_ij = g( [WY]_ij / sqrt(m) ).
    Here X corresponds to W^T in their notation if we view samples as rows.

    Proposal Eq (3): A = sigma(X W).

    Args:
        X (torch.Tensor): Input data (n_samples, d_in).
        W (torch.Tensor): Random weights (d_in, d_out).
        activation (str): 'relu', 'shifted_relu', or 'linear'.
        scale_by_sqrt_d (bool): If True, divides argument by sqrt(d_in).

    Returns:
        A (torch.Tensor): Feature matrix (n_samples, d_out).
    """
    Z = torch.matmul(X, W)

    if scale_by_sqrt_d:
        Z = Z / math.sqrt(X.shape[1])

    if activation == 'relu':
        A = torch.nn.functional.relu(Z)
    elif activation == 'shifted_relu':
        A = shifted_relu(Z)
    elif activation == 'linear':
        A = Z
    else:
        raise ValueError(f"Unknown activation: {activation}")

    return A

def generate_isotropic_matrix(n: int, d: int) -> torch.Tensor:
    """
    Generates a purely synthetic Gaussian matrix A for baseline verification.
    Entries are i.i.d N(0, 1).

    Args:
        n (int): Number of samples.
        d (int): Number of features.

    Returns:
        A (torch.Tensor): Matrix of shape (n, d).
    """
    return torch.randn(n, d)
