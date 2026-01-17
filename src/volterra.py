import numpy as np
import torch
from src.spectral import compute_h_kernels


class VolterraSolver:
    """
    GPU-accelerated solver for the Volterra Integral Equation describing SGD dynamics.

    Equation (Proposal Eq 2 / Paper Eq 12):
    psi(t) = z(t) + r * gamma^2 * Integral_0^t h_2(t-s) psi(s) ds

    where z(t) = (R/2)*h_1(t) + (R_tilde/2)*(r*h_0(t) + (1-r))
    """

    def __init__(
        self,
        eigenvalues: np.ndarray,
        gamma: float,
        r: float,
        R: float = 1.0,
        R_tilde: float = 0.0,
        device: torch.device = None
    ):
        """
        Args:
            eigenvalues (np.ndarray): Eigenvalues of H.
            gamma (float): Learning rate.
            r (float): Ratio d/n.
            R (float): Initial signal norm squared (||x0 - x*||^2).
            R_tilde (float): Noise variance (sigma_eta^2).
            device (torch.device): Device for computation. If None, auto-detects.
        """
        self.eigenvalues = eigenvalues
        self.gamma = gamma
        self.r = r
        self.R = R
        self.R_tilde = R_tilde
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def solve(self, t_max: float, dt: float) -> np.ndarray:
        """
        Solves the equation using fully GPU-accelerated computation.

        Args:
            t_max (float): Maximum time to simulate.
            dt (float): Time step size.

        Returns:
            psi (np.ndarray): Predicted loss values at steps 0, dt, ..., t_max.
            t_points (np.ndarray): Time points.
        """
        n_steps = int(t_max / dt) + 1

        # Use torch.linspace on GPU
        t_points_tensor = torch.linspace(0, t_max, n_steps, dtype=torch.float64, device=self.device)

        # Precompute kernels on GPU - pass torch tensor directly to avoid numpy conversion
        h0_t, h1_t, h2_t = compute_h_kernels(
            self.eigenvalues, self.gamma, t_points_tensor, self.r, device=self.device, return_torch=True
        )

        # Convert t_points to numpy only for return value (after kernel computation)
        t_points = t_points_tensor.cpu().numpy()

        # Forcing function z(t) - vectorized on GPU
        z = (self.R / 2.0) * h1_t + (self.R_tilde / 2.0) * (self.r * h0_t + (1 - self.r))

        # Initialize psi on GPU
        psi = torch.zeros(n_steps, dtype=torch.float64, device=self.device)
        coeff = self.r * (self.gamma**2) * dt

        # Sequential solve - keep everything on GPU (no .item() calls)
        # The loop is inherently sequential due to recurrence relation
        psi[0] = z[0]
        for i in range(1, n_steps):
            # Convolution using GPU tensor operations - no CPU sync
            current_h2 = h2_t[1:i+1].flip(0)
            current_psi = psi[:i]
            conv_sum = torch.dot(current_h2, current_psi)
            psi[i] = z[i] + coeff * conv_sum

        # Single GPU->CPU transfer at the end
        return psi.cpu().numpy(), t_points
