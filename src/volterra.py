import numpy as np
from src.spectral import compute_h_kernels

class VolterraSolver:
    """
    Numerical solver for the Volterra Integral Equation describing SGD dynamics.

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
        R_tilde: float = 0.0
    ):
        """
        Args:
            eigenvalues (np.ndarray): Eigenvalues of H.
            gamma (float): Learning rate.
            r (float): Ratio d/n.
            R (float): Initial signal norm squared (||x0 - x*||^2).
            R_tilde (float): Noise variance (sigma_eta^2).
        """
        self.eigenvalues = eigenvalues
        self.gamma = gamma
        self.r = r
        self.R = R
        self.R_tilde = R_tilde

    def solve(self, t_max: float, dt: float) -> np.ndarray:
        """
        Solves the equation using a discretization loop (Forward Euler / Left Riemann Sum).

        Args:
            t_max (float): Maximum time to simulate.
            dt (float): Time step size.

        Returns:
            psi (np.ndarray): Predicted loss values at steps 0, dt, ..., t_max.
        """
        n_steps = int(t_max / dt) + 1
        t_points = np.linspace(0, t_max, n_steps)

        # Precompute kernels for convolution
        h0, h1, h2 = compute_h_kernels(self.eigenvalues, self.gamma, t_points, self.r)

        # Forcing function z(t)
        z = (self.R / 2.0) * h1 + (self.R_tilde / 2.0) * (self.r * h0 + (1 - self.r))

        psi = np.zeros(n_steps)
        coeff = self.r * (self.gamma**2) * dt

        for i in range(n_steps):
            if i == 0:
                conv_sum = 0.0
            else:
                # Convolution Integral_0^{t_i} h2(t_i - s) psi(s) ds
                # Discretized: sum_{j=0}^{i-1} h2[i-j] * psi[j]
                # Slicing h2 corresponds to h2(dt), h2(2dt), ..., h2(idt)
                current_h2 = h2[1:i+1][::-1]
                current_psi = psi[:i]
                conv_sum = np.dot(current_h2, current_psi)

            psi[i] = z[i] + coeff * conv_sum

        return psi, t_points
