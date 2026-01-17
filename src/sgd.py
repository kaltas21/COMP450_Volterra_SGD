import torch
import numpy as np
from tqdm import tqdm
from src.random_features import compute_features

class LeastSquaresSGD:
    """
    Implements Stochastic Gradient Descent for the Least Squares problem.

    Problem: min_x (1/2n) * sum_i (a_i^T x - b_i)^2
    """

    def __init__(self, A: torch.Tensor, b: torch.Tensor, learning_rate: float, batch_size: int = 1):
        self.A = A
        self.b = b
        self.lr = learning_rate
        self.batch_size = batch_size
        self.n, self.d = A.shape
        self.x = torch.zeros(self.d, dtype=A.dtype, device=A.device)
        
    def train(self, steps: int) -> np.ndarray:
        losses = []
        n_factor = 1.0 / (2.0 * self.n)
        
        # Precompute XT X if small enough for faster loss calc, else do batch
        # For exact paper reproduction, we calculate full loss at each step
        
        for _ in range(steps):
            # 1. Full Loss (Expensive but required for trajectory plotting)
            # f(x) = 1/(2n) ||Ax - b||^2
            with torch.no_grad():
                residuals = self.A @ self.x - self.b
                loss = n_factor * torch.sum(residuals**2)
                losses.append(loss.item())
            
            # 2. Update
            indices = torch.randint(0, self.n, (self.batch_size,), device=self.A.device)
            a_batch = self.A[indices]
            b_batch = self.b[indices]
            
            grad = a_batch.T @ (a_batch @ self.x - b_batch) / self.batch_size
            self.x -= self.lr * grad
            
        return np.array(losses)

class StreamingSGD:
    """
    Simulates SGD on an infinite stream of data (One-pass).
    At each step, we generate batch_size new samples (a_i, b_i).
    """
    def __init__(self, d: int, learning_rate: float, data_gen_func, batch_size: int = 1, device: torch.device = None):
        self.d = d
        self.lr = learning_rate
        self.gen_func = data_gen_func  # Function that returns (A_batch, b_batch)
        self.batch_size = batch_size
        self.device = device
        # Defer x creation until we know the device
        self.x = None

    def train(self, steps: int, test_A: torch.Tensor, test_b: torch.Tensor) -> np.ndarray:
        losses = []
        n_test = test_A.shape[0]
        n_factor = 1.0 / (2.0 * n_test)

        # Initialize x on the correct device (lazy initialization)
        if self.x is None or self.x.device != test_A.device:
            self.x = torch.zeros(self.d, dtype=test_A.dtype, device=test_A.device)

        for _ in range(steps):
            # Measure performance on a fixed "test set" (the finite dataset A)
            # to make it comparable to the Volterra prediction for that specific A.
            with torch.no_grad():
                residuals = test_A @ self.x - test_b
                loss = n_factor * torch.sum(residuals**2)
                losses.append(loss.item())
            
            # Generate fresh data for update
            a_stream, b_stream = self.gen_func(self.batch_size)
            
            grad = a_stream.T @ (a_stream @ self.x - b_stream) / self.batch_size
            self.x -= self.lr * grad

        return np.array(losses)


class MultiOutputLeastSquaresSGD:
    """
    SGD for multi-output least squares: min (1/2n) ||AX - B||_F^2

    Solves k independent regression problems sharing the same feature matrix A.
    Used for classification as regression with one-hot encoded targets.
    """

    def __init__(self, A: torch.Tensor, B: torch.Tensor, learning_rate: float, batch_size: int = 1):
        """
        Args:
            A: Feature matrix (n, d)
            B: Target matrix (n, k) where k = number of outputs (e.g., 10 for MNIST)
            learning_rate: Step size (gamma / n in normalized form)
            batch_size: Mini-batch size
        """
        self.A = A
        self.B = B
        self.lr = learning_rate
        self.batch_size = batch_size
        self.n, self.d = A.shape
        self.k = B.shape[1]
        self.X = torch.zeros(self.d, self.k, dtype=A.dtype, device=A.device)

    def train(self, steps: int) -> np.ndarray:
        """
        Run SGD for specified number of steps.

        Returns:
            losses: Array of Frobenius norm losses at each step
        """
        losses = []
        n_factor = 1.0 / (2.0 * self.n)

        for _ in range(steps):
            # Full loss: (1/2n) * ||AX - B||_F^2
            with torch.no_grad():
                residuals = self.A @ self.X - self.B
                loss = n_factor * torch.sum(residuals**2)
                losses.append(loss.item())

            # SGD update with mini-batch
            indices = torch.randint(0, self.n, (self.batch_size,), device=self.A.device)
            a_batch = self.A[indices]  # (batch_size, d)
            b_batch = self.B[indices]  # (batch_size, k)

            grad = a_batch.T @ (a_batch @ self.X - b_batch) / self.batch_size  # (d, k)
            self.X -= self.lr * grad

        return np.array(losses)

    @property
    def solution(self) -> torch.Tensor:
        """Return current solution matrix X."""
        return self.X