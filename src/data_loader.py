import torch
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Optional, Dict

def load_mnist(
    root: str = './data',
    train: bool = True,
    flatten: bool = True,
    subset_size: Optional[int] = None,
    download: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads the MNIST dataset, normalizes it, and optionally flattens/subsets it.

    Args:
        root (str): Path to the dataset root directory.
        train (bool): If True, creates dataset from training set, otherwise test set.
        flatten (bool): If True, flattens images from (28, 28) to (784,).
        subset_size (int, optional): If provided, returns a random subset of this size.
        download (bool): If true, downloads the dataset from the internet and puts it in root directory.

    Returns:
        X (torch.Tensor): Data matrix of shape (n_samples, n_features).
        y (torch.Tensor): Labels of shape (n_samples,).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = torchvision.datasets.MNIST(root=root, train=train, download=download, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    X, y = next(iter(loader))

    if flatten:
        X = X.view(X.size(0), -1)

    if subset_size is not None and subset_size < len(X):
        X = X[:subset_size]
        y = y[:subset_size]

    return X, y

def load_cifar10(
    root: str = './data',
    train: bool = True,
    flatten: bool = True,
    grayscale: bool = True,
    subset_size: Optional[int] = None,
    download: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads the CIFAR-10 dataset.

    Args:
        root (str): Path to the dataset root directory.
        train (bool): If True, creates dataset from training set.
        flatten (bool): If True, flattens images.
        grayscale (bool): If True, converts RGB images to grayscale.
        subset_size (int, optional): If provided, returns a random subset.
        download (bool): If true, downloads the dataset.

    Returns:
        X (torch.Tensor): Data matrix.
        y (torch.Tensor): Labels.
    """
    transform_list = [transforms.ToTensor()]
    if grayscale:
        transform_list.append(transforms.Grayscale(num_output_channels=1))

    transform_list.append(transforms.Normalize((0.5,), (0.5,)))
    transform = transforms.Compose(transform_list)

    dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=download, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    X, y = next(iter(loader))

    if flatten:
        X = X.view(X.size(0), -1)

    if subset_size is not None and subset_size < len(X):
        X = X[:subset_size]
        y = y[:subset_size]

    return X, y


def whiten_data(X: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, Dict]:
    """
    Whiten data so E[x] = 0 and E[xx^T] = I.

    Applies the transform: x_tilde = Sigma^{-1/2} (x - mu)

    Args:
        X: Input tensor (n_samples, d_features)
        eps: Regularization for numerical stability

    Returns:
        X_whitened: Whitened data with identity covariance
        info: Dictionary with whitening statistics
    """
    n, d = X.shape
    device = X.device

    # 1. Center: subtract mean
    mu = X.mean(dim=0, keepdim=True)
    X_centered = X - mu

    # 2. Covariance: Sigma = (1/n) * X^T X
    Sigma = (X_centered.T @ X_centered) / n

    # 3. Eigendecomposition for whitening
    eigenvalues, eigenvectors = torch.linalg.eigh(Sigma)

    # Regularize small eigenvalues for numerical stability
    eigenvalues = torch.clamp(eigenvalues, min=eps)

    # 4. Compute whitening matrix: W = V * diag(1/sqrt(eigenvalues))
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(eigenvalues))
    W_whiten = eigenvectors @ D_inv_sqrt

    # 5. Apply whitening transform
    X_whitened = X_centered @ W_whiten

    info = {
        'mean': mu,
        'eigenvalues_original': eigenvalues.cpu().numpy(),
        'whitening_matrix': W_whiten,
        'condition_number': (eigenvalues.max() / eigenvalues.min()).item()
    }

    return X_whitened, info


def labels_to_onehot(y: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    """
    Convert integer labels to one-hot encoded matrix.

    Args:
        y: Label tensor (n_samples,) with values in [0, num_classes-1]
        num_classes: Number of classes (default 10 for MNIST)

    Returns:
        Y: One-hot matrix (n_samples, num_classes)
    """
    n = y.shape[0]
    Y = torch.zeros(n, num_classes, dtype=torch.float32, device=y.device)
    Y[torch.arange(n, device=y.device), y] = 1.0
    return Y
