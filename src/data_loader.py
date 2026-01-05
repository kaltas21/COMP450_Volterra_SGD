import torch
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Optional

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
