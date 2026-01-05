#!/bin/bash
# Setup script for COMP450 Volterra SGD Project (Mac/Linux)
# Usage: ./setup.sh

set -e

echo "============================================"
echo "  COMP450 Volterra SGD - Environment Setup"
echo "============================================"
echo ""

# Check Python version
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python not found. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Found Python: $($PYTHON_CMD --version)"

# Create virtual environment
echo ""
echo "[1/4] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "  -> venv already exists, skipping creation"
else
    $PYTHON_CMD -m venv venv
    echo "  -> Created venv/"
fi

# Activate virtual environment
echo ""
echo "[2/4] Activating virtual environment..."
source venv/bin/activate
echo "  -> Activated"

# Upgrade pip
echo ""
echo "[3/4] Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo ""
echo "[4/4] Installing dependencies..."
pip install -r requirements.txt --quiet
echo "  -> Installed all packages"

# Download datasets
echo ""
echo "[5/5] Downloading datasets (MNIST & CIFAR-10)..."
$PYTHON_CMD -c "
import torchvision.datasets as datasets
import os

data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)

print('  -> Downloading MNIST...')
datasets.MNIST(root=data_dir, train=True, download=True)
datasets.MNIST(root=data_dir, train=False, download=True)

print('  -> Downloading CIFAR-10...')
datasets.CIFAR10(root=data_dir, train=True, download=True)
datasets.CIFAR10(root=data_dir, train=False, download=True)

# Clean up archive files
import glob
for f in glob.glob('data/*.gz') + glob.glob('data/*.tar.gz'):
    os.remove(f)
    print(f'  -> Removed {f}')

print('  -> Datasets ready!')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run Jupyter notebooks:"
echo "  jupyter lab notebooks/"
echo ""
echo "To deactivate:"
echo "  deactivate"
echo ""
