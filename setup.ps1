# Setup script for COMP450 Volterra SGD Project (Windows)
# Usage: .\setup.ps1
# Note: You may need to run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

$ErrorActionPreference = "Stop"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  COMP450 Volterra SGD - Environment Setup" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
$pythonCmd = $null
if (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonCmd = "python"
} elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    $pythonCmd = "python3"
} else {
    Write-Host "Error: Python not found. Please install Python 3.8 or higher." -ForegroundColor Red
    exit 1
}

$pythonVersion = & $pythonCmd --version
Write-Host "Found Python: $pythonVersion" -ForegroundColor Green

# Create virtual environment
Write-Host ""
Write-Host "[1/5] Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "  -> venv already exists, skipping creation" -ForegroundColor Gray
} else {
    & $pythonCmd -m venv venv
    Write-Host "  -> Created venv/" -ForegroundColor Green
}

# Activate virtual environment
Write-Host ""
Write-Host "[2/5] Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1
Write-Host "  -> Activated" -ForegroundColor Green

# Upgrade pip
Write-Host ""
Write-Host "[3/5] Upgrading pip..." -ForegroundColor Yellow
pip install --upgrade pip --quiet

# Install dependencies
Write-Host ""
Write-Host "[4/5] Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt --quiet
Write-Host "  -> Installed all packages" -ForegroundColor Green

# Download datasets
Write-Host ""
Write-Host "[5/5] Downloading datasets (MNIST & CIFAR-10)..." -ForegroundColor Yellow

$downloadScript = @"
import torchvision.datasets as datasets
import os
import glob

data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)

print('  -> Downloading MNIST...')
datasets.MNIST(root=data_dir, train=True, download=True)
datasets.MNIST(root=data_dir, train=False, download=True)

print('  -> Downloading CIFAR-10...')
datasets.CIFAR10(root=data_dir, train=True, download=True)
datasets.CIFAR10(root=data_dir, train=False, download=True)

# Clean up archive files
for f in glob.glob('data/*.gz') + glob.glob('data/*.tar.gz'):
    os.remove(f)
    print(f'  -> Removed {f}')

print('  -> Datasets ready!')
"@

python -c $downloadScript

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Setup complete!" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To activate the environment:" -ForegroundColor White
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host ""
Write-Host "To run Jupyter notebooks:" -ForegroundColor White
Write-Host "  jupyter lab notebooks/" -ForegroundColor Gray
Write-Host ""
Write-Host "To deactivate:" -ForegroundColor White
Write-Host "  deactivate" -ForegroundColor Gray
Write-Host ""
