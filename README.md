# Testing the Volterra Model for SGD on Real Datasets

**COMP 450 Project** - Kaan Altas & Yigit Karasu

This project empirically tests the Volterra Integral Equation model from the paper *"SGD in the Large: Average-case Analysis, Asymptotics, and Stepsize Criticality"* (Paquette et al., 2021) on real datasets.

## Overview

The paper introduces a mathematical framework using Volterra integral equations to predict SGD training loss curves using only the eigenvalue distribution of the data matrix. This project:

1. Implements a numerical Volterra solver
2. Validates the theory on synthetic Gaussian data
3. Tests the model on real datasets (MNIST, CIFAR-10) using Random Features
4. Analyzes the critical step size phenomenon

## Quick Start

### Mac/Linux
```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### Windows (PowerShell)
```powershell
# If needed: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\setup.ps1
.\venv\Scripts\Activate.ps1
```

## Project Structure

```
.
├── data/                   # Datasets (auto-downloaded)
│   ├── MNIST/
│   └── cifar-10-batches-py/
├── docs/                   # Project documentation & papers
├── notebooks/              # Jupyter notebooks for experiments
│   └── a.ipynb            # Figure 4 reproduction
├── src/                    # Source code
│   ├── data_loader.py     # Dataset loading utilities
│   ├── random_features.py # Random features model
│   ├── sgd.py             # SGD implementations
│   ├── spectral.py        # Eigenvalue computation
│   ├── visualization.py   # Plotting utilities
│   └── volterra.py        # Volterra equation solver
├── requirements.txt        # Python dependencies
├── setup.sh               # Setup script (Mac/Linux)
└── setup.ps1              # Setup script (Windows)
```

## Key Equations

### Least Squares Problem
$$\min_{x \in \mathbb{R}^d} f(x) = \frac{1}{2n} \|Ax - b\|^2$$

### Volterra Integral Equation
The expected loss converges to $\psi_0(t)$ satisfying:
$$\psi_0(t) = z(t) + r\gamma^2 \int_0^t h_2(t-s) \psi_0(s) \, ds$$

### Critical Step Size
$$\gamma_{\max} = \frac{2}{r} \cdot \frac{1}{\bar{\lambda}}$$

where $r = d/n$ is the aspect ratio and $\bar{\lambda}$ is the mean eigenvalue.

## Experiments

### Figure 4 Reproduction (`notebooks/a.ipynb`)

Compares empirical SGD against:
- **Volterra**: Deterministic prediction from eigenvalues
- **Streaming SGD**: Infinite data regime (one-pass)
- **SME**: Stochastic Modified Equation
- **SDE**: Stochastic Differential Equation with isotropic noise

Parameters: $n=1000$, $d=1200$, $r=1.2$

## Requirements

- Python 3.8+
- PyTorch
- NumPy, SciPy, Matplotlib
- Jupyter (for notebooks)

See `requirements.txt` for full list.

## References

- Paquette, C., Lee, K., Pedregosa, F., & Paquette, E. (2021). *SGD in the Large: Average-case Analysis, Asymptotics, and Stepsize Criticality*. [arXiv:2102.04396](https://arxiv.org/abs/2102.04396)

## License

MIT License - See LICENSE file for details.
