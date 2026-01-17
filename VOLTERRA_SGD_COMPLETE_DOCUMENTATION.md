# Volterra Model for SGD: Complete Technical Documentation

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Theoretical Background](#2-theoretical-background)
3. [Mathematical Framework](#3-mathematical-framework)
4. [Key Quantities and Formulas](#4-key-quantities-and-formulas)
5. [Spectral Analysis](#5-spectral-analysis)
6. [Aspect Ratio Analysis](#6-aspect-ratio-analysis)
7. [Critical Step Size Theory](#7-critical-step-size-theory)
8. [Implementation Details](#8-implementation-details)
9. [Experiments](#9-experiments)
10. [Key Findings](#10-key-findings)

---

## 1. Project Overview

### 1.1 Motivation

This project empirically tests the **Volterra Integral Equation** model from the paper *"SGD in the Large: Average-case Analysis, Asymptotics, and Stepsize Criticality"* (Paquette et al., 2021). The paper provides a deterministic formula that predicts how the training loss of Stochastic Gradient Descent (SGD) evolves over time, using only the **eigenvalues** of the data matrix.

### 1.2 Core Question

> Can a theoretical formula derived for idealized "isotropic" (random Gaussian) data accurately predict SGD behavior on real-world datasets like MNIST?

### 1.3 Key Contributions

1. **Volterra Solver Implementation**: GPU-accelerated numerical solver for the Volterra integral equation
2. **Random Features Model**: Testing on MNIST using random feature projections
3. **Whitening Analysis**: Testing whether data whitening improves theory-practice alignment
4. **Non-linear Target Analysis**: Testing theory limits with non-realizable problems
5. **Critical Step Size Analysis**: Empirical validation of theoretical stability bounds

---

## 2. Theoretical Background

### 2.1 The Least Squares Problem

The project focuses on the fundamental **Least Squares regression** problem:

$$\min_{x \in \mathbb{R}^d} f(x) = \frac{1}{2n} \sum_{i=1}^{n} (a_i^T x - b_i)^2 = \frac{1}{2n} \|Ax - b\|^2$$

Where:
- $n$ = number of samples (rows of $A$)
- $d$ = number of features (columns of $A$)
- $A \in \mathbb{R}^{n \times d}$ = feature/design matrix
- $b \in \mathbb{R}^{n}$ = target vector
- $x \in \mathbb{R}^{d}$ = parameter vector to optimize

### 2.2 SGD Update Rule

Standard mini-batch SGD updates the parameter vector as:

$$x_{t+1} = x_t - \eta \cdot g_t$$

Where:
- $\eta = \gamma / n$ = effective learning rate (normalized by $n$)
- $\gamma$ = step size parameter (the key quantity in the theory)
- $g_t$ = stochastic gradient computed from a random mini-batch

For batch size 1 (single sample $i$):
$$g_t = a_i (a_i^T x_t - b_i)$$

### 2.3 The Central Insight

The paper proves that as $n, d \to \infty$ with $r = d/n$ fixed, the **random** loss trajectory $f(x_t)$ converges to a **deterministic** function $\psi_0(t)$ that depends only on:
1. The eigenvalues of $H = \frac{1}{n} A^T A$
2. The step size $\gamma$
3. The aspect ratio $r = d/n$
4. Initial conditions

---

## 3. Mathematical Framework

### 3.1 The Volterra Integral Equation

The deterministic loss prediction $\psi_0(t)$ satisfies:

$$\boxed{\psi_0(t) = z(t) + r\gamma^2 \int_0^t h_2(t-s) \psi_0(s) \, ds}$$

This is a **Volterra integral equation of the second kind** with convolution kernel.

### 3.2 The Forcing Function z(t)

$$z(t) = \frac{R}{2} h_1(t) + \frac{\tilde{R}}{2} \left( r \cdot h_0(t) + (1-r) \right)$$

Where:
- $R = \|x_0 - x^*\|^2$ = initial distance to optimum (squared)
- $\tilde{R} = \sigma_\eta^2$ = noise variance (label noise)
- For planted targets with no noise: $R = 1, \tilde{R} = 0$

### 3.3 The Kernel Functions h_k(t)

The kernels encode the **spectral properties** of the data matrix:

$$h_k(t) = \int \lambda^k e^{-2\gamma t \lambda} \, d\mu(\lambda)$$

Where $\mu$ is the **empirical spectral measure** of $H = \frac{1}{n} A^T A$.

In discrete form (given eigenvalues $\{\lambda_1, \ldots, \lambda_d\}$):

$$h_k(t) = \frac{1}{d} \sum_{j=1}^{d} \lambda_j^k \cdot e^{-2\gamma t \lambda_j}$$

Specifically:
- **$h_0(t)$**: Weighted average of exponential decays
- **$h_1(t)$**: Eigenvalue-weighted decay (drives convergence)
- **$h_2(t)$**: Kernel for the integral equation (encodes curvature)

### 3.4 Discretized Numerical Solution

The integral equation is solved iteratively with time step $\Delta t$:

$$\psi_0(t_i) = z(t_i) + r\gamma^2 \Delta t \sum_{j=0}^{i-1} h_2(t_i - t_j) \psi_0(t_j)$$

This is implemented in `src/volterra.py`.

---

## 4. Key Quantities and Formulas

### 4.1 The Hessian Matrix H

$$H = \frac{1}{n} A^T A$$

- Shape: $d \times d$
- Eigenvalues: $\{\lambda_1, \ldots, \lambda_d\}$ (all non-negative)
- The spectrum of $H$ completely determines the Volterra prediction

### 4.2 Aspect Ratio r

$$\boxed{r = \frac{d}{n}}$$

- $r < 1$: **Overdetermined** (more samples than parameters)
- $r = 1$: **Square** (interpolation threshold)
- $r > 1$: **Underdetermined** (more parameters than samples)

The aspect ratio controls:
1. The shape of the Marchenko-Pastur distribution
2. The coefficient in the Volterra equation ($r\gamma^2$)
3. The critical step size formula

### 4.3 Spectral Ratio

$$\boxed{\text{Spectral Ratio} = \frac{\lambda_{\max}}{\bar{\lambda}}}$$

Where:
- $\lambda_{\max} = \max_j \lambda_j$ = largest eigenvalue
- $\bar{\lambda} = \frac{1}{d}\sum_j \lambda_j$ = mean eigenvalue

**Why it matters**:
- For isotropic Gaussian data: Spectral Ratio $\approx (1 + \sqrt{r})^2$ (bounded)
- For real data: Spectral Ratio can be 100-300x (heavy tails)
- Large spectral ratio causes theory-practice gap in step size

### 4.4 Learning Rate Normalization

The paper uses **normalized time** where 1 epoch = 1 unit of time:

$$\eta_{\text{effective}} = \frac{\gamma}{n}$$

- $\gamma$ = step size parameter (appears in Volterra equation)
- After $n$ steps (1 epoch), the accumulated "time" is approximately 1

---

## 5. Spectral Analysis

### 5.1 Computing Eigenvalues

For efficiency, eigenvalues are computed from the smaller matrix:

```
If d > n:  Compute eigenvalues of (1/n) A A^T  (n x n matrix)
If d <= n: Compute eigenvalues of (1/n) A^T A  (d x d matrix)
```

Non-zero eigenvalues are identical for both matrices.

### 5.2 Marchenko-Pastur Distribution

For **isotropic** data (i.i.d. Gaussian entries), the eigenvalue distribution follows the **Marchenko-Pastur law**:

$$\rho(\lambda) = \frac{1}{2\pi \sigma^2 r} \frac{\sqrt{(\lambda_+ - \lambda)(\lambda - \lambda_-)}}{\lambda}$$

With support $[\lambda_-, \lambda_+]$ where:

$$\lambda_{\pm} = \sigma^2 (1 \pm \sqrt{r})^2$$

For $\sigma^2 = 1$:
- $\lambda_- = (1 - \sqrt{r})^2$
- $\lambda_+ = (1 + \sqrt{r})^2$

### 5.3 Deviation from Marchenko-Pastur

Real data deviates from Marchenko-Pastur due to:
1. **Correlations** between features
2. **Heavy-tailed** distributions
3. **Low-rank structure** (many near-zero eigenvalues)
4. **Outlier eigenvalues** (spikes outside the bulk)

This is the core reason the Volterra theory may fail on real data.

---

## 6. Aspect Ratio Analysis

### 6.1 Experimental Regimes

The experiments test three aspect ratios:

| Aspect Ratio | d | Regime | Interpretation |
|-------------|---|--------|----------------|
| r = 0.5 | 1500 | Overdetermined | More data than parameters |
| r = 1.0 | 3000 | Interpolation | At the threshold |
| r = 1.5 | 4500 | Underdetermined | More parameters than data |

### 6.2 Impact on Theory

As $r$ increases:
1. **Spectral spread** increases (larger ratio $\lambda_{\max}/\bar{\lambda}$)
2. **Critical step size** decreases ($\gamma^* \propto 1/r$)
3. **Theory-practice gap** often increases

### 6.3 Kernel Normalization for r > 1

When $r > 1$, there are $d - n$ zero eigenvalues (null space). The kernel computation accounts for this:

```python
if r <= 1.0:
    h_k = sum(lambda^k * exp(-2*gamma*t*lambda)) / d
else:
    # d - n zeros contribute exp(0) = 1 to h_0 only
    h_0 = (sum_nonzero(exp(...)) + (d - n)) / d
    h_1 = sum_nonzero(lambda * exp(...)) / d
    h_2 = sum_nonzero(lambda^2 * exp(...)) / d
```

---

## 7. Critical Step Size Theory

### 7.1 Theoretical Critical Step Size

From Theorem 1.2 of the paper, SGD converges when:

$$\boxed{\gamma < \gamma^*_{\text{theory}} = \frac{2}{r \cdot \bar{\lambda}}}$$

For normalized data where $\bar{\lambda} = 1$:

$$\gamma^*_{\text{theory}} = \frac{2}{r}$$

### 7.2 Safe (Practical) Step Size

In practice, convergence requires:

$$\boxed{\gamma < \gamma^*_{\text{safe}} = \frac{2}{\lambda_{\max}}}$$

This is the standard result that learning rate must be less than $2/L$ where $L$ is the Lipschitz constant of the gradient.

### 7.3 The Theory-Practice Gap

$$\text{Gap Ratio} = \frac{\gamma^*_{\text{theory}}}{\gamma^*_{\text{safe}}} = \frac{\lambda_{\max}}{r \cdot \bar{\lambda}}$$

For isotropic data: Gap $\approx (1 + \sqrt{r})^2 / r \approx$ small constant

For real data: Gap can be **50-300x** due to outlier eigenvalues!

### 7.4 Why the Gap Exists

The theoretical $\gamma^*$ assumes:
1. **Compact spectrum** (eigenvalues bounded, no outliers)
2. **Marchenko-Pastur law** holds
3. Asymptotic regime ($n, d \to \infty$)

Real data violates these with:
- Outlier eigenvalues (spikes)
- Heavy-tailed distributions
- Finite-size effects

---

## 8. Implementation Details

### 8.1 Core Modules

| Module | Purpose |
|--------|---------|
| `src/volterra.py` | Volterra integral equation solver |
| `src/sgd.py` | SGD implementations (batch, streaming, multi-output) |
| `src/spectral.py` | Eigenvalue computation and kernel functions |
| `src/random_features.py` | Random feature generation |
| `src/data_loader.py` | MNIST/CIFAR loading and whitening |

### 8.2 Random Features Model

Instead of raw pixels, features are generated via:

$$A = \sigma(X W)$$

Where:
- $X \in \mathbb{R}^{n \times m}$ = raw data (e.g., MNIST images)
- $W \in \mathbb{R}^{m \times d}$ = random weight matrix (fixed)
- $\sigma$ = activation function

**Shifted ReLU Activation** (from paper Appendix A.2.2):

$$g(z) = \text{ReLU}(z) - \frac{1}{\sqrt{2\pi}}$$

The shift ensures zero mean for Gaussian inputs.

### 8.3 Planted Targets

To create a realizable problem (where exact solution exists):

$$b = A \cdot x^*$$

Where $x^* \in \mathbb{R}^d$ with $\|x^*\| = 1$ (so $R = 1$).

### 8.4 Data Whitening

To make data more "isotropic":

$$\tilde{X} = \Sigma^{-1/2} (X - \mu)$$

Where:
- $\mu$ = sample mean
- $\Sigma = \frac{1}{n} X^T X$ = sample covariance
- After whitening: $E[\tilde{x}] = 0$ and $E[\tilde{x}\tilde{x}^T] = I$

---

## 9. Experiments

### 9.1 Figure 4 Reproduction

Reproduces the paper's Figure 4 comparing:
1. **Volterra**: Deterministic theoretical prediction
2. **Empirical SGD**: Actual training runs (averaged)
3. **Streaming SGD**: One-pass SGD with infinite fresh data
4. **SME**: Stochastic Modified Equation (gradient covariance noise)
5. **SDE**: Stochastic Differential Equation (isotropic noise)

**Settings**: n=1000, r=1.2, 50 epochs, 5 runs

### 9.2 Part 2: Base MNIST Experiment

Tests Volterra on MNIST random features with planted targets.

**Settings**: n=3000, epochs=20, runs=5, r=[0.5, 1.0, 1.5]

**Key outputs**:
- Loss curves (theory vs SGD)
- Eigenvalue histograms
- Volterra accuracy scatter plots
- Spectral ratio analysis

### 9.3 Part 2: Non-linear Target Experiment

Creates a **non-realizable** problem:

$$b = \text{ReLU}(A \cdot x^*)$$

This clips ~50% of targets to zero, making exact solution impossible.

**Key insight**: Volterra predicts convergence to zero, but SGD converges to non-zero irreducible error.

### 9.4 Part 2: Whitened MNIST Experiment

Tests whether whitening the data improves theory alignment.

**Pipeline**:
1. Whiten MNIST: $\tilde{X} = \Sigma^{-1/2}(X - \mu)$
2. Generate random features from $\tilde{X}$
3. Compare Volterra prediction vs SGD

**Expected result**: Lower spectral ratio, better theory-practice alignment.

### 9.5 Part 3: Criticality Analysis

Empirically maps the stability boundary by sweeping learning rates.

**Stability criterion**:
1. Loss remains finite (not NaN/Inf)
2. Final loss $\leq$ 10 $\times$ initial loss

**Learning rate sweep**: $\gamma = \gamma_{\text{safe}} \cdot 2^k$ for $k = -6, \ldots, 1$

**Key outputs**:
- Stability heatmap
- Critical gamma comparison (theory vs safe vs empirical)
- Theory overestimation factor

---

## 10. Key Findings

### 10.1 Spectral Ratio is the Key Indicator

The **spectral ratio** $\lambda_{\max}/\bar{\lambda}$ determines how well theory matches practice:

| Data Type | Spectral Ratio | Theory Accuracy |
|-----------|----------------|-----------------|
| Isotropic Gaussian | ~4-6 | Excellent |
| Whitened MNIST | ~5-15 | Good |
| Raw MNIST features | ~100-300 | Poor |

### 10.2 The Theory-Practice Gap

For MNIST random features:
- $\gamma^*_{\text{theory}}$ overestimates the safe step size by **50-300x**
- Using $\gamma^*_{\text{theory}}$ causes divergence
- Must use $\gamma^*_{\text{safe}} = 2/\lambda_{\max}$

### 10.3 Whitening Helps

Whitening the input data:
1. Reduces spectral ratio significantly
2. Makes eigenvalue distribution closer to Marchenko-Pastur
3. Reduces theory-practice gap
4. Improves Volterra prediction accuracy

### 10.4 Non-linear Targets Break the Model

When targets are non-linear ($b = \text{ReLU}(Ax^*)$):
- Problem becomes non-realizable
- SGD converges to non-zero error floor
- Volterra (which assumes realizability) predicts zero convergence
- Correlation between theory and practice drops

### 10.5 Practical Recommendations

1. **Always use** $\gamma_{\text{safe}} = 2/\lambda_{\max}$ as the stability bound
2. **Whiten data** when possible to improve theory applicability
3. **Check spectral ratio**: If $>20$, expect theory-practice gap
4. **Volterra works best** for isotropic/whitened features with realizable targets

---

## Appendix A: Complete Formula Reference

### Volterra Equation
$$\psi_0(t) = z(t) + r\gamma^2 \int_0^t h_2(t-s) \psi_0(s) \, ds$$

### Forcing Function
$$z(t) = \frac{R}{2} h_1(t) + \frac{\tilde{R}}{2} \left( r \cdot h_0(t) + (1-r) \right)$$

### Kernel Functions
$$h_k(t) = \frac{1}{d} \sum_{j=1}^{d} \lambda_j^k \cdot e^{-2\gamma t \lambda_j}$$

### Critical Step Size (Theory)
$$\gamma^*_{\text{theory}} = \frac{2}{r \cdot \bar{\lambda}}$$

### Critical Step Size (Safe)
$$\gamma^*_{\text{safe}} = \frac{2}{\lambda_{\max}}$$

### Spectral Ratio
$$\text{SR} = \frac{\lambda_{\max}}{\bar{\lambda}}$$

### Marchenko-Pastur Bounds
$$\lambda_{\pm} = (1 \pm \sqrt{r})^2$$

### Shifted ReLU
$$g(z) = \text{ReLU}(z) - \frac{1}{\sqrt{2\pi}}$$

### Whitening Transform
$$\tilde{X} = \Sigma^{-1/2} (X - \mu)$$

---

## Appendix B: Code Architecture

```
COMP450_Volterra_SGD/
├── src/
│   ├── volterra.py          # Volterra solver (VolterraSolver class)
│   ├── sgd.py                # SGD implementations
│   ├── spectral.py           # Eigenvalue computation, kernels, MP density
│   ├── random_features.py    # Feature generation, activations
│   ├── data_loader.py        # Data loading, whitening, one-hot encoding
│   └── Experiments/
│       ├── figure4_reproduction.py   # Paper figure reproduction
│       ├── part2_base.py             # Base MNIST experiment
│       ├── part2_nonlinear.py        # Non-linear target experiment
│       ├── part2_whitened.py         # Whitened data experiment
│       └── part3_criticality.py      # Step size criticality analysis
├── plots/                    # Generated figures
├── results/                  # Saved experiment results
└── data/                     # Downloaded datasets
```

---

## References

1. Paquette, C., Lee, K., Pedregosa, F., & Paquette, E. (2021). SGD in the Large: Average-case Analysis, Asymptotics, and Stepsize Criticality. *arXiv:2102.04396*

2. Marchenko, V. A., & Pastur, L. A. (1967). Distribution of eigenvalues for some sets of random matrices. *Matematicheskii Sbornik*, 114(4), 507-536.

3. Rahimi, A., & Recht, B. (2007). Random features for large-scale kernel machines. *NeurIPS*.
