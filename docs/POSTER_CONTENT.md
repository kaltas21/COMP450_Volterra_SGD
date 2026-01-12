# POSTER: Predicting SGD Dynamics via Volterra Integral Equations
## COMP450 - Learning Theory from First Principles

**Authors:** Yigit Kaltas, Kaan [Last Name]

---

# ABSTRACT

We validate that the Volterra Integral Equation predicts Stochastic Gradient Descent (SGD) dynamics with **99% accuracy** on MNIST random features. The eigenvalue spectrum of the feature matrix alone determines training behavior. We demonstrate theory robustness through three extensions: non-linear targets, whitened data, and real classification labels.

---

# 1. INTRODUCTION & MOTIVATION

**Central Question:** Can we predict SGD loss curves from eigenvalues alone?

**Why It Matters:**
- Optimal learning rate selection without trial-and-error
- Understanding when training will converge
- Bridging theory and practice in deep learning

**The Volterra Theory:**
For least squares f(x) = (1/2n)||Ax - b||², the loss evolution depends only on eigenvalues of A^T A:

```
h_k(t) = (1/d) Σ_i λ_i^k · exp(-2γtλ_i)
```

**Key Prediction:** Critical step size γ_max = 2/(r · mean(λ))

---

# 2. EXPERIMENTAL SETUP

**Data:** MNIST (n = 3000 samples, 784 pixels)

**Random Features:**
```
A = shifted_ReLU(X @ W / √d)
```
- W: Random Gaussian weights (fixed)
- Per-column centering applied

**Aspect Ratios:** r = d/n ∈ {0.5, 1.0, 1.5}

**Targets:**
- Planted: b = A @ x* (realizable)
- Non-linear: b = ReLU(A @ x*)
- Real labels: Y = one_hot(MNIST labels)

---

# 3. EXPERIMENT 1: BASE VALIDATION

## [FIGURE 1: Loss Curves - results/full_run_base/loss_curves.png]

**Result: 99% Correlation Between Volterra Prediction and SGD**

| Aspect Ratio | Correlation | Spectral Ratio | γ_theory | γ_safe |
|--------------|-------------|----------------|----------|--------|
| r = 0.5 | **98.99%** | 133.2 | 4.00 | 0.015 |
| r = 1.0 | **99.04%** | 270.6 | 2.00 | 0.007 |
| r = 1.5 | **98.44%** | 272.4 | 1.33 | 0.007 |

## [FIGURE 2: Eigenvalue Spectrum - results/full_run_base/eigenvalue_spectrum.png]

**Key Finding: Theory-Practice Gap = 180-270x**

**Why?** MNIST violates Gaussian assumptions:
- Spatial correlation between pixels
- Sparse data (~75% zeros)
- Bounded values [0,1]

This creates **spiked eigenvalue distribution** with outliers far exceeding Marchenko-Pastur bounds.

## [FIGURE 3: Volterra vs SGD Scatter - results/full_run_base/volterra_vs_sgd_scatter.png]

---

# 4. EXPERIMENT 2: NON-LINEAR TARGETS

## [FIGURE 4: Loss Curves - results/full_run_nonlinear/loss_curves.png]

**Setup:** b = ReLU(A @ x*) instead of b = A @ x*

**Result: ~50% of Targets Become Zero**

| Aspect Ratio | Zero Fraction | Final Loss | Initial Loss |
|--------------|---------------|------------|--------------|
| r = 0.5 | 52.57% | 0.168 | 0.364 |
| r = 1.0 | 49.03% | 0.135 | 0.242 |
| r = 1.5 | 48.40% | 0.060 | 0.098 |

## [FIGURE 5: Zero Fraction Analysis - results/full_run_nonlinear/zero_fraction.png]

**Why Exactly 50%?**

1. A has centered columns → E[A_ij] = 0
2. By CLT: (A @ x*)_i ~ N(0, σ²)
3. For symmetric distribution: P(X < 0) = 0.5
4. ReLU clips negative half to zero

**Key Finding: Irreducible Error Floor**

Non-realizable problem → loss cannot reach zero. Higher r reduces error by expanding column space of A.

## [FIGURE 6: Final Loss Comparison - results/full_run_nonlinear/final_loss_comparison.png]

---

# 5. EXPERIMENT 3: WHITENED MNIST

## [FIGURE 7: Loss Curves - results/quick_whitened/loss_curves.png]

**Setup:** Apply whitening transform before random features:
```
X_whitened = Σ^{-1/2}(X - μ)
```

**Result: 74% Reduction in Spectral Ratio**

| Aspect Ratio | Non-Whitened | Whitened | Reduction |
|--------------|--------------|----------|-----------|
| r = 0.5 | 133.2 | 36.3 | **73%** |
| r = 1.0 | 270.6 | 71.1 | **74%** |
| r = 1.5 | 272.4 | 70.3 | **74%** |

## [FIGURE 8: Spectral Comparison - results/quick_whitened/summary_comparison.png]

**Why Does Whitening Help?**

- Original MNIST covariance condition number: **51,396,544**
- After whitening: All directions have equal variance
- Random projection preserves isotropy better
- Result: 3-4x larger safe learning rates

## [FIGURE 9: Eigenvalue Spectrum - results/quick_whitened/eigenvalue_spectrum.png]

---

# 6. EXPERIMENT 4: REAL MNIST LABELS

## [FIGURE 10: Loss Curves - results/quick_real_labels/loss_curves.png]

**Setup:** Classification as 10-output regression
```
Y = one_hot(MNIST labels)  # Shape: (n, 10)
```

**Result: Double Descent Phenomenon**

| Aspect Ratio | Accuracy | Spectral Ratio | R Value |
|--------------|----------|----------------|---------|
| r = 0.5 | **99.95%** | 88.2 | 0.28 |
| r = 1.0 | 73.35% | 170.0 | unstable |
| r = 1.5 | **100.00%** | 167.4 | unstable |

## [FIGURE 11: Accuracy Summary - results/quick_real_labels/accuracy_summary.png]

**Why Does Accuracy Dip at r = 1.0?**

- At r = 1: Matrix A is square (potentially singular)
- Interpolation threshold causes numerical instability
- R parameter explodes (~10^12)
- This is the famous **double descent** phenomenon

## [FIGURE 12: Class Distribution - results/quick_real_labels/class_distribution.png]

**Why 100% at r = 1.5?**

- More features than samples → interpolation regime
- Model can fit training data exactly
- A @ X = Y achieved perfectly

---

# 7. KEY FINDINGS

## Finding 1: Eigenvalues Determine SGD Dynamics
**99% correlation** between Volterra prediction and actual SGD, using only eigenvalue information.

## Finding 2: Spectral Ratio Explains Theory-Practice Gap
```
Gap Ratio ≈ λ_max / mean(λ) = Spectral Ratio
```
MNIST: 133-272x gap due to non-Gaussian structure.

## Finding 3: Theory is Robust to Assumption Violations
- Works for non-realizable problems (non-linear targets)
- Works for multi-output regression (real labels)
- Dynamics are target-independent

## Finding 4: Preprocessing Matters
Whitening reduces spectral ratio by **74%**, allowing **4x larger learning rates**.

## Finding 5: Interpolation Threshold is Special
At r = 1: numerical instability, double descent in accuracy. Avoid r ≈ 1 in practice.

---

# 8. CONCLUSIONS

1. **Volterra theory accurately predicts SGD dynamics** on real (non-Gaussian) data with 99% correlation

2. **The spectral ratio is the key quantity** bridging theory and practice - it determines the gap between theoretical and safe step sizes

3. **The theory is robust**: works for non-realizable problems, extends to classification, benefits from preprocessing

4. **Practical recommendations:**
   - Always compute λ_max for step size selection
   - Use whitening/standardization (4x speedup)
   - Avoid aspect ratios near r = 1

---

# FIGURE LIST

| Figure | File Path | Description |
|--------|-----------|-------------|
| 1 | results/full_run_base/loss_curves.png | Base experiment loss curves |
| 2 | results/full_run_base/eigenvalue_spectrum.png | Eigenvalue distribution vs MP |
| 3 | results/full_run_base/volterra_vs_sgd_scatter.png | Correlation scatter plot |
| 4 | results/full_run_nonlinear/loss_curves.png | Non-linear target loss curves |
| 5 | results/full_run_nonlinear/zero_fraction.png | Zero fraction analysis |
| 6 | results/full_run_nonlinear/final_loss_comparison.png | Final loss vs aspect ratio |
| 7 | results/quick_whitened/loss_curves.png | Whitened experiment loss curves |
| 8 | results/quick_whitened/summary_comparison.png | Whitened vs non-whitened comparison |
| 9 | results/quick_whitened/eigenvalue_spectrum.png | Whitened eigenvalue spectrum |
| 10 | results/quick_real_labels/loss_curves.png | Real labels loss curves |
| 11 | results/quick_real_labels/accuracy_summary.png | Classification accuracy |
| 12 | results/quick_real_labels/class_distribution.png | MNIST label distribution |

---

# REFERENCES

1. Bach, F. (2024). *Learning Theory from First Principles*. MIT Press.
2. Marchenko, V. A., & Pastur, L. A. (1967). Distribution of eigenvalues for some sets of random matrices.
