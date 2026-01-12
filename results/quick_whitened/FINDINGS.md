# Whitened MNIST Experiment Findings

## Executive Summary

Whitening MNIST data before applying random features **dramatically reduces the spectral ratio** from 133-272x (non-whitened) to **36-71x** (whitened). This 3-4x improvement brings the theoretical critical step size closer to the safe practical step size, validating that the Volterra theory's Gaussian assumption matters for the theory-practice gap.

---

## Key Results

| Aspect Ratio | Spectral Ratio (Whitened) | Spectral Ratio (Non-whitened) | Improvement |
|--------------|---------------------------|-------------------------------|-------------|
| r = 0.5 | 36.26 | 133.20 | **3.7x** |
| r = 1.0 | 71.07 | 270.57 | **3.8x** |
| r = 1.5 | 70.34 | 272.44 | **3.9x** |

---

## Why Does Whitening Reduce Spectral Ratio? Mathematical Explanation

### 1. The Problem with Raw MNIST

Raw MNIST images have highly non-uniform structure:
```
Covariance condition number: 51,396,544 (huge!)
Eigenvalue range: [~0, 250]
```

This extreme condition number means:
- A few directions have very high variance (dominant patterns)
- Most directions have near-zero variance (noise)
- The data lies on a low-dimensional manifold embedded in R^784

### 2. What Causes This Structure?

**Spatial correlation**: Neighboring pixels are correlated
```
Cov(pixel_i, pixel_j) >> 0 when i,j are neighbors
```
Strokes and edges create local correlations.

**Sparsity**: Most pixels are zero (black background)
```
~75% of pixels are exactly 0
```
Only the digit region has non-zero values.

**Low intrinsic dimension**: Digit images lie on a ~50-dimensional manifold
```
dim(manifold) << 784 = ambient dimension
```

### 3. The Whitening Transform

Whitening applies:
```
X_whitened = Sigma^{-1/2} (X - mu)
```

Where:
- mu = mean(X) removes the bias
- Sigma^{-1/2} rescales each principal direction to have unit variance

**After whitening**:
```
E[X_whitened] = 0
E[X_whitened @ X_whitened^T] = I
```

All directions now have equal variance = 1.

### 4. Why This Reduces Spectral Ratio

**Before whitening**:
- Input covariance has eigenvalues spanning [0, 250]
- Random projection preserves this spread
- Feature matrix A inherits extreme eigenvalue spread

**After whitening**:
- Input covariance is identity (all eigenvalues = 1)
- Random projection of isotropic data stays more isotropic
- Feature matrix A has smaller eigenvalue spread

**Mathematical chain**:
```
Extreme input covariance → Extreme A^T A covariance → Large spectral ratio

Isotropic input covariance → More isotropic A^T A → Smaller spectral ratio
```

---

## Why Isn't Spectral Ratio = 1 After Whitening?

### 1. ReLU Non-Linearity

Even with whitened inputs, the feature construction involves ReLU:
```
A = shifted_ReLU(X_whitened @ W)
```

ReLU is non-linear and creates non-Gaussian features:
- Clips negative values to zero
- Creates asymmetric distribution
- Introduces correlations between outputs

### 2. Per-Column Centering

We center each column of A:
```
A = A - mean(A, axis=0)
```

This modifies the covariance structure and can affect eigenvalues.

### 3. Finite Sample Effects

With n = 2000 samples, we're not in the asymptotic regime where:
```
(1/n) A^T A → population covariance
```

Finite sample noise adds to the spectral spread.

### 4. The Marchenko-Pastur Bound

Even for truly Gaussian data, the spectral ratio is bounded by:
```
lambda_max / lambda_mean ≤ (1 + sqrt(r))^2 / 1 = (1 + sqrt(r))^2
```

For r = 1.0: theoretical bound = (1 + 1)^2 = 4
Observed with whitening: 71 (still much larger due to ReLU)

---

## Comparison: Before vs After Whitening

### Spectral Properties

| Property | Non-Whitened | Whitened | Change |
|----------|--------------|----------|--------|
| Input cov condition # | 51,396,544 | 1.0 | -99.99999% |
| Spectral ratio (r=0.5) | 133.2 | 36.3 | -73% |
| Spectral ratio (r=1.0) | 270.6 | 71.1 | -74% |
| Spectral ratio (r=1.5) | 272.4 | 70.3 | -74% |

### Step Size Implications

| Aspect Ratio | gamma_safe (Non-whitened) | gamma_safe (Whitened) | Improvement |
|--------------|---------------------------|------------------------|-------------|
| r = 0.5 | 0.0150 | 0.0552 | **3.7x** |
| r = 1.0 | 0.0074 | 0.0281 | **3.8x** |
| r = 1.5 | 0.0073 | 0.0284 | **3.9x** |

**Key insight**: Whitening allows 3-4x larger learning rates before instability.

---

## Why Does Spectral Ratio Increase from r=0.5 to r=1.0?

### The Transition at r = 1

| r | Spectral Ratio |
|---|----------------|
| 0.5 | 36.26 |
| 1.0 | 71.07 |
| 1.5 | 70.34 |

The spectral ratio roughly doubles from r=0.5 to r=1.0, then plateaus.

### Mathematical Explanation

At r = 1.0, the feature matrix A is square (n x n). This is the **interpolation threshold** where:
- For r < 1: System is overdetermined, unique solution exists
- For r > 1: System is underdetermined, infinitely many solutions

Near r = 1, the smallest non-zero eigenvalues approach zero, increasing the spectral ratio.

### The Plateau at r > 1

For r > 1, the matrix has n - d zero eigenvalues by construction (more columns than rows). The non-zero eigenvalues stabilize, causing the spectral ratio to plateau.

---

## Implications for Practice

### 1. Preprocess Your Data
Whitening (or at minimum standardization) significantly improves optimization:
- Allows larger learning rates
- Faster convergence
- More stable training

### 2. The 3-4x Rule
For MNIST-like structured data:
```
spectral_ratio(whitened) ≈ spectral_ratio(non-whitened) / 4
```

This suggests whitening provides roughly one "order of magnitude" improvement in step size safety.

### 3. When Whitening Helps Most
- **High help**: Structured data (images, text, correlated features)
- **Low help**: Already isotropic data (standardized tabular data)
- **No help**: Data with intrinsic low dimension (whitening can't fix this)

### 4. Theory-Practice Gap Reduced
With whitening:
- Theory assumes isotropic data (Gaussian-like)
- Whitened data is closer to isotropic
- Therefore theory predictions become more accurate

---

## Summary

The whitened MNIST experiment demonstrates:

1. **Whitening reduces spectral ratio by ~74%** (from 133-272 to 36-71) because it removes the extreme eigenvalue spread in the input covariance

2. **Condition number drops from 51M to 1** because whitening explicitly enforces E[XX^T] = I

3. **3-4x larger learning rates are safe** because lambda_max is 3-4x smaller relative to lambda_mean

4. **ReLU prevents perfect isotropy** - even whitened data has spectral ratio > 1 due to the non-linear activation

5. **Theory-practice gap shrinks** because whitened data better satisfies the Gaussian assumption underlying Volterra theory

The experiment validates that preprocessing matters: the same Volterra theory applies to both whitened and non-whitened data, but the gap between theoretical and practical step sizes depends on how well the data matches the theory's assumptions.
