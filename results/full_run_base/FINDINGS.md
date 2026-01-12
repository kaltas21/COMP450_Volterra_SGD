# Base Experiment Findings: Full Run Analysis

## Executive Summary

The Volterra Integral Equation achieves **98.4-99.0% correlation** with actual SGD dynamics on MNIST random features, validating that eigenvalues alone determine training behavior. This improved from ~95% in smaller test runs to ~99% with more samples and runs, demonstrating the theory's asymptotic nature.

---

## Key Results

| Aspect Ratio | Correlation | Spectral Ratio | Theory-Practice Gap |
|--------------|-------------|----------------|---------------------|
| r = 0.5 | **98.99%** | 133.2 | 266x |
| r = 1.0 | **99.04%** | 270.6 | 271x |
| r = 1.5 | **98.44%** | 272.4 | 182x |

---

## Why 99% Correlation? Mathematical Explanation

### 1. The Law of Large Numbers in Action

The Volterra theory predicts the **expected** (average) loss trajectory. With:
- **n = 3000 samples** (vs 2000 in test runs)
- **5 SGD runs** (vs 2 in test runs)
- **20 epochs** (vs 15 in test runs)

The empirical average converges closer to the true expectation. This is pure statistics: more data → less variance → better match to theoretical expectation.

**Mathematical basis**: By the Central Limit Theorem, the sample mean converges to the true mean at rate O(1/√n). With 3000 samples vs 2000, we get √(3000/2000) ≈ 1.22x improvement in convergence.

### 2. Why Eigenvalues Determine Everything

The loss function for least squares is:
```
f(x) = (1/2n) ||Ax - b||²
```

This is a **quadratic function**. The Hessian matrix is:
```
H = (1/n) A^T A
```

For quadratic functions, the eigenvalues of H completely determine:
- **Convergence rate**: Larger eigenvalues → faster convergence in that direction
- **Stability limit**: γ < 2/λ_max prevents divergence
- **Condition number**: λ_max/λ_min determines optimization difficulty

The Volterra equation exploits this by computing loss evolution purely from spectral moments:
```
h_k(t) = ∫ λ^k · exp(-2γtλ) dμ(λ)
```

where μ(λ) is the eigenvalue distribution.

### 3. Why MNIST Has Huge Spectral Ratios (133-272x)

**Observed**: Spectral ratios of 133-272, meaning λ_max is 133-272 times larger than λ_mean.

**Cause**: MNIST images have highly non-uniform structure:

1. **Spatial correlation**: Neighboring pixels are correlated (edges, strokes)
2. **Sparsity**: Most pixels are zero (black background)
3. **Low intrinsic dimension**: Digit images lie on a low-dimensional manifold

When we compute random features A = σ(XW), this structure creates:
- A few **dominant directions** (large eigenvalues) corresponding to common patterns
- Many **near-zero eigenvalues** corresponding to noise directions

**Mathematical consequence**: The covariance matrix X^T X has a few large eigenvalues and a long tail of small ones. After random projection and ReLU, this persists in A^T A.

### 4. Why Marchenko-Pastur Doesn't Fit

The Marchenko-Pastur distribution assumes:
- Entries of X are **i.i.d.** (independent, identically distributed)
- Entries are **Gaussian** or sub-Gaussian

MNIST violates both:
- Pixels are **correlated** (spatial structure)
- Pixels are **bounded** [0,1] and **sparse** (many zeros)
- Distribution is highly **non-Gaussian** (bimodal: black or white)

**Result**: The empirical eigenvalue distribution is far more concentrated near zero with heavier tails than Marchenko-Pastur predicts.

### 5. The Theory-Practice Gap (182-271x)

**Theory says**: γ_max = 2/(r · mean(λ)) = 2/r (since mean(λ)=1 after scaling)

**Practice requires**: γ_safe = 2/λ_max

**Gap ratio** = λ_max/mean(λ) = spectral ratio

This gap exists because:
- Theory assumes eigenvalues follow Marchenko-Pastur with **bounded support**
- Real data has **outlier eigenvalues** far beyond the bulk
- A single large eigenvalue can destabilize SGD even if most eigenvalues are small

**Intuition**: Imagine 999 eigenvalues at 1 and one eigenvalue at 270. The mean is ~1.27, but using γ = 2/1.27 = 1.57 would cause divergence because the update in the outlier direction would be 1.57 × 270 = 424 >> 2.

---

## Why Correlation Improved from 95% to 99%

### Test Run (n=2000, runs=2, epochs=15)
- Higher variance in SGD estimates
- Fewer samples → eigenvalue estimates less accurate
- Short trajectories → less data for correlation

### Full Run (n=3000, runs=5, epochs=20)
- Lower variance from averaging 5 runs
- More samples → better spectral estimation
- Longer trajectories → more points for correlation

**The Volterra theory is asymptotically exact** as n,d → ∞ with fixed ratio r. Larger n brings us closer to this asymptotic regime, hence better correlation.

---

## Aspect Ratio Effects

### r = 0.5 (Underdetermined: more samples than features)
- d = 1500 features, n = 3000 samples
- Lower spectral ratio (133 vs 270)
- More stable optimization
- Correlation: 98.99%

### r = 1.0 (Square system)
- d = 3000 features, n = 3000 samples
- Higher spectral ratio (270)
- Transition point in theory
- Correlation: 99.04%

### r = 1.5 (Overdetermined: more features than samples)
- d = 4500 features, n = 3000 samples
- Similar spectral ratio to r=1.0 (272)
- Interpolation regime (can fit training data perfectly)
- Correlation: 98.44%

**Key insight**: The spectral ratio increases dramatically from r=0.5 to r=1.0, then plateaus. This suggests the outlier eigenvalues are primarily determined by the data structure (MNIST), not the aspect ratio.

---

## Eigenvalue Spectrum Analysis

The eigenvalue histogram shows:
1. **Bulk near zero**: Most eigenvalues are tiny (< 5)
2. **Long tail**: A few eigenvalues extend to 130-270
3. **Gap from Marchenko-Pastur**: Empirical distribution much more concentrated

This "spiked" spectrum is characteristic of:
- **Low-rank structure** in the data
- **Random feature models** with structured inputs
- **Real-world data** vs synthetic Gaussian data

---

## Implications for Practice

### 1. Learning Rate Selection
Never trust γ_theory = 2/(r·mean(λ)). Always compute λ_max and use:
```
γ_safe = α · (2/λ_max), where α ∈ [0.5, 0.9]
```

### 2. Data Preprocessing Matters
The huge spectral ratio (133-272x) suggests preprocessing could help:
- **Whitening**: Could reduce spectral ratio significantly
- **PCA**: Remove low-variance directions
- **Normalization**: Standardize feature scales

### 3. The Theory Works Despite Violations
Even though MNIST violates Gaussian assumptions severely, the Volterra prediction achieves 99% correlation. This suggests the theory is **robust** and captures the essential dynamics determined by the spectrum.

---

## Summary

The base experiment validates that:

1. **Eigenvalues determine SGD dynamics** with 99% accuracy
2. **Spectral ratio** (not aspect ratio) is the key quantity for step-size selection
3. **Theory-practice gap** of 182-271x is explained by outlier eigenvalues
4. **More data improves correlation** (95% → 99%) due to asymptotic nature of theory
5. **MNIST's structure** creates concentrated spectra with heavy tails, far from Gaussian

The Volterra Integral Equation successfully predicts training dynamics from eigenvalues alone, even on real (non-Gaussian) data.
