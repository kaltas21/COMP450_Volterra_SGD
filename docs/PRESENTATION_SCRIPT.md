# Presentation Script: Volterra SGD Project
## COMP450 - Learning Theory from First Principles
### Duration: 15 minutes (Yigit: ~7-8 min, Kaan: ~7-8 min)

---

# PART 1: YIGIT (7-8 minutes)
## Theory, Setup, and Base Experiment

---

### [SLIDE: Title] (30 seconds)

"Good morning everyone. Today we're presenting our project on **predicting SGD dynamics using the Volterra Integral Equation**.

The central question we're investigating is: **Can we predict how SGD will behave just by looking at the eigenvalues of our data matrix?**

The answer, as we'll show, is yes - with 99% accuracy."

---

### [SLIDE: Motivation & Problem Setup] (1.5 minutes)

"Let's start with the setup. We're doing **least squares regression**:

```
minimize f(x) = (1/2n) ||Ax - b||²
```

This is the simplest possible learning problem - a quadratic loss. But even here, understanding SGD dynamics is non-trivial.

**Why does this matter?**

If we can predict how loss decreases over time, we can:
- Choose optimal learning rates without trial and error
- Understand when training will converge
- Know in advance if our setup will work

**The key insight from theory**: For quadratic loss, the Hessian is:

```
H = (1/n) A^T A
```

And here's the crucial point: **the eigenvalues of this Hessian completely determine the optimization landscape**.

- Large eigenvalues → fast convergence in that direction
- Small eigenvalues → slow convergence
- The ratio λ_max/λ_min → condition number → optimization difficulty

So the question becomes: can we predict the entire loss trajectory just from these eigenvalues?"

---

### [SLIDE: The Volterra Integral Equation] (2 minutes)

"This is where the Volterra theory comes in. From Bach's book and related work, we know that for least squares with SGD, the expected loss evolves according to an integral equation.

**The key formula** - the loss at time t depends on spectral moments:

```
h_k(t) = ∫ λ^k · exp(-2γtλ) dμ(λ)
```

where μ(λ) is the eigenvalue distribution.

**What does this mean intuitively?**

Each eigenvalue contributes an exponentially decaying term. Large eigenvalues decay fast, small eigenvalues decay slow. The overall loss is a weighted combination of these decays.

**The beautiful part**: We don't need to know anything about the data X, the targets b, or the initialization. Just the eigenvalues of A^T A.

**Critical step sizes from theory**:

```
γ_theory = 2 / (r · mean(λ))
```

where r = d/n is the aspect ratio. This is the theoretical maximum stable step size.

But there's a catch - and this is important - this assumes eigenvalues follow the Marchenko-Pastur distribution, which assumes Gaussian data. Real data isn't Gaussian."

---

### [SLIDE: Experimental Setup - Random Features on MNIST] (1 minute)

"For our experiments, we use **random features on MNIST**.

**Why this setup?**

1. MNIST gives us real, structured data - not synthetic Gaussians
2. Random features A = σ(XW) create a controlled high-dimensional setting
3. We can vary the aspect ratio r = d/n easily

**Our construction**:
```
A = shifted_ReLU(X @ W / √d)
```

Then we center columns so mean(A_j) = 0.

**The planted target**: b = A @ x* where ||x*|| = 1

This makes the problem **realizable** - there exists an exact solution.

**Parameters**: n = 3000 samples, d varies with r, 20 epochs, 5 runs averaged."

---

### [SLIDE: Base Experiment Results] (2 minutes)

"Here are our main results. [Point to loss curves figure]

**What you're seeing**: Blue is actual SGD loss, dashed line is Volterra prediction.

**The correlation**: 98.4% to 99.0% across aspect ratios.

This is remarkable - the theory predicts real SGD behavior almost perfectly, just from eigenvalues.

**But look at the spectral ratios**: [Point to table]

| r | Spectral Ratio | γ_theory | γ_safe |
|---|----------------|----------|--------|
| 0.5 | 133 | 4.0 | 0.015 |
| 1.0 | 271 | 2.0 | 0.007 |
| 1.5 | 272 | 1.3 | 0.007 |

**The theory-practice gap**: γ_theory is 180-270 times larger than γ_safe!

**Why does this happen?**

MNIST has highly non-uniform structure:
- Spatial correlations between pixels
- Sparse (mostly black background)
- Low intrinsic dimension (~50D manifold in 784D space)

This creates a **spiked eigenvalue distribution** - a few very large eigenvalues, many near zero. The theory assumes Marchenko-Pastur with bounded support, but real data has outliers.

**The critical insight**: One eigenvalue at 270 can destabilize training even if the mean is 1. Using γ = 2/mean would give updates of 2 × 270 = 540 in that direction - divergence!

So while Volterra predicts the dynamics beautifully, the critical step size must be computed from λ_max, not from theory."

---

### [TRANSITION TO KAAN] (15 seconds)

"So we've validated that eigenvalues determine SGD dynamics with 99% accuracy. But does this hold when we change the problem? Kaan will now show three extensions that test the robustness of this theory."

---

# PART 2: KAAN (7-8 minutes)
## Extensions, Insights, and Conclusions

---

### [SLIDE: Extension 1 - Non-Linear Targets] (2 minutes)

"Thank you Yigit. Our first extension tests: **what happens when the problem isn't realizable?**

**The change**: Instead of b = A @ x*, we use:
```
b = ReLU(A @ x*)
```

This applies ReLU to the targets, creating a **non-realizable problem** - no x exists such that Ax = b exactly.

**Key observation**: [Point to zero fraction figure]

~50% of targets become zero. Why exactly 50%?

**The mathematical reason**:
1. A has centered columns, so E[A_ij] = 0
2. By Central Limit Theorem, (A @ x*)_i is approximately Gaussian with mean 0
3. For any symmetric distribution around 0: P(X < 0) = P(X > 0) = 0.5
4. ReLU clips exactly half to zero

This is pure geometry, not a coincidence.

**The consequence**: An **irreducible error floor** appears.

| r | Zero Fraction | Final Loss |
|---|---------------|------------|
| 0.5 | 52.6% | 0.168 |
| 1.0 | 49.0% | 0.135 |
| 1.5 | 48.4% | 0.060 |

The loss can't reach zero because the problem is non-realizable.

**Why does loss decrease with r?**

More features → larger column space of A → better linear approximation to the non-linear target. At r=1.5, we can project b more accurately onto span(A).

**Key finding**: Volterra still predicts the dynamics correctly! The theory works even for non-realizable problems because it depends on eigenvalues, not targets."

---

### [SLIDE: Extension 2 - Whitened MNIST] (2 minutes)

"Our second extension asks: **can we reduce the theory-practice gap?**

Yigit showed that MNIST has spectral ratio 133-272 because of non-uniform structure. What if we remove that structure?

**Whitening transform**:
```
X_whitened = Σ^{-1/2}(X - μ)
```

This enforces E[X] = 0 and E[XX^T] = I - all directions have equal variance.

**The results are dramatic**: [Point to comparison figure]

| r | Non-Whitened | Whitened | Reduction |
|---|--------------|----------|-----------|
| 0.5 | 133 | 36 | 73% |
| 1.0 | 271 | 71 | 74% |
| 1.5 | 272 | 70 | 74% |

**Spectral ratio drops by 74%** - almost 4x improvement!

**Why does whitening help?**

The original MNIST covariance has condition number **51 million**. A few directions (common digit patterns) have huge variance, most have near-zero. This extreme spread propagates through random features to A^T A.

After whitening, all input directions have equal variance. Random projection of isotropic data stays more isotropic.

**Practical implication**:
- γ_safe increases from 0.007 to 0.028 (4x larger learning rate is safe!)
- Theory predictions become more accurate because data is closer to the Gaussian assumption

**But spectral ratio isn't 1** - why? Because ReLU is non-linear. Even whitened inputs produce non-Gaussian features. The activation creates asymmetry and correlations."

---

### [SLIDE: Extension 3 - Real MNIST Labels] (2 minutes)

"Our final extension: **does the theory work for real classification?**

**Setup**: One-hot encode MNIST labels as targets.
```
y = 3 → Y = [0,0,0,1,0,0,0,0,0,0]
```

This transforms 10-class classification into 10-output regression.

**Results**: [Point to accuracy figure]

| r | Accuracy | Notes |
|---|----------|-------|
| 0.5 | 99.95% | Excellent |
| 1.0 | 73.35% | Dip! |
| 1.5 | 100% | Perfect |

**The dip at r=1.0** - this is the famous **double descent** phenomenon!

**Why does accuracy dip at r=1?**

At r=1, the matrix A is square. This is the **interpolation threshold**:
- r < 1: overdetermined, unique solution
- r = 1: exactly determined, potentially singular
- r > 1: underdetermined, infinitely many solutions

Near r=1, smallest singular values approach zero. The matrix becomes ill-conditioned, numerical instability causes poor solutions.

**Why 100% at r=1.5?**

With more features than samples, the model can **interpolate** - fit training data exactly. Perfect fit → perfect classification.

**For Volterra theory**: We need to compute R from the optimal solution:
```
R = (1/10) ||X_opt||_F²
```

At r=1, this explodes (R ~ 10^12) because the pseudoinverse is unstable. Theory still applies in principle, but the R parameter becomes unreliable near the threshold."

---

### [SLIDE: Key Insights Summary] (1 minute)

"Let me summarize what we learned:

**1. Eigenvalues determine everything** (for quadratic loss)
- 99% correlation between Volterra prediction and SGD
- Just need the spectrum of A^T A

**2. Spectral ratio explains the theory-practice gap**
- Theory assumes bounded eigenvalue support
- Real data has outliers → must use λ_max for step size
- Gap ratio ≈ spectral ratio

**3. Theory is robust to assumption violations**
- Works for non-realizable problems (non-linear targets)
- Works for multi-output regression (real labels)
- Dynamics depend on eigenvalues, not targets

**4. Preprocessing matters**
- Whitening reduces spectral ratio by 74%
- Allows 4x larger learning rates
- Brings theory closer to practice

**5. The interpolation threshold is special**
- r=1 causes numerical issues
- Double descent in accuracy
- Avoid r ≈ 1 in practice"

---

### [SLIDE: What Would Happen If...?] (1 minute)

"Finally, some predictions based on our understanding:

**Q: What if we used sigmoid instead of ReLU on targets?**
A: Similar ~50% clipping (sigmoid(0)=0.5), but smoother transition. Expect similar irreducible error.

**Q: What if we increased n with fixed r?**
A: Correlation should improve (currently 99%, asymptotically exact). Spectral ratio would remain similar - it's determined by data structure, not sample size.

**Q: What if we used PCA instead of whitening?**
A: Partial improvement. PCA removes small eigenvalue directions but doesn't equalize remaining variances. Whitening is more thorough.

**Q: Why not use the theoretical γ_max with clipping?**
A: You could! Gradient clipping prevents divergence. But you'd still see oscillations in the large-eigenvalue directions.

**Q: Would regularization help at r=1?**
A: Yes! Ridge regression (adding λI to A^T A) removes the singularity. This stabilizes both the solution and the R parameter for Volterra theory."

---

### [SLIDE: Conclusions] (30 seconds)

"To conclude:

We validated that the **Volterra Integral Equation predicts SGD dynamics with 99% accuracy** on real MNIST data - the eigenvalue spectrum alone determines training behavior.

The **spectral ratio** is the key quantity bridging theory and practice - it explains why theoretical step sizes are 100-270x larger than safe practical values.

And through our extensions, we showed the theory is **robust**: it works for non-realizable problems, benefits from preprocessing, and extends to real classification tasks.

Thank you. We're happy to take questions."

---

# POSTER FIGURE SUGGESTIONS

1. **Hero figure**: Loss curves with Volterra vs SGD overlay (shows 99% match)
2. **Eigenvalue histogram**: With MP overlay showing the gap
3. **Spectral ratio comparison**: Bar chart (whitened vs non-whitened)
4. **Results table**: All 4 experiments with key metrics
5. **Causal diagram**: Data structure → Spectral ratio → Theory-practice gap

---

# ANTICIPATED QUESTIONS

**Q: Why use random features instead of training a neural network?**
A: Random features give us a controlled setting where the feature matrix is fixed. This isolates the SGD dynamics from representation learning. The Volterra theory applies to the final layer of any neural network.

**Q: How does this relate to NTK (Neural Tangent Kernel)?**
A: Closely related! NTK describes infinite-width neural networks as kernel regression. Our random features are a finite-width analog. The eigenvalue dependence is the same.

**Q: What about mini-batch SGD instead of single-sample?**
A: Volterra theory extends to mini-batches. The effective step size becomes γ/batch_size, and variance terms change. The eigenvalue dependence remains.

**Q: Why does the correlation drop slightly at r=1.5?**
A: In the interpolation regime, multiple solutions exist. The minimum-norm solution found by SGD may differ slightly from what Volterra predicts. Also finite sample effects are larger.

**Q: Could you use this to adaptively set learning rates?**
A: Yes! Compute eigenvalues once, then use γ = α · (2/λ_max) where α ∈ [0.5, 0.9]. No hyperparameter search needed.
