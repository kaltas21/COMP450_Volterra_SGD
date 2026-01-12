# Complete Project Explanation: Volterra SGD
## A Detailed Guide for Undergraduates (and Anyone Who Wants to Really Understand)

---

# Table of Contents

1. [What Are We Even Doing?](#part-1-what-are-we-even-doing)
2. [The Mathematical Foundation](#part-2-the-mathematical-foundation)
3. [Why Random Features on MNIST?](#part-3-why-random-features-on-mnist)
4. [The Base Experiment: Step by Step](#part-4-the-base-experiment)
5. [Extension 1: Non-Linear Targets](#part-5-non-linear-targets)
6. [Extension 2: Whitened MNIST](#part-6-whitened-mnist)
7. [Extension 3: Real MNIST Labels](#part-7-real-mnist-labels)
8. [Connecting Everything Together](#part-8-connecting-everything)
9. [What We Learned](#part-9-what-we-learned)

---

# Part 1: What Are We Even Doing?

## The Big Picture Question

Imagine you're training a machine learning model. You pick a learning rate, hit "train", and watch the loss curve go down. Sometimes it works great. Sometimes it explodes. Sometimes it's painfully slow.

**The question we're asking is**: Can we predict exactly how that loss curve will look BEFORE we even start training?

Not approximately. Not "it'll probably go down." We want to predict the exact shape of the loss curve, at every point in time, with high accuracy.

**The answer**: Yes, we can. With 99% accuracy. And we only need one thing: the eigenvalues of a certain matrix.

## Why Should Anyone Care?

This matters for several reasons:

1. **Learning Rate Selection**: Instead of trying 50 different learning rates, we could compute the optimal one mathematically.

2. **Understanding Deep Learning**: Even though we're studying a simple model, the insights apply to the last layer of any neural network.

3. **Theory Meets Practice**: There's beautiful math (the Volterra Integral Equation) that supposedly predicts SGD behavior. Does it actually work on real data? We're testing that.

4. **Debugging Training**: If you know what the loss curve SHOULD look like, you can detect problems early.

## What's the Volterra Integral Equation?

Don't worry about the name. Here's what it says in plain English:

> "If you know the eigenvalues of your data matrix, you can compute exactly how the loss will decrease over time during SGD training."

That's it. Eigenvalues in, loss curve out.

The math is a bit involved (we'll get there), but the concept is simple: the eigenvalues encode everything about the optimization landscape, and the optimization landscape determines how training progresses.

---

# Part 2: The Mathematical Foundation

## Let's Start from Scratch: What is Least Squares?

We have:
- A matrix A of size (n × d) - this is our "features" or "data"
- A vector b of size (n × 1) - this is our "targets" or "labels"
- We want to find x of size (d × 1) such that Ax ≈ b

The least squares problem is:

```
minimize f(x) = (1/2n) ||Ax - b||²
```

In words: find the x that makes Ax as close to b as possible, measured by squared error.

**Why (1/2n)?**
- The 1/2 is for convenience - when we take derivatives, the 2 from the square cancels with the 1/2
- The 1/n normalizes by sample size, so the loss doesn't depend on how much data we have

## What is SGD (Stochastic Gradient Descent)?

Full gradient descent would compute:

```
x_{t+1} = x_t - γ · ∇f(x_t)
```

where γ is the learning rate and ∇f is the gradient of the full loss.

**SGD** instead picks ONE random sample i at each step:

```
x_{t+1} = x_t - γ · ∇f_i(x_t)
```

where f_i is the loss on just sample i.

**Why use SGD instead of full gradient descent?**
- Cheaper: computing the gradient on one sample is n times faster than on all samples
- Noise helps: the randomness can help escape local minima
- Often works better in practice

## The Gradient: What Are We Actually Computing?

For least squares, the gradient on sample i is:

```
∇f_i(x) = (1/n) · A_i^T (A_i x - b_i)
```

where A_i is the i-th row of A.

**In English**: Look at how wrong our prediction is on sample i (that's A_i x - b_i), then adjust x in the direction that would fix that error.

## What Are Eigenvalues and Why Do They Matter?

The **Hessian** of our loss function is:

```
H = (1/n) A^T A
```

This is a (d × d) matrix. Its eigenvalues λ_1, λ_2, ..., λ_d tell us about the "shape" of the loss landscape.

**Intuition with a 2D example**:

Imagine the loss function is a bowl. The eigenvalues tell you:
- How steep the bowl is in each direction
- If λ_1 = 100 and λ_2 = 1, the bowl is very steep in one direction and shallow in another (like a long valley)
- If λ_1 = λ_2 = 1, the bowl is perfectly round

**Why this matters for optimization**:

- **Large eigenvalue** = steep direction = gradient descent moves fast there
- **Small eigenvalue** = shallow direction = gradient descent moves slow there
- **Ratio λ_max/λ_min** = condition number = how "stretched" the bowl is = how hard the problem is

## The Critical Learning Rate

There's a maximum learning rate you can use before SGD explodes:

```
γ_max = 2 / λ_max
```

**Why?**

Think about it: in the direction of the largest eigenvalue, the gradient is steepest. If you take too big a step, you'll overshoot and end up further from the minimum than you started. The step size γ times the eigenvalue λ must be less than 2 for convergence.

**The problem**: Computing λ_max requires computing all eigenvalues, which is expensive for large matrices.

## The Volterra Equation: Predicting Loss Over Time

Here's where the magic happens. The Volterra theory says:

The expected loss at time t (measured in epochs) can be computed from:

```
E[f(x_t)] = function of (eigenvalues, γ, t, initial conditions)
```

Specifically, there are "spectral moments":

```
h_k(t) = (1/d) Σ_i λ_i^k · exp(-2γtλ_i)
```

These are weighted sums of exponentially decaying terms, one for each eigenvalue.

**What does this mean intuitively?**

Each eigenvalue contributes a term that decays exponentially. Large eigenvalues decay fast (exp(-2γt·big) goes to zero quickly). Small eigenvalues decay slowly.

The loss at any time t is a combination of these decaying terms. Early in training, all terms contribute. Late in training, only the small-eigenvalue terms remain (explaining why convergence slows down at the end).

## The Marchenko-Pastur Distribution

The theory assumes eigenvalues follow the **Marchenko-Pastur distribution**. This is what you get when:
- Your data matrix A has entries that are independent, identically distributed (i.i.d.)
- The entries are Gaussian (normal distribution)
- The matrix is "tall" or "wide" with aspect ratio r = d/n

The Marchenko-Pastur density is:

```
ρ(λ) = (1/(2πrλ)) · sqrt((λ_+ - λ)(λ - λ_-))
```

where:
- λ_+ = (1 + √r)² is the upper edge
- λ_- = (1 - √r)² is the lower edge

**Key property**: The eigenvalues are BOUNDED. They all lie between λ_- and λ_+.

**The theoretical maximum learning rate** comes from this:

```
γ_theory = 2 / (r · mean(λ)) = 2/r (when mean(λ) = 1)
```

This assumes you're safe as long as you're below the upper edge of the Marchenko-Pastur distribution.

---

# Part 3: Why Random Features on MNIST?

## The Problem with Real Neural Networks

We could test the Volterra theory on a neural network, but that's complicated:
- The features (activations) change during training
- The eigenvalues change during training
- It's hard to isolate what's happening

## The Random Features Trick

Instead, we use **random features**:

```
A = σ(X @ W / √d)
```

where:
- X is our input data (MNIST images, flattened to 784 dimensions)
- W is a RANDOM matrix (fixed, not trained)
- σ is an activation function (we use shifted ReLU)
- The division by √d keeps the scale reasonable

**Why is this useful?**

1. **Fixed features**: W doesn't change, so A doesn't change during training
2. **Controlled setting**: We know exactly what A is
3. **Still realistic**: This is equivalent to the last layer of a neural network with random hidden layer
4. **Theory applies**: The Volterra equation works for fixed feature matrices

## Why MNIST?

We could use synthetic Gaussian data (where the theory is exact), but that's boring. We want to know: **does the theory work on REAL data?**

MNIST is:
- Real images with real structure
- NOT Gaussian (pixels are bounded 0-1, correlated, sparse)
- A standard benchmark everyone knows
- Small enough to run experiments quickly

## The Shifted ReLU Activation

We use:

```
σ(z) = max(0, z) - E[max(0, z)]
```

This is ReLU, but shifted so the output has mean zero.

**Why shift?**

If we don't shift, the columns of A have non-zero mean. This messes up the eigenvalue theory (which assumes centered data). By subtracting the expected value, we center the activation.

**What's E[max(0, z)] for Gaussian z?**

If z ~ N(0, 1), then E[max(0, z)] = 1/√(2π) ≈ 0.399.

## Per-Column Centering

After computing A, we also center each column:

```
A = A - mean(A, axis=0)
```

This ensures each feature has exactly zero mean across samples.

**Why?**

The theory assumes E[A_ij] = 0. Real data might not satisfy this exactly, so we enforce it.

## The Aspect Ratio r = d/n

We run experiments at different aspect ratios:
- r = 0.5: 1500 features, 3000 samples (more samples than features)
- r = 1.0: 3000 features, 3000 samples (square)
- r = 1.5: 4500 features, 3000 samples (more features than samples)

**Why vary r?**

The Marchenko-Pastur distribution depends on r. Different r values give different eigenvalue distributions. We want to test if the theory works across all these cases.

**What happens at r = 1?**

This is special - the matrix is square. At r > 1, the matrix has more columns than rows, so there are (d - n) zero eigenvalues. This is the "interpolation threshold" - the model can fit the training data exactly.

## Planted Targets: b = A @ x*

For most experiments, we use "planted" targets:

```
x* = random unit vector (||x*|| = 1)
b = A @ x*
```

**Why?**

This makes the problem **realizable** - we know for sure that there exists an x (namely x*) such that Ax = b exactly. This lets us focus on the optimization dynamics without worrying about irreducible error.

---

# Part 4: The Base Experiment

## What We Did

**Parameters**:
- n = 3000 MNIST samples
- d = n × r features (1500, 3000, or 4500 depending on r)
- 20 training epochs
- 5 independent runs (to average out randomness)
- Aspect ratios: r ∈ {0.5, 1.0, 1.5}

**For each aspect ratio**:
1. Build random features A
2. Compute eigenvalues of (1/n) A^T A
3. Scale A so mean(eigenvalues) = 1 (for comparability)
4. Generate planted targets b = A @ x*
5. Run SGD at various learning rates
6. Compute Volterra prediction
7. Compare

## The Results: 99% Correlation

Here's what we found:

| Aspect Ratio | Correlation | Spectral Ratio | γ_theory | γ_safe |
|--------------|-------------|----------------|----------|--------|
| r = 0.5 | 98.99% | 133.2 | 4.0 | 0.015 |
| r = 1.0 | 99.04% | 270.6 | 2.0 | 0.007 |
| r = 1.5 | 98.44% | 272.4 | 1.3 | 0.007 |

**What does "99% correlation" mean?**

If we plot Volterra's prediction against actual SGD loss at every time point, the points lie almost perfectly on a straight line. The theory predicts real SGD behavior almost exactly.

## The Spectral Ratio: Why Theory ≠ Practice

Look at the "Spectral Ratio" column: 133 to 272.

This is λ_max / mean(λ). It tells us how much bigger the largest eigenvalue is compared to the average.

**Why is this so large?**

Remember, Marchenko-Pastur says eigenvalues should be bounded by (1 + √r)² ≈ 2.4 to 6 for our r values. But we're seeing eigenvalues of 133-272!

**The reason**: MNIST is NOT Gaussian.

MNIST images have:
1. **Spatial correlation**: Neighboring pixels are correlated (edges, strokes)
2. **Sparsity**: ~75% of pixels are exactly 0 (black background)
3. **Low intrinsic dimension**: Digit images lie on a ~50D manifold in 784D space
4. **Bounded values**: Pixels are in [0, 1], not unbounded Gaussian

This creates a "spiked" eigenvalue distribution:
- Most eigenvalues are tiny (corresponding to noise directions)
- A few eigenvalues are huge (corresponding to dominant patterns like edges)

## The Theory-Practice Gap

**Theory says**: Use γ ≤ 2/r ≈ 4.0 (for r = 0.5)

**Practice requires**: Use γ ≤ 2/λ_max ≈ 0.015

**The gap**: 4.0 / 0.015 = 267x!

**Why?**

The theory assumes all eigenvalues are below (1 + √r)² ≈ 2.4. But our largest eigenvalue is 133. If we use γ = 4.0, the update in the largest-eigenvalue direction would be:

```
step = γ × λ_max = 4.0 × 133 = 532
```

This is way bigger than 2, so we'd diverge immediately.

**Intuition**: Imagine 2999 eigenvalues at 1 and one eigenvalue at 133. The mean is ~1.04, but that one outlier ruins everything. You can't use a learning rate appropriate for the "average" eigenvalue - you must respect the maximum.

## Why Does Correlation Improve with More Data?

We ran test experiments with n = 2000 and got ~95% correlation. With n = 3000, we got ~99%.

**Why?**

The Volterra theory predicts the EXPECTED loss - the average over all possible random initializations and sample orderings. With more data and more runs:
- The empirical average converges to the true expectation (Law of Large Numbers)
- Eigenvalue estimates become more accurate
- Finite-sample noise decreases

The theory is **asymptotically exact** as n, d → ∞ with fixed ratio r. Larger n gets us closer to this asymptotic regime.

---

# Part 5: Non-Linear Targets (Extension 1)

## The Question

So far, our targets were b = A @ x*, which is a LINEAR function of A. This guarantees the problem is realizable.

**What if we make the targets non-linear?**

Specifically, what if:

```
b = ReLU(A @ x*)
```

This applies ReLU to the linear targets, clipping negative values to zero.

## Why This Matters

This creates a **non-realizable** problem. There's no x such that Ax = ReLU(A @ x*) exactly, because:
- The left side (Ax) is linear in the columns of A
- The right side (ReLU(...)) is non-linear
- You can't match a non-linear function with a linear combination

This tests: does Volterra theory still work when the problem can't be solved exactly?

## The Results: ~50% Zero Targets

| Aspect Ratio | Zero Fraction | Final Loss | Initial Loss |
|--------------|---------------|------------|--------------|
| r = 0.5 | 52.57% | 0.168 | 0.364 |
| r = 1.0 | 49.03% | 0.135 | 0.242 |
| r = 1.5 | 48.40% | 0.060 | 0.098 |

About half of the targets are zero!

## Why Exactly 50%?

This is beautiful mathematics:

**Step 1**: What is (A @ x*)_i?

It's a sum of d random terms:
```
(A @ x*)_i = Σ_j A_ij × (x*)_j
```

**Step 2**: What distribution does this sum follow?

By the **Central Limit Theorem**, the sum of many random variables approaches a Gaussian distribution.

**Step 3**: What's the mean of this Gaussian?

Since we centered A (each column has mean 0) and x* is symmetric around 0, the mean is:
```
E[(A @ x*)_i] = Σ_j E[A_ij] × (x*)_j = 0
```

**Step 4**: What's the probability that a mean-zero Gaussian is negative?

For any symmetric distribution centered at 0:
```
P(X < 0) = P(X > 0) = 0.5
```

**Step 5**: What does ReLU do to negative values?

It sets them to zero.

**Conclusion**: ReLU(A @ x*) has about 50% zeros because A @ x* is approximately Gaussian with mean zero, and ReLU clips exactly the negative half.

## Why Does Zero Fraction Vary Slightly?

We observed 52.57%, 49.03%, 48.40% for r = 0.5, 1.0, 1.5.

The Central Limit Theorem is asymptotic - it's exact only as d → ∞. With finite d:
- Smaller d (r = 0.5 has d = 1500) → more variance in the distribution → slightly more deviation from 50%
- Larger d (r = 1.5 has d = 4500) → CLT applies better → closer to exactly 50%

## The Irreducible Error Floor

Unlike the base experiment (where loss goes to zero), the non-linear experiment has **non-zero final loss**.

**Why can't we reach zero loss?**

The problem is non-realizable. Even the best possible x can only make Ax approximate ReLU(A @ x*), not match it exactly.

**Mathematical decomposition**:

```
Final Loss = Realizable Error + Irreducible Error
```

- **Realizable Error**: Goes to zero with training (SGD finds the best x)
- **Irreducible Error**: Stays forever (fundamental limitation of linear models)

## Why Does Final Loss Decrease with r?

| r | Final Loss |
|---|------------|
| 0.5 | 0.168 |
| 1.0 | 0.135 |
| 1.5 | 0.060 |

Higher r → lower final loss. Why?

**The column space of A expands.**

At r = 0.5, A has 1500 columns in 3000-dimensional space. The column space (all possible Ax) is at most 1500-dimensional.

At r = 1.5, A has 4500 columns. The column space can span all of 3000-dimensional space (since we have more columns than rows).

**Better approximation**:

The irreducible error is the distance from b to the column space of A:
```
Irreducible Error = ||b - projection of b onto column space of A||²
```

Larger column space → b can be approximated better → smaller irreducible error.

## Does Volterra Still Work?

Yes! The Volterra equation predicts the DYNAMICS (how loss decreases over time), not the ENDPOINT (what final loss will be).

The eigenvalues of A^T A don't depend on the targets b. They only depend on the features A. Since we're using the same A, the same eigenvalue-based dynamics apply.

What changes is:
- The initial loss (different for non-linear targets)
- The final loss (non-zero for non-linear targets)

But the rate of convergence is still determined by eigenvalues.

---

# Part 6: Whitened MNIST (Extension 2)

## The Question

We saw that MNIST has spectral ratio 133-272 because of its non-Gaussian structure. Can we preprocess the data to reduce this?

**Whitening** is a classic preprocessing technique that makes data "more Gaussian-like."

## What is Whitening?

Whitening transforms the data so that:
1. Mean is zero: E[X] = 0
2. Covariance is identity: E[XX^T] = I

The transform is:
```
X_whitened = Σ^{-1/2} (X - μ)
```

where:
- μ = mean(X) is the data mean
- Σ = covariance(X) is the data covariance
- Σ^{-1/2} is the inverse square root of the covariance

**In words**: Subtract the mean, then rotate and scale so all directions have equal variance.

## Why Should This Help?

The Volterra theory assumes Gaussian data. Gaussian data has:
- Zero mean ✓ (we can always center)
- Identity covariance ✓ (whitening achieves this)
- Gaussian marginals ✗ (whitening doesn't fix this, but it helps)

By whitening, we make the data satisfy 2 out of 3 Gaussian properties. This should bring the eigenvalue distribution closer to Marchenko-Pastur.

## The MNIST Covariance is Extremely Non-Isotropic

Before whitening:
```
Condition number of covariance: 51,396,544
Eigenvalue range: [~0, 250]
```

**51 million!** This means the largest variance direction has 51 million times more variance than the smallest.

**Why so extreme?**

Most MNIST pixels are always black (background). These have near-zero variance. A few pixels (the digit region) have high variance. This creates a huge spread.

## The Results: 74% Reduction in Spectral Ratio

| Aspect Ratio | Non-Whitened | Whitened | Reduction |
|--------------|--------------|----------|-----------|
| r = 0.5 | 133.2 | 36.3 | 73% |
| r = 1.0 | 270.6 | 71.1 | 74% |
| r = 1.5 | 272.4 | 70.3 | 74% |

Spectral ratio drops by about 3-4x!

## Why Does Whitening Reduce Spectral Ratio?

**The causal chain**:

1. **Raw MNIST**: Covariance has eigenvalues from 0 to 250
2. **Random projection**: A = σ(X @ W) inherits this spread
3. **A^T A**: Has eigenvalues spanning a huge range
4. **Spectral ratio**: λ_max / λ_mean is large

After whitening:

1. **Whitened MNIST**: Covariance is identity (all eigenvalues = 1)
2. **Random projection**: A = σ(X_whitened @ W) starts from isotropic data
3. **A^T A**: Has eigenvalues with smaller spread
4. **Spectral ratio**: Much smaller

## Why Isn't Spectral Ratio = 1 After Whitening?

We got 36-71, not 1. Why?

**The ReLU non-linearity breaks isotropy.**

Even if X_whitened is perfectly isotropic, applying ReLU creates:
- Asymmetry (negative values become zero)
- Non-Gaussian distribution (half the values are exactly zero)
- Correlations between features

**The Marchenko-Pastur bound**:

Even for truly Gaussian data, the spectral ratio is bounded by:
```
(1 + √r)² / 1 ≈ 2.4 to 6 for our r values
```

We're getting 36-71, which is still 10x larger than this theoretical bound. The ReLU activation is responsible.

## Practical Implication: 4x Larger Learning Rates

| r | γ_safe (Non-whitened) | γ_safe (Whitened) | Improvement |
|---|----------------------|-------------------|-------------|
| 0.5 | 0.015 | 0.055 | 3.7x |
| 1.0 | 0.007 | 0.028 | 3.8x |
| 1.5 | 0.007 | 0.028 | 3.9x |

By whitening, you can use ~4x larger learning rates safely!

This means:
- 4x faster training (roughly)
- Fewer hyperparameter tuning iterations
- More stable optimization

---

# Part 7: Real MNIST Labels (Extension 3)

## The Question

Everything so far used planted targets (b = A @ x*). These are synthetic - not what you'd use in practice.

**What if we use REAL MNIST labels?**

This tests whether Volterra theory applies to actual classification tasks.

## Classification as Regression

MNIST has 10 classes (digits 0-9). We convert this to regression using **one-hot encoding**:

```
label = 3 → target = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
```

The full target matrix Y is (n × 10), with exactly one 1 per row.

**The optimization problem**:

```
minimize (1/2n) ||A @ X - Y||_F²
```

where X is (d × 10) - one column per class.

**Classification rule**:

```
predicted_class = argmax_k (A @ X)[i, k]
```

Pick the class with the highest regression output.

## Why Does This Work?

One-hot targets create a "winner-take-all" structure:
- The correct class should have output ≈ 1
- Incorrect classes should have output ≈ 0

Even if the outputs aren't exactly 0/1, as long as the correct class has the HIGHEST output, classification is correct.

## The Results: Double Descent!

| Aspect Ratio | Accuracy | Spectral Ratio | R Value |
|--------------|----------|----------------|---------|
| r = 0.5 | 99.95% | 88.2 | 0.28 |
| r = 1.0 | 73.35% | 170.0 | unstable |
| r = 1.5 | 100.00% | 167.4 | unstable |

**Wait, why does accuracy DIP at r = 1.0?**

This is the famous **double descent** phenomenon!

## Understanding Double Descent

**At r = 0.5 (underdetermined)**:
- 1000 features, 2000 samples
- System has unique least-squares solution
- Well-conditioned, stable
- Accuracy: 99.95%

**At r = 1.0 (exactly determined)**:
- 2000 features, 2000 samples
- Matrix A is SQUARE
- May be singular or nearly singular
- Numerical instability
- Accuracy: 73.35% (dip!)

**At r = 1.5 (overdetermined)**:
- 3000 features, 2000 samples
- System has infinitely many solutions
- We find the minimum-norm solution
- Can interpolate (fit training data exactly)
- Accuracy: 100%

## Why Does r = 1.0 Fail?

At r = 1, the matrix A is square. Its determinant may be zero or nearly zero.

**What happens mathematically**:

```
X_optimal = (A^T A)^{-1} A^T Y
```

If A^T A is singular, this inverse doesn't exist. If A^T A is nearly singular, the inverse is huge and numerically unstable.

**The R parameter explodes**:

We compute R = ||X_optimal||² / 10. At r = 1.0, we got R ≈ 10^12. This is numerical garbage - the solution is unstable.

## Why Does r = 1.5 Work Perfectly?

At r > 1, there are infinitely many solutions. The lstsq function returns the **minimum-norm** solution:

```
X_optimal = A^+ Y
```

where A^+ is the Moore-Penrose pseudoinverse.

This solution:
- Is well-defined (even for rank-deficient A)
- Has the smallest norm among all solutions
- Can fit the training data exactly (if rank(A) ≥ n)

**Result**: 100% training accuracy because the model interpolates.

## The R Parameter for Real Labels

For planted targets, R = ||x*||² = 1 (we normalized x*).

For real labels, R must be computed:

```
R = (1/10) ||X_optimal||_F²
```

This is the average squared norm of the optimal solution across the 10 classes.

**The problem**: At r ≈ 1, X_optimal is unstable, so R is meaningless. The Volterra theory technically still applies, but with garbage R, predictions are unreliable.

---

# Part 8: Connecting Everything Together

## The Central Theme: Eigenvalues Determine Everything

Across all four experiments, one theme emerges:

**The eigenvalues of A^T A determine the optimization dynamics.**

- **Base experiment**: 99% correlation between Volterra (eigenvalue-based) prediction and actual SGD
- **Non-linear targets**: Same eigenvalues, same dynamics, just different endpoint
- **Whitened data**: Changed eigenvalues (smaller spread), changed dynamics (faster convergence possible)
- **Real labels**: Same eigenvalues (target doesn't affect them), theory still applies

## The Spectral Ratio: The Key Quantity

If you remember one number from this project, remember the **spectral ratio**:

```
Spectral Ratio = λ_max / mean(λ)
```

This tells you:
1. **How hard the optimization is**: Higher ratio = harder (more ill-conditioned)
2. **The theory-practice gap**: Ratio ≈ γ_theory / γ_safe
3. **Whether preprocessing helps**: Whitening reduced it by 74%

## Why Theory ≠ Practice

The Volterra theory assumes:
1. Gaussian data
2. Eigenvalues follow Marchenko-Pastur
3. Bounded eigenvalue support

Real data violates all three:
1. MNIST is highly non-Gaussian
2. Eigenvalues are spiked, not Marchenko-Pastur
3. Outlier eigenvalues far exceed the theoretical bound

**But the theory still predicts dynamics accurately!** The 99% correlation shows that the functional form of the Volterra equation is correct - only the parameters (like critical step size) need adjustment.

## The Three Extensions: What They Teach Us

**Extension 1 (Non-linear targets)**: Theory is robust to non-realizability
- Dynamics depend on eigenvalues, not targets
- Non-realizable problems have irreducible error, but convergence is still predictable

**Extension 2 (Whitening)**: Preprocessing can bridge theory and practice
- Whitening makes data more Gaussian-like
- Spectral ratio decreases
- Larger learning rates become safe
- Theory predictions become more accurate

**Extension 3 (Real labels)**: Theory extends to classification
- Multi-output regression follows the same dynamics
- The R parameter must be computed from data
- Beware the interpolation threshold (r ≈ 1)

## The Hierarchy of What Matters

From most to least important for predicting SGD behavior:

1. **Eigenvalue distribution** (determines everything)
2. **Learning rate** (must be < 2/λ_max)
3. **Initial conditions** (affects initial loss, not dynamics)
4. **Target choice** (affects endpoint, not dynamics)
5. **Data distribution** (affects eigenvalues, which then affect dynamics)

---

# Part 9: What We Learned

## For the Theorist

1. **Volterra equation works on real data** with 99% accuracy, not just synthetic Gaussians

2. **The Gaussian assumption matters for step sizes**, not dynamics. The functional form is correct; only the critical step size needs λ_max instead of theoretical bounds.

3. **Eigenvalue outliers dominate**. Even one large eigenvalue can determine the maximum safe learning rate.

4. **Non-realizability doesn't break the theory**. The dynamics are target-independent.

## For the Practitioner

1. **Always compute λ_max** (or at least estimate it). Don't trust theoretical step size formulas.

2. **Preprocess your data**. Whitening or standardization can give you 4x larger learning rates for free.

3. **Avoid r ≈ 1**. The interpolation threshold causes numerical issues. Use r < 0.8 or r > 1.2.

4. **Random features are surprisingly powerful**. We achieved 100% MNIST training accuracy with just 3000 random features.

## For the Student

1. **Eigenvalues encode the optimization landscape**. Understanding linear algebra helps understand deep learning.

2. **Theory and practice can meet**. The 99% correlation shows that mathematical theory can predict real-world behavior.

3. **Assumptions matter but aren't everything**. MNIST violates Gaussian assumptions badly, but the theory still works (with adjusted parameters).

4. **Simple models reveal insights**. We learned about spectral ratios, double descent, and preprocessing by studying least squares - the simplest possible problem.

## Open Questions

1. **Can we extend to non-linear models?** The Volterra equation assumes fixed features. What about trained neural networks?

2. **What about mini-batch SGD?** We used single-sample SGD. How do larger batches change the dynamics?

3. **Can we predict generalization?** We predicted training loss. What about test loss?

4. **Optimal preprocessing?** Whitening helps, but is there something even better?

---

# Appendix: The Experiments at a Glance

## Experiment 1: Base

**Question**: Does Volterra theory predict SGD dynamics on real data?

**Setup**:
- A = shifted_ReLU(MNIST @ W), b = A @ x*
- n = 3000, r ∈ {0.5, 1.0, 1.5}

**Result**: 99% correlation

**Key insight**: Eigenvalues determine SGD dynamics

---

## Experiment 2: Non-Linear Targets

**Question**: What happens when the problem isn't realizable?

**Setup**:
- Same A, but b = ReLU(A @ x*)

**Result**: ~50% zero targets, non-zero final loss, theory still predicts dynamics

**Key insight**: ReLU clips exactly half (CLT + symmetry), irreducible error exists

---

## Experiment 3: Whitened MNIST

**Question**: Can preprocessing reduce the theory-practice gap?

**Setup**:
- X_whitened = Σ^{-1/2}(X - μ), then same A construction

**Result**: 74% reduction in spectral ratio, 4x larger safe learning rates

**Key insight**: Whitening makes data more isotropic, improving conditioning

---

## Experiment 4: Real MNIST Labels

**Question**: Does theory work for actual classification?

**Setup**:
- Y = one_hot(labels), 10-output regression

**Result**: 99.95% accuracy (r=0.5), 73% (r=1.0, double descent!), 100% (r=1.5)

**Key insight**: Theory extends to multi-output, but r=1 is numerically unstable

---

# Final Thoughts

This project demonstrates that beautiful mathematical theory (the Volterra Integral Equation) can accurately predict the behavior of practical algorithms (SGD) on real data (MNIST).

The key is understanding what the theory assumes, where those assumptions fail, and how to adjust. The eigenvalue spectrum is the bridge between theory and practice - it encodes everything about the optimization landscape, and once you have it, you can predict training dynamics with remarkable accuracy.

Whether you're a theorist trying to understand why neural networks train, or a practitioner trying to choose learning rates, the message is the same: **look at the eigenvalues**.
