# Experiment: Whitened MNIST Data

## What This Tests
Tests whether Volterra theory works better when MNIST data satisfies Gaussian assumptions (E[x]=0, Cov=I).

## Key Difference from run_part2.py
- **Original:** Raw MNIST X (after standard normalization)
- **This experiment:** Whitened MNIST using `X_whitened = Sigma^{-1/2}(X - mu)`

Whitening transforms data so:
- E[x] = 0 (zero mean)
- E[xx^T] = I (identity covariance)

## How to Run
```bash
python run_part2_whitened.py
```

## Expected Output
- `results/part2_whitened/all_results.npy`
- Console shows before/after whitening statistics
- Spectral ratio should decrease (eigenvalues more uniform)

## Interpretation
- **Spectral ratio:** Should be much smaller than non-whitened (e.g., ~10x vs ~100x)
- **Theory accuracy:** Should improve since data is closer to Gaussian assumption
- **Key insight:** Whitening reduces condition number, making step-size selection easier

## Theoretical Background
The Volterra theory is derived under Gaussian data assumptions. MNIST violates this heavily (sparse, bounded pixels). Whitening makes the covariance structure closer to identity, though higher-order moments still differ from Gaussian.

## Whitening Formula
```
mu = E[X]                           # Mean
Sigma = E[(X - mu)(X - mu)^T]       # Covariance
X_whitened = Sigma^{-1/2} (X - mu)  # Whitened data
```
