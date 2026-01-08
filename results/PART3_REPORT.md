# Part 3: Step-Size Criticality Analysis

## Experiment Summary
- Dataset: MNIST (n=2000)
- Features: Random Features (ReLU)
- Definition of Critical Step Size: Largest $\gamma$ where loss remains finite and $\le 10 \times$ initial loss.

## Results by Regime

### Aspect Ratio r = 0.5
- **Spectral Ratio** ($\lambda_{max} / \overline{\lambda}$): 88.22
- **Theoretical Limit** ($\gamma_{theory}$): 4.0000
- **Safe Limit** ($\gamma_{safe}$): 0.0227
- **Empirical Critical Limit**: 0.0453
- **Gap**: Theory predicts step size 88.2x larger than possible.

### Aspect Ratio r = 1.0
- **Spectral Ratio** ($\lambda_{max} / \overline{\lambda}$): 169.95
- **Theoretical Limit** ($\gamma_{theory}$): 2.0000
- **Safe Limit** ($\gamma_{safe}$): 0.0118
- **Empirical Critical Limit**: 0.0235
- **Gap**: Theory predicts step size 85.0x larger than possible.

### Aspect Ratio r = 1.5
- **Spectral Ratio** ($\lambda_{max} / \overline{\lambda}$): 167.37
- **Theoretical Limit** ($\gamma_{theory}$): 1.3333
- **Safe Limit** ($\gamma_{safe}$): 0.0119
- **Empirical Critical Limit**: 0.0239
- **Gap**: Theory predicts step size 55.8x larger than possible.

## Interpretation

1. **Spectral Ratio**: The large ratio between $\lambda_{max}$ and mean($\lambda$) explains early divergence. Volterra theory (based on mean) overestimates stability for non-isotropic data.
2. **Theory vs Practice**: $\gamma_{theory}$ assumes isotropic features. Real data has 'outlier' eigenvalues that dictate the true stability limit ($\gamma_{safe} \approx 2/\lambda_{max}$).
3. **Mismatch**: The mismatch is not a model failure but a violation of the asymptotic assumptions (specifically, that spectrum is compact).
