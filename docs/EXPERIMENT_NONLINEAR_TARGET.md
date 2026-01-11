# Experiment: Non-Linear Target (ReLU on Planted Targets)

## What This Tests
Tests whether Volterra theory predicts SGD dynamics when targets are non-linear (non-realizable problem).

## Key Difference from run_part2.py
- **Original:** `b = A @ x_star` (linear, realizable)
- **This experiment:** `b = ReLU(A @ x_star)` (non-linear, non-realizable)

The ReLU clips negative values to zero, creating targets that cannot be perfectly fit by any linear combination of features.

## How to Run
```bash
python run_part2_nonlinear_target.py
```

## Expected Output
- `results/part2_nonlinear_target/all_results.npy`
- Console output shows "zero fraction" - percentage of targets clipped by ReLU

## Interpretation
- **Zero fraction:** ~50% of targets become zero (half of Gaussian is negative)
- **Theory accuracy:** May decrease since Volterra assumes realizable problem
- **Key insight:** Tests robustness of theory to model mismatch

## Theoretical Background
The Volterra integral equation assumes a planted model where `b = A @ x_star` exactly. By applying ReLU, we violate this assumption. The experiment reveals how sensitive the theory is to this violation.
