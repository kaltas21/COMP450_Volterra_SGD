# Experiment: Real MNIST Labels (Classification as Regression)

## What This Tests
Tests whether Volterra theory predicts SGD dynamics on actual MNIST classification task using one-hot encoded labels.

## Key Difference from run_part2.py
- **Original:** Synthetic planted targets `b = A @ x_star` (n x 1)
- **This experiment:** One-hot MNIST labels `Y` (n x 10)

This is a 10-dimensional regression problem:
```
min (1/2n) ||AX - Y||_F^2
```
where X is (d x 10) parameter matrix and Y is one-hot encoded labels.

## How to Run
```bash
python run_part2_real_labels.py
```

## Expected Output
- `results/part2_real_labels/all_results.npy`
- Console shows R value (computed from optimal solution)
- Loss curves for 10-output regression

## Interpretation
- **R value:** Computed as `(1/k) * ||X_opt||_F^2` where X_opt is least squares solution
- **Theory accuracy:** Tests if eigenvalue-based prediction works for real tasks
- **Key insight:** If theory works here, it can predict actual training dynamics

## Theoretical Background
The multi-output problem decomposes into k=10 independent regression problems sharing the same feature matrix A. The Volterra equation predicts the average loss trajectory. Each output dimension evolves independently under SGD.

## One-Hot Encoding
```
Label 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
Label 7 -> [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
```
This converts classification into regression with 10 output dimensions.

## Multi-Output SGD
Uses `MultiOutputLeastSquaresSGD` class which handles matrix targets B (n x k) instead of vector targets b (n x 1).
