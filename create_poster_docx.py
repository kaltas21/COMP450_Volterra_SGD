"""
Create a Word document poster with all figures and findings
"""
from docx import Document
from docx.shared import Inches, Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
import os

doc = Document()

# Set up styles
style = doc.styles['Normal']
style.font.name = 'Arial'
style.font.size = Pt(11)

# Title
title = doc.add_heading('Predicting SGD Dynamics via Volterra Integral Equations', 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

# Subtitle
subtitle = doc.add_paragraph('COMP450 - Learning Theory from First Principles')
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER

# Authors
authors = doc.add_paragraph('Yigit Kaltas, Kaan')
authors.alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph()

# Abstract
doc.add_heading('Abstract', level=1)
doc.add_paragraph(
    'We validate that the Volterra Integral Equation predicts Stochastic Gradient Descent (SGD) '
    'dynamics with 99% accuracy on MNIST random features. The eigenvalue spectrum of the feature '
    'matrix alone determines training behavior. We demonstrate theory robustness through three '
    'extensions: non-linear targets, whitened data, and real classification labels.'
)

# 1. Introduction
doc.add_heading('1. Introduction & Motivation', level=1)
doc.add_paragraph(
    'Central Question: Can we predict SGD loss curves from eigenvalues alone?\n\n'
    'For least squares f(x) = (1/2n)||Ax - b||², the Volterra theory predicts that loss evolution '
    'depends only on eigenvalues of A^T A. The key theoretical prediction is that the critical '
    'step size is γ_max = 2/(r × mean(λ)), where r = d/n is the aspect ratio.'
)

# 2. Experimental Setup
doc.add_heading('2. Experimental Setup', level=1)
doc.add_paragraph(
    'Data: MNIST (n = 3000 samples, 784 pixels)\n\n'
    'Random Features: A = shifted_ReLU(X @ W / √d)\n'
    '• W: Random Gaussian weights (fixed, not trained)\n'
    '• Per-column centering applied\n\n'
    'Aspect Ratios: r = d/n ∈ {0.5, 1.0, 1.5}\n\n'
    'Four Experiments:\n'
    '1. Base: Planted targets b = A @ x*\n'
    '2. Non-linear: b = ReLU(A @ x*)\n'
    '3. Whitened: Whitening transform on X before features\n'
    '4. Real Labels: Y = one_hot(MNIST labels)'
)

# 3. Experiment 1: Base
doc.add_heading('3. Experiment 1: Base Validation', level=1)
doc.add_paragraph('Result: 99% Correlation Between Volterra Prediction and SGD')

# Table 1
table1 = doc.add_table(rows=4, cols=5)
table1.style = 'Table Grid'
headers = ['Aspect Ratio', 'Correlation', 'Spectral Ratio', 'γ_theory', 'γ_safe']
for i, header in enumerate(headers):
    table1.rows[0].cells[i].text = header
data1 = [
    ['r = 0.5', '98.99%', '133.2', '4.00', '0.015'],
    ['r = 1.0', '99.04%', '270.6', '2.00', '0.007'],
    ['r = 1.5', '98.44%', '272.4', '1.33', '0.007']
]
for i, row in enumerate(data1):
    for j, val in enumerate(row):
        table1.rows[i+1].cells[j].text = val

doc.add_paragraph()

# Figure 1
if os.path.exists('results/full_run_base/loss_curves.png'):
    doc.add_paragraph('Figure 1: Loss Curves - Volterra Prediction vs Actual SGD')
    doc.add_picture('results/full_run_base/loss_curves.png', width=Inches(6))
    doc.add_paragraph()

if os.path.exists('results/full_run_base/eigenvalue_spectrum.png'):
    doc.add_paragraph('Figure 2: Eigenvalue Spectrum vs Marchenko-Pastur Distribution')
    doc.add_picture('results/full_run_base/eigenvalue_spectrum.png', width=Inches(6))
    doc.add_paragraph()

doc.add_paragraph(
    'Key Finding: Theory-Practice Gap = 180-270x\n\n'
    'Why? MNIST violates Gaussian assumptions:\n'
    '• Spatial correlation between pixels\n'
    '• Sparse data (~75% zeros)\n'
    '• Bounded values [0,1]\n\n'
    'This creates a spiked eigenvalue distribution with outliers far exceeding Marchenko-Pastur bounds.'
)

# 4. Experiment 2: Non-linear
doc.add_heading('4. Experiment 2: Non-Linear Targets', level=1)
doc.add_paragraph('Setup: b = ReLU(A @ x*) instead of b = A @ x*')
doc.add_paragraph('Result: ~50% of Targets Become Zero')

table2 = doc.add_table(rows=4, cols=4)
table2.style = 'Table Grid'
headers2 = ['Aspect Ratio', 'Zero Fraction', 'Final Loss', 'Initial Loss']
for i, header in enumerate(headers2):
    table2.rows[0].cells[i].text = header
data2 = [
    ['r = 0.5', '52.57%', '0.168', '0.364'],
    ['r = 1.0', '49.03%', '0.135', '0.242'],
    ['r = 1.5', '48.40%', '0.060', '0.098']
]
for i, row in enumerate(data2):
    for j, val in enumerate(row):
        table2.rows[i+1].cells[j].text = val

doc.add_paragraph()

if os.path.exists('results/full_run_nonlinear/loss_curves.png'):
    doc.add_paragraph('Figure 3: Non-Linear Target Loss Curves')
    doc.add_picture('results/full_run_nonlinear/loss_curves.png', width=Inches(6))
    doc.add_paragraph()

if os.path.exists('results/full_run_nonlinear/zero_fraction.png'):
    doc.add_paragraph('Figure 4: Zero Fraction Analysis')
    doc.add_picture('results/full_run_nonlinear/zero_fraction.png', width=Inches(6))
    doc.add_paragraph()

doc.add_paragraph(
    'Why Exactly 50%?\n'
    '1. A has centered columns → E[A_ij] = 0\n'
    '2. By Central Limit Theorem: (A @ x*)_i ~ N(0, σ²)\n'
    '3. For symmetric distribution: P(X < 0) = 0.5\n'
    '4. ReLU clips negative half to zero\n\n'
    'Key Finding: Irreducible Error Floor\n'
    'Non-realizable problem → loss cannot reach zero. Higher r reduces error by expanding column space of A.'
)

# 5. Experiment 3: Whitened
doc.add_heading('5. Experiment 3: Whitened MNIST', level=1)
doc.add_paragraph('Setup: Apply whitening transform X_whitened = Σ^(-1/2)(X - μ)')
doc.add_paragraph('Result: 74% Reduction in Spectral Ratio')

table3 = doc.add_table(rows=4, cols=4)
table3.style = 'Table Grid'
headers3 = ['Aspect Ratio', 'Non-Whitened', 'Whitened', 'Reduction']
for i, header in enumerate(headers3):
    table3.rows[0].cells[i].text = header
data3 = [
    ['r = 0.5', '133.2', '36.3', '73%'],
    ['r = 1.0', '270.6', '71.1', '74%'],
    ['r = 1.5', '272.4', '70.3', '74%']
]
for i, row in enumerate(data3):
    for j, val in enumerate(row):
        table3.rows[i+1].cells[j].text = val

doc.add_paragraph()

if os.path.exists('results/quick_whitened/summary_comparison.png'):
    doc.add_paragraph('Figure 5: Whitened vs Non-Whitened Spectral Ratio Comparison')
    doc.add_picture('results/quick_whitened/summary_comparison.png', width=Inches(6))
    doc.add_paragraph()

doc.add_paragraph(
    'Why Does Whitening Help?\n'
    '• Original MNIST covariance condition number: 51,396,544\n'
    '• After whitening: All directions have equal variance\n'
    '• Random projection preserves isotropy better\n'
    '• Result: 3-4x larger safe learning rates'
)

# 6. Experiment 4: Real Labels
doc.add_heading('6. Experiment 4: Real MNIST Labels', level=1)
doc.add_paragraph('Setup: Classification as 10-output regression with Y = one_hot(MNIST labels)')
doc.add_paragraph('Result: Double Descent Phenomenon')

table4 = doc.add_table(rows=4, cols=4)
table4.style = 'Table Grid'
headers4 = ['Aspect Ratio', 'Accuracy', 'Spectral Ratio', 'R Value']
for i, header in enumerate(headers4):
    table4.rows[0].cells[i].text = header
data4 = [
    ['r = 0.5', '99.95%', '88.2', '0.28'],
    ['r = 1.0', '73.35%', '170.0', 'unstable'],
    ['r = 1.5', '100.00%', '167.4', 'unstable']
]
for i, row in enumerate(data4):
    for j, val in enumerate(row):
        table4.rows[i+1].cells[j].text = val

doc.add_paragraph()

if os.path.exists('results/quick_real_labels/accuracy_summary.png'):
    doc.add_paragraph('Figure 6: Classification Accuracy vs Aspect Ratio')
    doc.add_picture('results/quick_real_labels/accuracy_summary.png', width=Inches(6))
    doc.add_paragraph()

doc.add_paragraph(
    'Why Does Accuracy Dip at r = 1.0? (Double Descent)\n'
    '• At r = 1: Matrix A is square (potentially singular)\n'
    '• Interpolation threshold causes numerical instability\n'
    '• R parameter explodes (~10^12)\n\n'
    'Why 100% at r = 1.5?\n'
    '• More features than samples → interpolation regime\n'
    '• Model can fit training data exactly'
)

# 7. Key Findings
doc.add_heading('7. Key Findings', level=1)
doc.add_paragraph(
    '1. Eigenvalues Determine SGD Dynamics\n'
    '   99% correlation between Volterra prediction and actual SGD.\n\n'
    '2. Spectral Ratio Explains Theory-Practice Gap\n'
    '   Gap ≈ λ_max / mean(λ) = Spectral Ratio (133-272x for MNIST)\n\n'
    '3. Theory is Robust to Assumption Violations\n'
    '   Works for non-realizable problems and multi-output regression.\n\n'
    '4. Preprocessing Matters\n'
    '   Whitening reduces spectral ratio by 74%, allowing 4x larger learning rates.\n\n'
    '5. Interpolation Threshold is Special\n'
    '   At r = 1: numerical instability, double descent. Avoid r ≈ 1 in practice.'
)

# 8. Conclusions
doc.add_heading('8. Conclusions', level=1)
doc.add_paragraph(
    '• Volterra theory accurately predicts SGD dynamics on real data (99% correlation)\n\n'
    '• The spectral ratio is the key quantity bridging theory and practice\n\n'
    '• The theory is robust: works for non-realizable problems, extends to classification\n\n'
    '• Practical recommendations:\n'
    '  - Always compute λ_max for step size selection\n'
    '  - Use whitening/standardization (4x speedup possible)\n'
    '  - Avoid aspect ratios near r = 1'
)

# References
doc.add_heading('References', level=1)
doc.add_paragraph(
    '1. Bach, F. (2024). Learning Theory from First Principles. MIT Press.\n'
    '2. Marchenko, V. A., & Pastur, L. A. (1967). Distribution of eigenvalues for some sets of random matrices.'
)

# Save
doc.save('docs/POSTER.docx')
print("Poster saved to docs/POSTER.docx")
