
import numpy as np
from logistic_regression import LogisticRegressionScratch

np.random.seed(42)

# Generate synthetic loan-default-like dataset
n_samples = 500
income = np.random.normal(50000, 15000, n_samples)
debt = np.random.normal(15000, 5000, n_samples)
credit_score = np.random.normal(650, 50, n_samples)

X = np.column_stack([income, debt, credit_score])
y = ((0.00004*debt - 0.00003*income - 0.002*credit_score + np.random.normal(0,0.05,n_samples)) > -0.5).astype(int)

# Normalize manually (no sklearn)
X = (X - X.mean(axis=0)) / X.std(axis=0)

model = LogisticRegressionScratch(lr=0.01, n_iters=8000)
model.fit(X, y)

print("Weights:", model.weights)
print("Bias:", model.bias)
