
from logistic_regression import LogisticRegressionScratch
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X, y = data.data, data.target

model = LogisticRegressionScratch(lr=0.001, n_iter=8000)
model.fit(X, y)

theta = model.get_params()

print("\nInterpreting Learned Coefficients:")
print("----------------------------------")

for feature, weight in zip(data.feature_names, theta[1:]):
    direction = "↑ increases likelihood of class 1" if weight > 0 else "↓ decreases likelihood of class 1"
    print(f"{feature}: {weight:.4f}  →  {direction}")

print("\nBias term (θ₀):", theta[0])
