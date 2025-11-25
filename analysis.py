import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

def generate_data():
    X, y = make_classification(
        n_samples=500,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=42
    )
    feature_names = [f"feature_{i+1}" for i in range(5)]
    return X, y, feature_names

def analyze():
    X, y, feature_names = generate_data()
    model = LogisticRegression().fit(X, y)
    weights = model.coef_[0]

    interpretations = []
    for name, w in zip(feature_names, weights):
        direction = "increases" if w > 0 else "decreases"
        interpretations.append(
            f"{name}: A higher value {direction} the likelihood of class 1 (weight={w:.4f})."
        )
    return interpretations

if __name__ == "__main__":
    for line in analyze():
        print(line)
