import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

from logistic_regression import LogisticRegressionScratch

data = load_breast_cancer()
X, y = data.data, data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegressionScratch(lr=0.01, n_iters=5000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

print("Custom Logistic Regression:")
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("Coefficients:", model.weights)

sk = LogisticRegression(max_iter=5000)
sk.fit(X_train, y_train)
y_pred_sk = sk.predict(X_test)

print("\nScikit-learn Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred_sk))
print("Precision:", precision_score(y_test, y_pred_sk))
print("Recall:", recall_score(y_test, recall_score(y_test, y_pred_sk)))
