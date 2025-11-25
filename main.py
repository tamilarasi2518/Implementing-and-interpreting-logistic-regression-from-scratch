
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from logistic_regression import LogisticRegressionScratch

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegressionScratch(lr=0.001, n_iter=8000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = (y_pred == y_test).mean()

print("\nCustom Logistic Regression Accuracy:", accuracy)
print("\nLearned Parameters (θ):\n", model.get_params())
