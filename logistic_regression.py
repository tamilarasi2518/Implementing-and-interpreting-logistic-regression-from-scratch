
import numpy as np

class LogisticRegressionScratch:
    def __init__(self, lr=0.01, n_iter=5000):
        self.lr = lr
        self.n_iter = n_iter

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.c_[np.ones(len(X)), X]
        self.theta = np.zeros(X.shape[1])
        for _ in range(self.n_iter):
            z = X.dot(self.theta)
            y_pred = self.sigmoid(z)
            gradient = X.T.dot(y_pred - y) / len(y)
            self.theta -= self.lr * gradient

    def predict_proba(self, X):
        X = np.c_[np.ones(len(X)), X]
        return self.sigmoid(X.dot(self.theta))

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def get_params(self):
        return self.theta
