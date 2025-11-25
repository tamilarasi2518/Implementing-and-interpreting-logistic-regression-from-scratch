import numpy as np

class LogisticRegressionScratch:
    def __init__(self, lr=0.01, n_iter=3000):
        self.lr = lr
        self.n_iter = n_iter

    def _sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def fit(self, X, y):
        X=np.c_[np.ones((X.shape[0],1)),X]
        self.theta=np.zeros(X.shape[1])
        for _ in range(self.n_iter):
            preds=self._sigmoid(X@self.theta)
            grad=X.T@(preds-y)/len(y)
            self.theta-=self.lr*grad

    def predict(self,X):
        X=np.c_[np.ones((X.shape[0],1)),X]
        return (self._sigmoid(X@self.theta)>=0.5).astype(int)

    def coefficients(self, feature_names):
        return {name: w for name,w in zip(["bias"]+feature_names, self.theta)}
