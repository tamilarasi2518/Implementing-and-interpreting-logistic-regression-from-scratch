from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from logistic_regression import LogisticRegressionScratch

X,y=make_classification(n_samples=1000,n_features=5,n_informative=3,random_state=42)
feature_names=[f"feature_{i}" for i in range(X.shape[1])]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LogisticRegressionScratch()
model.fit(X_train,y_train)

preds=model.predict(X_test)
print("Accuracy:",accuracy_score(y_test,preds))
print("Coefficients:",model.coefficients(feature_names))
