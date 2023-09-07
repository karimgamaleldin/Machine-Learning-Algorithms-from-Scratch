import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, lr = 0.001, n_iterations=1000):
        self.lr = lr
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # Initializing W
        self.bias = 0  # Initializing b

        # Gradient descent
        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias  # Getting the values
            y_pred = self._sigmoid(linear_model)  # Passing it to our sigmoid activation function

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))  # Getting the dj/dw
            db = (1 / n_samples) * np.sum(y_pred - y)  # Getting the dj/db

            self.weights -= self.lr * dw  # updating our weights
            self.bias -= self.lr * db  # updating our bias


    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_preds = self._sigmoid(linear_model)
        y_class = [1 if i > 0.5 else 0 for i in y_preds]
        return y_class


data = datasets.load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

regressor = LogisticRegression(lr=0.0001, n_iterations=1000)
regressor.fit(X_train, y_train)
y_preds = regressor.predict(X_test)

print("LR classification accuracy:", np.sum(y_test == y_preds) / len(y_test))
        