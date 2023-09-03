import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr  # Learning rate for gradient descent
        self.n_iterations = n_iters  # Number of iterations of gradient descent
        self.weights = None  # W in  y = Wx + b
        self.bias = None  # b in y = Wx + b

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))  # Declaring the weights as zeros "can start as random"
        self.bias = 0  # Declaring the bias as zero

        for _ in range(self.n_iterations):  # Gradient descent loop for the number of iterations
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))  # The derivative equation for the Mean Squared error loss
            db = (1 / n_samples) * np.sum(y_pred - y)  # The derivative equation for the Mean Squared error loss

            self.weights -= self.lr * dw  # Updating our weights
            self.bias -=  self.lr * db  # Updating our bias

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias  # predicting the y for x

    def mse(self, y_test, predictions):
        return np.mean((y_test - predictions) ** 2)

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

fig = plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], y, color="b", marker="o", s=30)
plt.show()

reg = LinearRegression(lr=0.01)
y_train = y_train.reshape(80, 1)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)




mse = reg.mse(y_test, predictions)
print(mse)

y_pred_line = reg.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
plt.show()
