import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets

# Perceptron is a single layer feed forward neural network.

def activation_function(x):  # unit step function
    return np.where(x > 0, 1, 0)


class Perceptron:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.activation_func = activation_function
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.random.randn(
            n_features)  # Randomly initialize the weights by sampling from a standard normal distribution
        '''
        The above line is important here to make each of our units learn a different function
        '''
        self.bias = 0  # Initialize the bias by 0

        # Making our class labels 0 and 1
        y = np.where(y > 0, 1, 0)

        # Learn weights
        for i in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = self.activation_func(linear_output)

                # Update rule of our update rule
                u = self.lr * (y[idx] - y_pred)  # If predictions == true value this will be 0
                self.weights += u * x_i  # We multiply by x_i to update the weights of the whole matrix W
                self.bias += u

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = self.activation_func(linear_output)
        return y_pred


def acc(y_true, y_pred):
    return 100 * np.sum(y_true == y_pred) / len(y_true)


X, y = datasets.make_blobs(n_samples=15, n_features=2, centers=2, cluster_std=1.05, random_state=2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

p = Perceptron(lr=0.001, n_iters=10000)
p.fit(X_train, y_train)

preds = p.predict(X_test)

print('Perceptron Accuracy: ', acc(y_test, preds))
