# In this script we will be creating linear regression from scratch
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

def create_dataset(hm, variance, step=2, correlation=False):
    """
    hm: how much points
    variance: the variance of the y
    step: the step between each point
    correlation: +ve or -ve or no correlation
    """
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    X = [i for i in range(len(ys))]
    return np.array(X, dtype=np.float64), np.array(ys, dtype=np.float64)


X, y = create_dataset(40, 40, 2, correlation='pos')



# the slope of the best fit line is got from (mean(x) * mean(y) - mean(x*y)) / (mean(x) ** 2 - mean(x**2))
def best_fit_slope(X, y):
    X_bar = mean(X)
    y_bar = mean(y)
    Xy_bar = mean(X * y)
    X_2_bar = mean(X ** 2)
    X_bar_2 = mean(X) ** 2
    best = (X_bar * y_bar - Xy_bar) / (X_bar_2 - X_2_bar)
    return best


# To get the best y intercept = mean(y) - m * mean(x)
def best_y_intercept(X, y):
    y_bar = mean(y)
    X_bar = mean(X)
    m = best_fit_slope(X, y)
    b = y_bar - m * X_bar
    return b


# Combined function
def best_fit_slope_and_intercept(X, y):
    m = best_fit_slope(X, y)
    b = best_y_intercept(X, y)
    return m, b


def predict(X, m, b):
    return (m * X) + b


def squared_error(y, y_pred):
    return sum((y - y_pred) ** 2)


def r_squared(y, y_pred):  # coefficient of determination
    y_mean_line = [mean(y) for _ in y]
    r2 = 1 - (squared_error(y, y_pred) / squared_error(y, y_mean_line))
    return r2


m, b = best_fit_slope_and_intercept(X, y)
print(m, b)

# Getting the regression line
regression_line = [(m * x) + b for x in X]
predict_y = predict(8, m, b)

r2 = r_squared(y, regression_line)
print(f"r2: {r2}")

plt.scatter(X, y)
plt.scatter(8, predict_y, c='g')
plt.plot(X, regression_line)
plt.show()

predict_y = predict(8, m, b)
print(f"Prediction: {predict_y}")

