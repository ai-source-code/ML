import numpy as np
from costFunction import cost_function as cost_function
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    j_history = np.zeros((iterations, 1))
    # print(X.shape, np.transpose(X).shape, theta.shape, y.shape)
    for index in range(iterations):
        X_tranpose = np.transpose(X)
        hypothesis = np.dot(X, theta)
        theta = theta - (alpha/m) * (np.dot(X_tranpose, (hypothesis - y)))
        j_history[index] = cost_function(X, y, theta)
    return theta, j_history