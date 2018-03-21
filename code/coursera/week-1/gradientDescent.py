import numpy as np
from computeCost import computeCost

def gradientDescent(X, y, theta, alpha, iterations):
    m = len(y)
    J_History = np.zeros((iterations, 1))

    for index in range(iterations):
        theta = theta - (alpha/m) * (np.dot(np.transpose(X), (np.dot(X, theta) - y)))
        J_History[index] = computeCost(X, y, theta)
    return theta, J_History