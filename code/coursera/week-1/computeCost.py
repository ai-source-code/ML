import numpy as np
def computeCost(X, y, theta):
    m = len(y)
    J = (1.0/(2*m)) * sum((np.dot(X, theta) - y) ** 2)
    return J