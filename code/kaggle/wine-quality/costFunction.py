import numpy as np
def cost_function(X, y, theta):
    m = len(y)
    hypothesis = np.dot(X, theta)
    J = (1.0/(2*m)) * sum((hypothesis - y) ** 2)
    return J