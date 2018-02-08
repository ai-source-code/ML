import numpy as np

def getBodyData():
    # Normal takes distribution mean, distribution standard deviation and
    # number of samples as parameters
    height = np.round(np.random.normal(1.75, 0.20, 5000), 2)
    weight = np.round(np.random.normal(60, 15, 5000), 2)
    bodyData = np.array([height, weight])
    return(bodyData)