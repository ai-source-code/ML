import os
import numpy as np
from gradientDescent import gradient_descent
from featureNormalization import normalize_feature
from predict import predict

# To clear the screen
os.system("clear")

# Load data from excel sheet
# data = np.loadtxt(open('winequality-red.csv', 'rt'), delimiter=",", skiprows=1)
data = np.loadtxt("winequality-red.csv", delimiter=',', skiprows=1)

X = data[:, :3]
y = data[:, -1, None]

# Gradient settings
alpha = 0.01
iterations = 1500

m = len(y)

# Normalize Features
X = normalize_feature(X)

# Add X0 as 1
X = np.column_stack([np.ones(m), X])

theta = np.zeros((X.shape[1], 1))
theta, j_history = gradient_descent(X, y, theta, alpha, iterations)

print(f'Predicted Wine quality for params "7.4, 0.7, 0.0" is: {predict([1, 7.4, 0.7, 0.0], theta)} and actual is "5"')
print(f'Predicted Wine quality for params "7.9, 0.32, 0.51" is: {predict([1, 7.9, 0.32, 0.51], theta)} and actual is "6"')
print(f'Predicted Wine quality for params "7.7, 0.58, 0.01" is: {predict([1, 7.7, 0.58, 0.01], theta)} and actual is "7"')