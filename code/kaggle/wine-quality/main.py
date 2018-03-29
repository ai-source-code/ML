import os
import numpy as np
import matplotlib.pyplot as plt
from gradientDescent import gradient_descent
from costFunction import cost_function

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
X = np.column_stack([np.ones(m), X])
# print(X.shape)
theta = np.zeros((X.shape[1], 1))
# cost = cost_function(X, y, theta)
# print(cost)
theta, j_history = gradient_descent(X, y, theta, alpha, iterations)
print(f'Predicted Wine quality for params "7.4, 0.7, 0.0" is: {np.dot([1, 7.4, 0.7, 0.0], theta)} and actual is "5"')
print(f'Predicted Wine quality for params "7.9, 0.32, 0.51" is: {np.dot([1, 7.9, 0.32, 0.51], theta)} and actual is "6"')
# print(f'optional theta is: {theta} and j_history is {j_history}')
# x1 = data[:,0, None]
# x2 = data[:,1, None]
# x3 = data[:,2, None]
# x4 = data[:,3, None]
# x5 = data[:,4, None]
# x6 = data[:,5, None]
# x7 = data[:,6, None]
# x8 = data[:,7, None]
# x9 = data[:,8, None]
# x10 = data[:,9, None]
# x11 = data[:,10, None]
# x = [x1, x2, x3, x4, x5, x6, x7, x8, x9]
# print(X, y)
# plt.scatter(x1, x2, x3, x4, x5, x6, x7, x8, x9, y)
# plt.scatter(data[:,1, None], y, color='r')
# plt.scatter(data[:,2, None], y, color='y')
# plt.show()