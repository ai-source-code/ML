import numpy as np
import os
from warmUpExercise import create_inverse_matrix
import matplotlib.pyplot as plt
from computeCost import computeCost
from plotData import plotData
from gradientDescent import gradientDescent

os.system("clear")
# 1. Warmup excerise for creating identity matrix
print("Running warmUpExercise ... \n");
print("5x5 Identity Matrix: \n");
print(create_inverse_matrix(5))
input("Press the <ENTER> key to continue...")

# 2. Plotting data
print("Plotting Data ...\n")
data = np.loadtxt("ex1data1.txt", delimiter=',')

x = data[:,0,None]
y = data[:,1,None]

m = len(y)

plotData(x, y)
input("Press the <ENTER> key to continue...")

# 3. Cost and Gradient Descent
print("\nTesting the cost function ...\n")
X = np.column_stack([np.ones(m), x])
theta = np.zeros((2,1))

# Some gradient descent settings
iterations = 1500
alpha = 0.01

J = computeCost(X, y, theta)
print(f"With theta = [0 ; 0]\nCost computed = {J}\n")
print("Expected cost value (approx) 32.07\n")

# further testing of the cost function
abc = np.zeros((2,1))
abc[0] = -1
abc[1] = 2
J = computeCost(X, y, np.array(abc))
print(f"\nWith theta = [-1 ; 2] \n Cost computed = {J} \n ")
print("Expected cost value (approx) 54.24\n");
input("Press the <ENTER> key to continue...")

print("\nRunning Gradient Descent ...\n")
theta_array, J_History = gradientDescent(X, y, theta, alpha, iterations);

# print theta to screen
print("Theta found by gradient descent:\n");
print(f"{theta_array[0][0]} \n {theta_array[1][0]} \n");
print("Expected theta values (approx)\n");
print(" -3.6303\n  1.1664\n\n");

plt.plot(X[:,1, None], np.dot(X, theta))
plt.show()