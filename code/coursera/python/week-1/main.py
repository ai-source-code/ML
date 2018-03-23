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
newTheta = np.zeros((2,1))
newTheta[0] = -1
newTheta[1] = 2
J = computeCost(X, y, np.array(newTheta))
print(f"\nWith theta = [-1 ; 2] \n Cost computed = {J} \n ")
print("Expected cost value (approx) 54.24\n");
input("Press the <ENTER> key to continue...")

print("\nRunning Gradient Descent ...\n")
theta, J_History = gradientDescent(X, y, theta, alpha, iterations);

# print theta to screen
print("Theta found by gradient descent:\n");
print(f"{theta[0][0]} \n {theta[1][0]} \n");
print("Expected theta values (approx)\n");
print(" -3.6303\n 1.1664\n\n");

plt.plot(X[:,1, None], np.dot(X, theta),'-', label='Linear regression')
plt.legend(loc='lower right')
plt.xticks([5, 10, 15, 20, 25])
plt.yticks([-5, 0, 5, 10, 15, 20, 25])
plt.show()

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot(np.array([1, 3.5]),theta)
predict2 = np.dot(np.array([1, 7]),theta)
print(f"For population = 35,000, we predict a profit of {predict1[0]*10000}\n")
print(f"For population = 70,000, we predict a profit of {predict2[0]*10000}\n")