import numpy as np
import os
from warmUpExercise import create_inverse_matrix
import matplotlib.pyplot as plt
from computeCost import computeCost

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
plt.scatter(x,y, marker='x', color='r')
plt.xlabel("Pouplation of city in 10,000")
plt.ylabel("Profits in $10,000s")
plt.show()
input("Press the <ENTER> key to continue...")

# 3. Cost and Gradient Descent
print("\nTesting the cost function ...\n")
X = np.column_stack([np.ones(m), x])
theta = np.zeros((2,1))
print(computeCost(X,y, theta))