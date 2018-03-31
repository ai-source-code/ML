import os
import numpy as np
import pandas as pd
from gradientDescent import gradient_descent
from featureNormalization import normalize_feature
from predict import predict
from plotMultipleFeatures import plot_multiple_features

# To clear the screen
os.system("clear")

# Load data from excel sheet
# data = np.loadtxt(open('winequality-red.csv', 'rt'), delimiter=",", skiprows=1)
# data = np.loadtxt("winequality-red.csv", delimiter=',', skiprows=1)
data = pd.read_csv("winequality-red.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

y = y.reshape(y.shape[0], 1)

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

predicted_output = []
for index in range(len(X)):
    predicted_output.append(np.around(predict(X[index], theta)))
    print(f'Predicted value is: {predicted_output[index]} and actual value is: {y[index]}')
print(f'Train accurancy is : {np.mean(predicted_output == y) * 100}')
