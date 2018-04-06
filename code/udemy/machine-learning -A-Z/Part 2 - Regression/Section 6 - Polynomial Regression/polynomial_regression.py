# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting linear regression to the dataset
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)

# Fitting polynomial regression to the dataset
polynomial_regressor = PolynomialFeatures(degree = 4)
X_poly = polynomial_regressor.fit_transform(X)
# This line may be redundant
polynomial_regressor.fit(X_poly, y)
linear_regressor_2 = LinearRegression()
linear_regressor_2.fit(X_poly, y)

# Visualize Linear Regression
plt.scatter(X, y, color = 'red')
plt.plot(X, linear_regressor.predict(X), color = 'blue')
plt.show()

# Visualize Polynomial Linear Regression
plt.scatter(X, y, color = 'red')
plt.plot(X, linear_regressor_2.predict(polynomial_regressor.fit_transform(X)), color = 'blue')
plt.show()


# Predicting a new result using Linear Regression
linear_regressor.predict(6.5)

# Predicting a new result using Polynomial Regression
linear_regressor_2.predict(polynomial_regressor.fit_transform(6.5))