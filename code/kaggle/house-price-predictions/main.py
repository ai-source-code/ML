"""
Created on Thu Apr  5 12:22:43 2018

@author: Jitesh Lalwani
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.formula.api as sm
import seaborn as sns

# Importing the dataset
data_set = pd.read_csv('kc_house_data.csv')

# Data analysis and cleaning
# Looking for nulls
# print(data_set.isnull().any())
# Inspecting type
# print(data_set.dtypes)
data_set = data_set.drop(['id', 'date'], axis = 1)

X = data_set.iloc[:, 0:-1].values
y = data_set.iloc[:, -1].values

# Visualization the dataset
# sns.pairplot(data_set[['sqft_lot','sqft_above','price','sqft_living','bedrooms']], 
#                  hue='price', palette='tab20',size=6)
# plt.show()

# Splitting the dataset into Training and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)

y_pred = linear_regressor.predict(X_test)

# print(np.sqrt(mean_squared_error(y_test, y_pred)))
print(linear_regressor.score(X_test, y_test) * 100)

# Building the optimal model using Backward Elimination
X_ = np.append(arr = np.ones((X.shape[0], 1)), values = X, axis = 1)
X_opt = X_[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
linear_regressor_OLS  = sm.OLS(endog = y, exog = X_opt).fit()
# linear_regressor_OLS.summary()

X_opt = X_[:, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
linear_regressor_OLS  = sm.OLS(endog = y, exog = X_opt).fit()
# linear_regressor_OLS.summary()

# Splitting the dataset into Training and Test set
X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size = 0.20, random_state = 0)

linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)

y_pred_opt = linear_regressor.predict(X_test)

# print(np.sqrt(mean_squared_error(y_test, y_pred_opt)))
print(linear_regressor.score(X_test, y_test) * 100)


