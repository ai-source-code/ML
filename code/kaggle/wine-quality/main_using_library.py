import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('winequality-red.csv', header=None, skiprows=1)
X = df.iloc[:, :-1]
y = df.iloc[:,-1]

regr = linear_model.LinearRegression()

regr.fit(X, y)

y_pred = regr.predict(X)

for index in range(len(y)):
    print(f'Predicted value is: {np.round(y_pred[index])} and actual value is: {y[index]}')

print(f'Train accurancy is : {np.mean(np.round(y_pred) == y) * 100}')
# print('Variance score: %.2f' % r2_score(y, y_pred))
# print("Mean squared error: %.2f"
#       % mean_squared_error(y, y_pred))
