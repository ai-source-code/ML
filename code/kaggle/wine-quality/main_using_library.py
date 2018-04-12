import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier

df = pd.read_csv('winequality-red.csv', header=None, skiprows=1)
X = df.iloc[:, :-1].values
y = df.iloc[:,-1].values

plt.scatter(X[:, 1], y, color='red')
plt.show()

# y_pred = regr.predict(X)

# for index in range(len(y)):
#     print(f'Predicted value is: {np.round(y_pred[index])} and actual value is: {y[index]}')
X_train, X_test, Y_train, Y_test = train_test_split(X, y)

model = linear_model.LinearRegression()
model.fit(X_train, Y_train)
# print(f'Train accurancy is : {np.mean(np.round(y_pred) == y) * 100}')
print("LinearRegression train score:", model.score(X_train, Y_train))
print("LinearRegression test score:", model.score(X_test, Y_test))

model = MultinomialNB()
model.fit(X_train, Y_train)
print("MultinomialNB train score:", model.score(X_train, Y_train))
print("MultinomialNB test score:", model.score(X_test, Y_test))

model = AdaBoostClassifier()
model.fit(X_train, Y_train)
print("AdaBoostClassifier train score:", model.score(X_train, Y_train))
print("AdaBoostClassifier test score:", model.score(X_test, Y_test))

# print('Variance score: %.2f' % r2_score(y, y_pred))
# print("Mean squared error: %.2f"
#       % mean_squared_error(y, y_pred))
