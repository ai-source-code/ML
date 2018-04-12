import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier

data_set= pd.read_csv('spambase.data')
np.random.permutation(data_set)
X = data_set.iloc[:, :48].values
y = data_set.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = MultinomialNB()
model.fit(X_train, y_train)
print("Classification rate for NB:", model.score(X_test, y_test))

model = AdaBoostClassifier()
model.fit(X_train, y_train)
print("Classification rate for AdaBoost:", model.score(X_test, y_test))
