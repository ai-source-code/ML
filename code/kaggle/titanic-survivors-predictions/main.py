# Data Preprocessing Template

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# Importing the dataset
training_data_set = pd.read_csv('data/train.csv')
training_data_set['Age'].fillna(training_data_set['Age'].median(), inplace = True)
training_data_set['Embarked'].fillna(training_data_set['Embarked'].mode()[0], inplace = True)
#training_data_set['Age'] = training_data_set['Age'].fillna(0)
#training_data_set['Embarked'] = training_data_set['Embarked'].fillna('S')
X_train = training_data_set.iloc[:, [2, 4, 5, 6, 7, 11]].values
y_train = training_data_set.iloc[:, 1].values

test_data_set = pd.read_csv('data/test.csv')
test_data_set['Age'].fillna(test_data_set['Age'].median(), inplace = True)
test_data_set['Embarked'].fillna(test_data_set['Embarked'].mode()[0], inplace = True)
"""test_data_set['Age'] = test_data_set['Age'].fillna(0)
test_data_set['Embarked'] = test_data_set['Embarked'].fillna('S')"""
X_test = test_data_set.iloc[:, [1, 3, 4, 5, 6, 10]].values
passenger_Id = test_data_set.iloc[:, 0].values

# Taking care of missing data
# imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# imputer = imputer.fit(X[:,1:3])
# X[:,1:3] = imputer.transform(X[:,1:3])

# Encoding categorical data
label_encoder_X = LabelEncoder()
X_train[:,1] = label_encoder_X.fit_transform(X_train[:,1])
one_hot_encoder = OneHotEncoder(categorical_features = [1])
X_train = one_hot_encoder.fit_transform(X_train).toarray()

label_encoder_X = LabelEncoder()
X_train[:,-1] = label_encoder_X.fit_transform(X_train[:,-1])
one_hot_encoder = OneHotEncoder(categorical_features = [-1])
X_train = one_hot_encoder.fit_transform(X_train).toarray()

# Encoding categorical data
label_encoder_X = LabelEncoder()
X_test[:,1] = label_encoder_X.fit_transform(X_test[:,1])
one_hot_encoder = OneHotEncoder(categorical_features = [1])
X_test = one_hot_encoder.fit_transform(X_test).toarray()

label_encoder_X = LabelEncoder()
X_test[:,-1] = label_encoder_X.fit_transform(X_test[:,-1])
one_hot_encoder = OneHotEncoder(categorical_features = [-1])
X_test = one_hot_encoder.fit_transform(X_test).toarray()

# label_encoder_y = LabelEncoder()
# y = label_encoder_y.fit_transform(y)

# classifier = LogisticRegression(random_state = 0)
# classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

df = pd.DataFrame(np.column_stack((passenger_Id, y_pred)))
df.to_csv('data/predictions-random-forest.csv', header = ['PassengerId', 'Survived'], index = False)

# plt.plot(X_train[:, 1], y_train, color='red')
# plt.show()

# Feature Scaling
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)