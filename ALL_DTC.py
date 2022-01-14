# DECISION TREE CLASSIFICATION

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv('ALL_FINAL.csv')
X = dataset.iloc[:, 1:12].values
y = dataset.iloc[:, 12].values

# Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()

X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
X[:, 10] = labelencoder_X.fit_transform(X[:, 10])
onehotencoder = OneHotEncoder(categorical_features = [0,10])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
OneHotEncoder()

#Splitting the data into test set and training set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 7)

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test) 

# predicting accuracy
from sklearn.metrics import accuracy_score
a = accuracy_score(y_test,y_pred) 