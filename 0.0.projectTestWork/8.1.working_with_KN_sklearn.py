import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets


from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
iris = load_iris()

# https://www.youtube.com/watch?v=PymyBRzlRXc Machine Learning in Python: Iris Classification - Part 3

# 75%:25% split - using 75% of the data for training and 25% of the data for testing

X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)

# this considers the nearest neighbour to classified your test. This can be increased based on your preference
knn = KNeighborsClassifier(n_neighbors=1)

# fitting the nearest neighbour and getting it to fit to the training data and training target
knn.fit(X_train, y_train)

X_new = np.array([[5.0, 2.9, 1.0, 0.2]])

# prediction = knn.predict(X_new)
# print(prediction)

print(knn.score(X_test, y_test))