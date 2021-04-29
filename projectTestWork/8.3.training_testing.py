# Training and Testing the dataset
# Author : Michelle O'Connor

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# Notes and references contain in 2.1 

path = ""
filenameForIrisData = path + "iris.csv"
header_list = ["sepallengthcm", "sepalwidthcm", "petallengthcm", "petalwidthcm", "species"]
df = pd.read_csv(filenameForIrisData)
iris_df = df.rename(columns = {"sepallength" : "sepallengthcm", "sepalwidth" : "sepalwidthcm", "petallength" : "petallengthcm", "petalwidth" : "petalwidthcm", "class" : "species"})


# Before implementing any model we need to split the dataset to train and test sets. We use train_test_split class from sklearn.model_selection library to split our dataset.

from sklearn.model_selection import train_test_split
train,test=train_test_split(iris_df,test_size=0.25)

# print (train.shape, test.shape)

target_attribute = iris_df["species"]

X_train, X_test, y_train, y_test = train_test_split(iris_df, target_attribute , test_size=0.25, random_state=0)


# X_train, X_test, y_train, y_test = train_test_split(iris_df,test_size=0.25)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

### https://medium.com/gft-engineering/start-to-learn-machine-learning-with-the-iris-flower-classification-challenge-4859a920e5e3


# X_train, X_test, y_train, y_test = train_test_split(iris_df, ["sepallengthcm","sepalwidthcm", "petallengthcm", "petalwidthcm"] ,test_size=0.25)

# print("X_train shape: {}".format(X_train.shape))
# print("y_train shape: {}".format(y_train.shape))

# https://www.kaggle.com/sharmajayesh76/iris-data-train-test-split


'''
X_train=train[‘Sepallengthcm’,”Sepalwidthcm”,”Petallengthcm”,”Petalwidthcm”]
y_train=train.species
X_test=test[[‘Sepallengthcm’,”Sepalwidthcm”,”Petallengthcm”,”Petalwidthcm”]]
y_test_y=test.species
''' 

knn = KNeighborsClassifier(n_neighbors=1)

# fitting the nearest neighbour and getting it to fit to the training data and training target
knn.fit(X_train, y_train)

X_new = np.array([[5.0, 2.9, 1.0, 0.2]])

# prediction = knn.predict(X_new)
# print(prediction)

print(knn.score(X_test, y_test))
