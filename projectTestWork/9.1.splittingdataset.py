
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
from matplotlib.colors import ListedColormap

path = ""
filenameForIrisData = path + "iris.csv"
header_list = ["sepallengthcm", "sepalwidthcm", "petallengthcm", "petalwidthcm", "species"]
df = pd.read_csv(filenameForIrisData)
iris_df = df.rename(columns = {"sepallength" : "sepallengthcm", "sepalwidth" : "sepalwidthcm", "petallength" : "petallengthcm", "petalwidth" : "petalwidthcm", "class" : "species"})
# print (iris_df)

test_size = 0.30
seed = 7
score = 'accuracy'

def models(X_train, Y_train,score):
    clfs = []
    result = []
    names = []
    clfs.append(('LR', LogisticRegression()))
    clfs.append(('LDA', LinearDiscriminantAnalysis()))
    clfs.append(('KNN', KNeighborsClassifier()))
    clfs.append(('CART', DecisionTreeClassifier()))
    clfs.append(('NB', GaussianNB()))
    clfs.append(('SVM', SVC()))
    for algo_name, clf in clfs:
        k_fold = model_selection.KFold(n_splits=10, random_state=None)
        cv_score = model_selection.cross_val_score(clf, X_train, Y_train, cv=k_fold, scoring=score)
        #result = "%s: %f (%f)" % (algo_name, cv_score.mean(), cv_score.std())
        result.append((algo_name,cv_score.mean(), cv_score.std()))
        names.append(algo_name)
    return (result)

# X_all = iris_df.iloc[:,:4]
# Y_all = iris_df.iloc[:,4] 

X_all = iris_df[['petallengthcm','petalwidthcm']]
Y_all = iris_df.species 

X_train_all, X_test_all, Y_train_all, Y_test_all = model_selection.train_test_split(X_all, Y_all, test_size=test_size, random_state=seed)

models(X_train_all, Y_train_all, score)

svm = SVC()
svm.fit(X_train_all, Y_train_all)
pred = svm.predict(X_test_all)
print(accuracy_score(Y_test_all, pred))
print(confusion_matrix(Y_test_all, pred))
print(classification_report(Y_test_all, pred))

