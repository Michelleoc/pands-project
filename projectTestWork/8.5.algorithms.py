

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
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_confusion_matrix as cm


from sklearn.datasets import load_iris
iris=load_iris()
# for keys in iris.keys() :
    # print(keys)

X=iris.data
y=iris.target

# iris_dataset = load_iris()
# print("Target names: {}".format(iris_dataset['target_names']))
# print("Feature names: {}".format(iris_dataset['feature_names']))
# print("Type of data: {}".format(type(iris_dataset['data'])))

'''
path = ""
filenameForIrisData = path + "iris.csv"
header_list = ["sepallengthcm", "sepalwidthcm", "petallengthcm", "petalwidthcm", "species"]
df = pd.read_csv(filenameForIrisData)
iris_df = df.rename(columns = {"sepallength" : "sepallengthcm", "sepalwidth" : "sepalwidthcm", "petallength" : "petallengthcm", "petalwidth" : "petalwidthcm", "class" : "species"})

X = iris_df[['sepallengthcm','sepalwidthcm','petallengthcm','petalwidthcm']]
y= iris_df.species 
'''

# Splitting the iris dataset 75% for train, evaluate and select amount our models and 25% is the test dataset
# X_train, y_train for training the model
# X_test, y_test for testing the model
# X = iris.data  (4 features/variables sepal length, sepal width, petal length, petal width)
# y = iris.target (species)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

# https://medium.com/gft-engineering/start-to-learn-machine-learning-with-the-iris-flower-classification-challenge-4859a920e5e3

# to show the shape/size of the train data samples
# X_train is 75% of 150 = 112 rows of data in 4 columns (the 4 features/variables)
# y_train is 75% of 150 = 112 rows of the species column
print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
# to show the shape/size of the test data samples
# X_train is 25% of 150 = 38 rows of data in 4 columns (the 4 features/variables)
# y_train is 25% of 150 = 38 rows of the species column
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))


# k-nearest neighbors classifier makes a prediction for a new data point, 
# the algorithm finds the point in the training set, then it assigns the label of this training point to the new data point.
# The k in k-nearest neighbors signifies that instead of using only the closest neighbor to the new data point, 
# we can consider any fixed number k of neighbors in the training
# We can now make a prediction using the majority class among them. For our example, we will use one neighbor (k=1).

knn = KNeighborsClassifier(n_neighbors=1)

# I now use the fit method of the knn object, which takes as arguments the array X_train 
# (containing the training data of the 4 features) and the array y_train (containing the corresponding species).
# This builds up our model on the training set.
knn.fit(X_train, y_train)

# Enter sample data
X_new = np.array([[5, 2.9, 1, 0.2]])
# Show the shape of the data, it is one row (1 sample) with 4 columns of data (the 4 features/variables sepal and petal measurements)
print("X_new.shape: {}".format(X_new.shape))

# I now use the predict method of the knn object to predict the species of the sample data X_new
prediction = knn.predict(X_new)
# prediction to be species '0'
print("Prediction: {}".format(prediction))
# species '0' equals setosa
print("Predicted target name: {}".format(iris['target_names'][prediction]))



# But how can we trust the results of the model 
# The test set that was created was not used to build the model, but we do know the correct species for each iris in the test set. 
# Therefore, we can make a prediction for each iris in the test data and compare it against its label — so we can know if the model 
# is correctly predicting the label for a given flower.

# To measure how well the model works, we can obtain the accuracy - 
# the fraction of flowers for which the right species was predicted (number that we can calculate using the NumPy “mean” method, comparing both datasets)

# using the X_test data containing the testing data of the 4 features 
y_pred = knn.predict(X_test)
# we print out the prediction of the species using only the 4 features
print("Test set predictions:\n {}".format(y_pred))
# we then compare the prediction of the species 'y_pred' to the actual species 'y_test'
print("Test set score (np.mean): {:.2f}".format(np.mean(y_pred == y_test)))
# another way to do this would be to use the score method of the knn object, which will compute the test set accuracy
print("Test set score (knn.score): {:.2f}".format(knn.score(X_test, y_test)))

# For this model, the accuracy on the test set is 0.97, which means the model made the right prediction for 95% of the irises 
# in the given dataset. We can expect the model to be correct 97% of the time for predicting the species of new irises.
# For a hobby botanist application, this is a high level of accuracy and it means that our model may be trustworthy enough to use
# 
# Albeit simple, the iris flower classification problem (and our implementation) 
# is a perfect example to illustrate how a machine learning problem should be approached and how useful the outcome can be to a potential user

# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
# https://machinelearningmastery.com/make-predictions-scikit-learn/

# Spot Check other Algorithms to see 
# We get an idea from the plots that some of the classes are partially linearly separable in some dimensions, so we are expecting generally good result
# we will test 6 different algorithms 
# This is a good mixture of simple linear (LR and LDA), nonlinear (KNN, CART, NB and SVM) algorithms.

models = []
models.append(('LR', LogisticRegression(solver='liblinear')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: accuracy %f with a standard deviation of (%f) ' % (name, cv_results.mean(), cv_results.std()))

# We can also create a plot of the model evaluation results and compare the spread and the mean accuracy of each model. 
# A useful way to compare the samples of results for each algorithm is to create a box and whisker plot for each distribution and compare the distributions.

plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()
# We can see that the box and whisker plots are squashed at the top of the range, 
# with many evaluations achieving 100% accuracy, and some pushing down into the high 80% accuracies.

# Make predictions on test/validation dataset 

# The results in the previous section suggest that the ____ was perhaps the most accurate model. We will use this model as our final model.
# Now we want to get an idea of the accuracy of the model on our validation set.
# This will give us an independent final check on the accuracy of the best model. 
# It is valuable to keep a testingn set just in case you made a slip during training, 
# such as overfitting to the training set or a data leak. Both of these issues will result in an overly optimistic result.

# We can fit the model on the entire training dataset and make predictions on the validation dataset.
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
predictions = lda.predict(X_test)

# Evaluate predictions
# We can evaluate the predictions by comparing them to the expected results in the validation set, 
# then calculate classification accuracy, as well as a confusion matrix and a classification report.

print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

print("Accruacy score: {:.3f}".format(accuracy_score(y_test, predictions)))

class_names = iris.target_names
cnf_matrix = confusion_matrix(y_test, predictions) 
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix)
plt.show()

# confusion = plot_confusion_matrix(X_test, y_test, predictions)
# plt.show()

# We can see that the accuracy is 1.0 or about 100% on the hold out dataset.
# The confusion matrix provides an indication of the errors made.
# Finally, the classification report provides a breakdown of each class by precision,
# recall, f1-score and support showing excellent results (granted the validation dataset was small).

# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

# https://medium.com/gft-engineering/start-to-learn-machine-learning-with-the-iris-flower-classification-challenge-4859a920e5e3