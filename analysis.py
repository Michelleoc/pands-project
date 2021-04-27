# Investigating the Iris Dataset
# Author : Michelle O'Connor

import pandas as pd             # data processing and csv file i/o library
import numpy as np              # 
import seaborn as sns           # 
import matplotlib.pyplot as plt # plotting library
from sklearn import model_selection # Python graphing library based on matplotlib
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression # # for Logistic Regression algorithm
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm
from sklearn.neighbors import KNeighborsClassifier # for K nearest neighbours
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC #for Support Vector Machine (SVM) Algorithm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split #to split the dataset for training and testing

# Import the dataset from iris.csv file 
path = ""
filenameForIrisData = path + "irisoriginal.csv"
dataframe = pd.read_csv(filenameForIrisData)

# Rename column titles
# https://www.geeksforgeeks.org/python-pandas-dataframe-rename/
iris_dataframe = dataframe.rename(columns = {"sepallength" : "Sepal_length(cm)", "sepalwidth" : "Sepal_width(cm)", "petallength" : "Petal_length(cm)", "petalwidth" : "Petal_width(cm)", "class" : "species"})

# To show the full list print (iris_df), but for now I just want to see the first 5 rows of data 
print(iris_dataframe.head(5))   

# To show the shape of the dataset : (150, 5) 150 rows, 5 columns
# https://www.c-sharpcorner.com/article/a-first-machine-learning-project-in-python-with-iris-dataset/
print(iris_dataframe.shape)

# Shows the number of instances and the number of attributes in the dataset. 
# Show if there are any null values
print(iris_dataframe.info())

# View how many instances the data frame containsby species
# https://www.c-sharpcorner.com/article/a-first-machine-learning-project-in-python-with-iris-dataset/
print(iris_dataframe.groupby('species').size())   

# Shows the basic statistical details of a the dataframe (iris dataset)
# https://www.kaggle.com/adityabhat24/iris-data-analysis-and-machine-learning-python
print(iris_dataframe.describe())  

# Shows the first 5 rows of each type of species
# https://www.geeksforgeeks.org/python-basics-of-pandas-using-iris-dataset/
print(iris_dataframe[0:5])
print(iris_dataframe[50:55])
print(iris_dataframe[100:105])  

# Shows if duplicates exist within the dataset 
print(iris_dataframe.duplicated().sum())

# Shows the duplicated rows 
print (iris_dataframe[iris_dataframe.duplicated()]) 

# To output a summary of each variable (feature) to a single text file
# input has to be in string format
# class lectures showed how to export to a newly created txt file
# https://towardsdatascience.com/how-to-use-groupby-and-aggregate-functions-in-pandas-for-quick-data-analysis-c19e7ea76367
with open("Variable_Summary.txt", "wt") as f:
    print ("Shape of Data \n", str(iris_dataframe.shape),"\n", file = f) 
    print ("Count by Species \n", str(iris_dataframe.groupby('species').size()),"\n", file = f)
    print ("Statistical Data feature \n", str(iris_dataframe.describe()),"\n", file = f)
    print ("Summary of each feature by species \n",str(iris_dataframe.groupby("species").describe()), "\n", file = f)

# The duplicate row analysis above highlighted that my original dataset was incorrect (only 1 duplicate should exist, not 3).  
# Therefore I import the correct iris dataset.   

path = ""
filenameForIrisData = path + "iris.csv"
df = pd.read_csv(filenameForIrisData)

iris_df = df.rename(columns = {"sepallength" : "Sepal_length(cm)", "sepalwidth" : "Sepal_width(cm)", "petallength" : "Petal_length(cm)", "petalwidth" : "Petal_width(cm)", "class" : "species"})

# Checking to ensure I now only have 1 duplicate  
print(iris_df.duplicated().sum())
print (iris_df[iris_df.duplicated()])   

# I start by showing a simple histogram of each 4 features
# https://www.python-graph-gallery.com/25-histogram-with-several-variables-seaborn

# This sets out the layout and size of the visual ouput
fig, axs = plt.subplots(2, 2, figsize=(7, 7))

# list the data source, followed by the x axis source, the colour of the columns and the position in the layout
sns.histplot(data=iris_df, x="Sepal_length(cm)", color="skyblue", ax=axs[0, 0])
sns.histplot(data=iris_df, x="Sepal_width(cm)", color="olive", ax=axs[0, 1])
sns.histplot(data=iris_df, x="Petal_length(cm)", color="gold", ax=axs[1, 0])
sns.histplot(data=iris_df, x="Sepal_width(cm)", color="teal", ax=axs[1, 1])
plt.savefig("Features_Histogram.png")
plt.show()

# Plotting, displaying and saving a histogram of each variable to png files  
# https://www.kaggle.com/dhruvmak/iris-flower-classification-with-eda
# https://www.geeksforgeeks.org/box-plot-and-histogram-exploration-on-iris-data/

# FacetGrid within Seaborn is a multi-plot grid to help visualise distribution of a variable
# iris_df = defining the data input
# hue - allows a variable that determines the colour of the plot elements, in this case it is species that drives the different colours on the visual  
# plt.savefig = saving the histogram as an image to the folder  
# plt.show() = displaying the histogram  

sns.FacetGrid(iris_df,hue="species",height=5).map(sns.histplot,"Petal_length(cm)").add_legend()
plt.savefig("Petal_Length.png")
sns.FacetGrid(iris_df,hue="species",height=5).map(sns.histplot,"Petal_width(cm)").add_legend()
plt.savefig("Petal_Width.png")
sns.FacetGrid(iris_df,hue="species",height=5).map(sns.histplot,"Sepal_length(cm)").add_legend()
plt.savefig("Sepal_Length.png")
sns.FacetGrid(iris_df,hue="species",height=5).map(sns.histplot,"Sepal_width(cm)").add_legend()
plt.savefig("Sepal_Width.png")
plt.show()


# https://www.kaggle.com/adityabhat24/iris-data-analysis-and-machine-learning-python

# Boxplot grouped by Species
# https://www.c-sharpcorner.com/article/a-first-machine-learning-project-in-python-with-iris-dataset/
# To show 4 Boxplots on the one output requires a 2 x 2 (2 columns and 2 rows), therefore subplot(2,2) is required. 
# The 3rd value in the subplot indicates where on the output the plot is shown, as follows 1 - Top Left, 2 - Top Right, 3 - Bottom Left, 4 - Bottom Right, 
# for example (2,2,3) would show onthe bottom left of the output.  
# On the plot the x axis is the species type, y axis is the attribute 
plt.figure(figsize=(15,10))    
plt.subplot(2,2,1)    
sns.boxplot(x='species',y='Sepal_length(cm)',data=iris_df)     
plt.subplot(2,2,2)    
sns.boxplot(x='species',y='Sepal_width(cm)',data=iris_df)     
plt.subplot(2,2,3)    
sns.boxplot(x='species',y='Petal_length(cm)',data=iris_df)     
plt.subplot(2,2,4)    
sns.boxplot(x='species',y='Petal_width(cm)',data=iris_df)
plt.savefig("Box_plot.png")
plt.show()

# Violinplot grouped by series
# To show how the length and width vary according to the species
# To show 4 Violinplots on the one output requires a 2 x 2 (2 columns and 2 rows), therefore subplot(2,2) is required. 
# The 3rd value in the subplot indicates where on the output the plot is shown, as follows 1 - Top Left, 2 - Top Right, 3 - Bottom Left, 4 - Bottom Right, 
# for example (2,2,3) would show onthe bottom left of the output.  
# On the plot the x axis is the species type, y axis is the attribute 
plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='species',y='Petal_length(cm)',data=iris_df)
plt.subplot(2,2,2)
sns.violinplot(x='species',y='Petal_width(cm)',data=iris_df)
plt.subplot(2,2,3)
sns.violinplot(x='species',y='Sepal_length(cm)',data=iris_df)
plt.subplot(2,2,4)
sns.violinplot(x='species',y='Sepal_width(cm)',data=iris_df) 
plt.savefig("Violin_plot.png")
plt.show()


# A “pairs plot” is also known as a scatterplot, in which one variable in the same data row is matched with another variable's value,
#  like this: Pairs plots are just elaborations on this showing all variables paired with all the other variables.
# Scatterplot matrices are very good visualization tools and may help identify correlations or lack of it
# https://www.kaggle.com/biphili/seaborn-matplotlib-iris-data-visualization-code-1
sns.pairplot(iris_df, hue="species", diag_kind="kde")
plt.savefig("Pairsplot.png")
plt.show()


# to show correlation 
print(iris_df.corr())

# Heatmap used to show correlation. 
# As I plan to train algorithms, the number of features and their correlation plays an important role. 
# If there are features and many of the features are highly correlated, then training an algorithm with all the featues will reduce the accuracy. 
# Thus features selection should be done carefully. This dataset has less featues but still we will see the correlation. 
# https://stackabuse.com/ultimate-guide-to-heatmaps-in-seaborn-with-python/

# To show the values on the heatmap, insert "annot = True"
# cmap to pick the colour palette of your choice
# The Sepal Width and Length are not correlated The Petal Width and Length are highly correlated
# https://www.kaggle.com/ash316/ml-from-scratch-with-iris
plt.figure(figsize=(15,10))
sns.heatmap(iris_df.corr(), annot = True, cmap = 'rocket') 
plt.savefig("Heatmap.png")
plt.show()

from sklearn.datasets import load_iris
iris=load_iris()

# X = iris.data  (4 features/variables sepal length, sepal width, petal length, petal width)
# y = iris.target (species)
X=iris.data
y=iris.target

# Splitting the iris dataset 75% for training and 25% for testing
# X_train, y_train for training the model
# X_test, y_test for testing the model


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

# https://medium.com/gft-engineering/start-to-learn-machine-learning-with-the-iris-flower-classification-challenge-4859a920e5e3

# Show the shape/size of the train data samples
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

# I now use the fit method of the knn object, 
# Train the algorithm with the training data X_train (containing the training data of the 4 features)
# and the training output y_train (containing the corresponding species)
# This builds up our model on the training set.
knn.fit(X_train, y_train)

# We enter sample data
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
# Therefore, we can make a prediction for each iris in the test data and compare it against its species — so we can know if the model 
# is correctly predicting the species for a given flower.

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

# For this model, the accuracy on the test set is 0.97, which means the model made the right prediction for 97% of the irises 
# in the given dataset. We can expect the model to be correct 97% of the time for predicting the species of new irises.
# This is a high level of accuracy and it means that our model may be trustworthy enough to use
# 
# While the iris dataset and classification is simple, it is a good example to illustrate 
# how a machine learning problem should be approached and how useful the outcome can be to a potential user

# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
# https://machinelearningmastery.com/make-predictions-scikit-learn/

# I spot Check other Algorithms to see 
# We get an idea from the plots that some of the classes are partially linearly separable in some dimensions, so we are expecting generally good result
# we will test 6 different algorithms 
# This is a good mixture of simple linear (LR and LDA), nonlinear (KNN, CART, NB and SVM) algorithms.

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
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
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# We can also create a plot of the model evaluation results and compare the spread and the mean accuracy of each model. 
# A useful way to compare the samples of results for each algorithm is to create a box and whisker plot for each distribution and compare the distributions.

plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()
# We can see that the box and whisker plots are squashed at the top of the range, 
# with many evaluations achieving 100% accuracy, and some pushing down into the high 80% accuracies.

# Make predictions on test/validation dataset 

# The results in the previous section suggest that the LDA was perhaps the most accurate model. We will use this model as our final model.
# Now we want to get an idea of the accuracy of the model on our validation set.
# This will give us an independent final check on the accuracy of the best model. 
# It is valuable to keep a testing set just in case you made a slip during training, 
# such as overfitting to the training set or a data leak. Both of these issues will result in an overly optimistic result.

# We can fit the model on the entire training dataset and make predictions on the testing dataset.
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
predictions = lda.predict(X_test)

# Evaluate predictions
# We can evaluate the predictions by comparing them to the expected results in the validation set, 
# then calculate classification accuracy, as well as a confusion matrix and a classification report.

print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# We can see that the accuracy is 1.0 or about 100% on the hold out dataset.
# The confusion matrix provides an indication of the errors made.
# Finally, the classification report provides a breakdown of each class by precision,
# recall, f1-score and support showing excellent results (granted the validation dataset was small).
