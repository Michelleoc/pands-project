# Investigating and analysing the Iris Dataset
# 2021 Programming and Scripting project 
# Author : Michelle O'Connor

# Import all the required libaries 
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
from sklearn.model_selection import train_test_split 

# Import the dataset from iris.csv file (after finding the duplicates this was changed to irisoriginal.csv as I had 2 iris files) 
path = ""
filenameForIrisData = path + "irisoriginal.csv"
dataframe = pd.read_csv(filenameForIrisData)

# I have 2 dataframes: iris_dataframe using the original incorrect file and later on I use iris_df for the correct file)

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
# I now use the iris.csv and the iris_df dataset

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
sns.histplot(data=iris_df, x="Petal_width(cm)", color="teal", ax=axs[1, 1])
# Save and show the plot
plt.savefig("Plot_Images/Features_Histogram.png")
plt.show()

# Plotting, displaying and saving a histogram of each variable to png files  
# https://www.kaggle.com/dhruvmak/iris-flower-classification-with-eda

# FacetGrid within Seaborn is a multi-plot grid to help visualise distribution of a variable
# iris_df = defining the data input
# hue - allows a variable that determines the colour of the plot elements, in this case it is species that drives the different colours on the visual  
# histplot to determine a histogram output
# adding in feature by feature on separate rows 
# add_legend - to add a legend to each visual 
# plt.savefig = saving the histogram as an image to the folder  
# plt.show() = displaying the histogram  

sns.FacetGrid(iris_df,hue="species",height=5).map(sns.histplot,"Petal_length(cm)").add_legend()
plt.savefig("Plot_Images/Petal_Length.png")
sns.FacetGrid(iris_df,hue="species",height=5).map(sns.histplot,"Petal_width(cm)").add_legend()
plt.savefig("Plot_Images/Petal_Width.png")
sns.FacetGrid(iris_df,hue="species",height=5).map(sns.histplot,"Sepal_length(cm)").add_legend()
plt.savefig("Plot_Images/Sepal_Length.png")
sns.FacetGrid(iris_df,hue="species",height=5).map(sns.histplot,"Sepal_width(cm)").add_legend()
plt.savefig("Plot_Images/Sepal_Width.png")
plt.show()


# Boxplot grouped by Species
# https://www.c-sharpcorner.com/article/a-first-machine-learning-project-in-python-with-iris-dataset/
# To show 4 Boxplots on the one output requires a 2 x 2 (2 columns and 2 rows), therefore subplot(2,2) is required. 
# The 3rd value in the subplot indicates where on the output the plot is shown, as follows 1 - Top Left, 2 - Top Right, 3 - Bottom Left, 4 - Bottom Right, 
# for example (2,2,3) would show onthe bottom left of the output.  
# On the plot the x axis is the species type, y axis is the feature
# dataset = iris_df 

plt.figure(figsize=(15,10))    
plt.subplot(2,2,1)    
sns.boxplot(x='species',y='Sepal_length(cm)',data=iris_df)     
plt.subplot(2,2,2)    
sns.boxplot(x='species',y='Sepal_width(cm)',data=iris_df)     
plt.subplot(2,2,3)    
sns.boxplot(x='species',y='Petal_length(cm)',data=iris_df)     
plt.subplot(2,2,4)    
sns.boxplot(x='species',y='Petal_width(cm)',data=iris_df)
plt.savefig("Plot_Images/Box_plot.png")
plt.show()

# Violinplot grouped by series
# To show how the length and width vary according to the species
# To show 4 Violinplots on the one output requires a 2 x 2 (2 columns and 2 rows), therefore subplot(2,2) is required. Similar to boxplot
# The 3rd value in the subplot indicates where on the output the plot is shown, as follows 1 - Top Left, 2 - Top Right, 3 - Bottom Left, 4 - Bottom Right, 
# for example (2,2,3) would show onthe bottom left of the output.  
# On the plot the x axis is the species type, y axis is the feature 
# dataset = iris_df 

plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='species',y='Sepal_length(cm)',data=iris_df)
plt.subplot(2,2,2)
sns.violinplot(x='species',y='Sepal_width(cm)',data=iris_df)
plt.subplot(2,2,3)
sns.violinplot(x='species',y='Petal_length(cm)',data=iris_df)
plt.subplot(2,2,4)
sns.violinplot(x='species',y='Petal_width(cm)',data=iris_df) 
plt.savefig("Plot_Images/Violin_plot.png")
plt.show()


# A “pairs plot” is also known as a scatterplot, in which one variable in the same data row is matched with another variable's value,
#  like this: Pairs plots are just elaborations on this showing all variables paired with all the other variables.
# Scatterplot matrices are very good visualization tools and may help identify correlations or lack of it
# Each iris species scatters plots are represented in different colours 
# the 'hue' option allows a variable that determines the colour of the plot elements
# in this case it is species that drives the different colours on the visual. 
# Diag_kind="kde" determines that the plot of the diagonal subplots, I have chosen a density plot kde

sns.pairplot(iris_df, hue="species", diag_kind="kde")
plt.savefig("Plot_Images/Pairsplot.png")
plt.show()


# to show correlation 
print(iris_df.corr())

# Heatmap used to show correlation. 
# As I plan to train algorithms, the number of features and their correlation plays an important role. 
# If there are features and many of the features are highly correlated, then training an algorithm with all the featues will reduce the accuracy. 
# https://stackabuse.com/ultimate-guide-to-heatmaps-in-seaborn-with-python/

# To show the values on the heatmap, insert "annot = True"
# cmap to pick the colour palette of your choice, I have chosen the "rocket" colour
# https://www.kaggle.com/ash316/ml-from-scratch-with-iris
plt.figure(figsize=(15,10))
sns.heatmap(iris_df.corr(), annot = True, cmap = 'rocket') 
plt.savefig("Plot_Images/Heatmap.png")
plt.show()

# MACHINE LEARNING CODE 

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
# i have chosen one neighbor (k=1), this is known as nearest neighbour algorithm 
knn = KNeighborsClassifier(n_neighbors=1)

# I now use the fit method of the knn object, 
# Train the algorithm with the training data X_train (containing the training data of the 4 features)
# and the training output y_train (containing the corresponding species)
# This builds up our model on the training set.
knn.fit(X_train, y_train)

# We enter sample data in as an array
X_new = np.array([[5, 2.9, 1, 0.2]])
# Show the shape of the data, it is one row (1 sample) with 4 columns of data (the 4 features/variables sepal and petal measurements)
print("X_new.shape: {}".format(X_new.shape))

# I now use the predict method of the knn object to predict the species of the sample data X_new
prediction = knn.predict(X_new)
# output prediction of species '0', '1' or '2'
print("Prediction: {}".format(prediction))
# output the species name 
print("Predicted target name: {}".format(iris['target_names'][prediction]))



# But how can I trust the results of the model 
# The test set that was created was not used to build the model, but I do know the correct species for each iris in the test set. 
# Therefore, I can make a prediction for each iris in the test data and compare it against its species — so I can know if the model 
# is correctly predicting the species for a given flower.

# To measure how well the model works, I can obtain the accuracy - 
# the fraction of flowers for which the right species was predicted 
# (number that can be calculated using the NumPy “mean” method, comparing both datasets)
# using the X_test data containing the testing data of the 4 features 
y_pred = knn.predict(X_test)

# we print out the predictions of the species using only the 4 features
print("Test set predictions:\n {}".format(y_pred))  

# we then compare the predictions of the species 'y_pred' to the actual species 'y_test'
print("Test set score (np.mean): {:.2f}".format(np.mean(y_pred == y_test)))  

# another way to do this would be to use the score method of the knn object, which will compute the test set accuracy
print("Test set score (knn.score): {:.2f}".format(knn.score(X_test, y_test)))

# For this model, the accuracy on the test set is 0.97, which means the model made the right prediction for 97% of the irises 
# in the given dataset. We can expect the model to be correct 97% of the time for predicting the species of new irises.
# This is a high level of accuracy

# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
# https://machinelearningmastery.com/make-predictions-scikit-learn/

# I spot Check other Algorithms to see their results. 
# Test 6 different algorithms 
# This is a good mixture of simple linear (LR and LDA), nonlinear (KNN, CART, NB and SVM) algorithms.
# for LR, a solver must be selected for the algorithm to compute. I have selected the Liblinear as it applies automatic parameter selection
# for SVC, gamma relates to the scale, I have set it to auto 

models = []
models.append(('LR', LogisticRegression(solver='liblinear')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn 
# Cross-validation is primarily used in applied machine learning to estimate the skill of a machine learning model on unseen data. 
# start with creating a results and names list  
results = []
names = []

for name, model in models:
	# With KFolds and shuffle, the data is shuffled once at the start, and then divided into the number of splits set out (in this case 10 splits)
	# I set shuffle to True and random state to 1 to avoid repeat shuffling/splits and overlap of data.
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	# Calculate the results based on the model, the training data. 
	# There is an option to set the scorer object with the scoring parameter
	# I have set the scoring method to be accuracy to get the count of correct predictions.
	cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
	# populate the results into the list  
	results.append(cv_results)
	names.append(name)
	# print the results of each model 
	# Accruacy score will give the mean % of the number correct results and the standard deviation for this %
	print('%s: accuracy %f with a standard deviation of (%f)' % (name, cv_results.mean(), cv_results.std()))

# We can also create a plot of the model evaluation results and compare the spread and the mean accuracy of each model. 
# A useful way to compare the samples of results for each algorithm is to create a box and whisker plot for each distribution and compare the distributions.

# using the results and the model names, generate a boxplot 
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show() 

# We can see that the box and whisker plots are squashed at the top of the range, 
# with many evaluations achieving 100% accuracy, and some pushing down into the high 80% accuracies.

# Make predictions on test/validation dataset 

# Based on the results in the previous section pick the most accurate model as the final model
# in this case it is the LDA 
# Now I calculate the accuracy of the model on the testing set.

# Similar to Step 4 above, I pass the training set to the LDA algorithm. I fit the model on the entire training dataset 
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train) 

# And then make predictions on the testing dataset
# the predictions from the algorithm are labelled as LDApredicitions. 
LDApredictions = lda.predict(X_test)

# Evaluate predictions
# Evaluate the predictions (LDApredicitions) of the LDA model by comparing them  
# to the expected results in the testing set (y_test), i.e. the species of the test dataset.  
# then calculate classification accuracy to 3 decimal places, as well as a confusion matrix.

print("Accruacy score: {:.3f}".format(accuracy_score(y_test, LDApredictions)))

print(confusion_matrix(y_test, LDApredictions))

# can see that the accuracy is 1.0/100% on the testing dataset.
# The confusion matrix provides an indication of the errors made.
