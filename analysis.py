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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Import the dataset from iris.csv file 
path = ""
filenameForIrisData = path + "iris.csv"
df = pd.read_csv(filenameForIrisData)

# Rename column titles
# https://www.geeksforgeeks.org/python-pandas-dataframe-rename/
iris_df = df.rename(columns = {"sepallength" : "sepallengthcm", "sepalwidth" : "sepalwidthcm", "petallength" : "petallengthcm", "petalwidth" : "petalwidthcm", "class" : "species"})

# To show the full list print (iris_df), but for now I just want to see the first 5 rows of data 
print(iris_df.head(5))   

# to show the shape of the dataset : (150, 5) 150 rows, 5 columns
# https://www.c-sharpcorner.com/article/a-first-machine-learning-project-in-python-with-iris-dataset/
print(iris_df.shape)

# Shows the number of instances and the number of attributes in the dataset. 
# Show if there are any null values
print(iris_df.info())

# View how many instances the data frame containsby species
# https://www.c-sharpcorner.com/article/a-first-machine-learning-project-in-python-with-iris-dataset/
print(iris_df.groupby('species').size())   

# Shows the basic statistical details of a the dataframe (iris dataset)
# https://www.kaggle.com/adityabhat24/iris-data-analysis-and-machine-learning-python
print(iris_df.describe())

# this is to output a summary of each variable to a single text file
# input has to be in string format
# class lectures showed how to export to a newly created txt file
# https://towardsdatascience.com/how-to-use-groupby-and-aggregate-functions-in-pandas-for-quick-data-analysis-c19e7ea76367
with open(".\Variable_Summary.txt", "wt") as i:
    i.write(str(iris_df.groupby("species").describe()))

# Plotting, displaying and saving a histogram of each variable to png files
# https://www.geeksforgeeks.org/box-plot-and-histogram-exploration-on-iris-data/

# plt.figure(figsize = (10, 7)) = set the size (in inches)of the histogram
# x = iris_df["sepallengthcm"] = defining the data input
# plt.hist(x, bins = 20, color = "green") = plotting the histogram (x=input data, bin = number of columns, colour of bin/column = green)
# plt.title("Sepal Length in cm") = adding a title
# plt.xlabel("Sepal Length cm") = adding a name to the xlabel (horizontal line)
# plt.ylabel("Count") = adding a name to the ylabel (vertical line)
# plt.savefig("Sepal_Length.png") = saving the histogram as an image to the folder
# plt.show() = displaying the histogram

plt.figure(figsize = (10, 7))
x = iris_df["sepallengthcm"]
plt.hist(x, bins = 20, color = "green")
plt.title("Sepal Length in cm")
plt.xlabel("Sepal Length cm")
plt.ylabel("Count")
# plt.savefig("Sepal_Length.png")
plt.show()

plt.figure(figsize = (10, 7))
x = iris_df["sepalwidthcm"]
plt.hist(x, bins = 20, color = "red")
plt.title("Sepal Width in cm")
plt.xlabel("Sepal Width cm")
plt.ylabel("Count")
# plt.savefig("Sepal_Width.png")
plt.show()

plt.figure(figsize = (10, 7))
x = iris_df["petallengthcm"]
plt.hist(x, bins = 20, color = "yellow")
plt.title("Petal Length in cm")
plt.xlabel("Petal Length cm")
plt.ylabel("Count")
# plt.savefig("Petal_Length.png")
plt.show()

plt.figure(figsize = (10, 7))
x = iris_df["petalwidthcm"]
plt.hist(x, bins = 20, color = "blue")
plt.title("Petal Width in cm")
plt.xlabel("Petal Width cm")
plt.ylabel("Count")
# plt.savefig("Petal_Width.png")
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
sns.boxplot(x='species',y='sepallengthcm',data=iris_df)     
plt.subplot(2,2,2)    
sns.boxplot(x='species',y='sepalwidthcm',data=iris_df)     
plt.subplot(2,2,3)    
sns.boxplot(x='species',y='petallengthcm',data=iris_df)     
plt.subplot(2,2,4)    
sns.boxplot(x='species',y='petalwidthcm',data=iris_df)
plt.savefig("Box_plot.png")
plt.show()

# A “pairs plot” is also known as a scatterplot, in which one variable in the same data row is matched with another variable's value,
#  like this: Pairs plots are just elaborations on this showing all variables paired with all the other variables.
# https://www.kaggle.com/biphili/seaborn-matplotlib-iris-data-visualization-code-1
sns.pairplot(iris_df, hue="species", diag_kind="kde")
plt.show()