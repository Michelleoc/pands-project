# testing importing iris data
# Author : Michelle O'Connor

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# for colours in the visualuations??
# sns.set(color_codes=True)
 

# Import the dataset from iris.csv file 

path = ""
filenameForIrisData = path + "iris.csv"
header_list = ["sepallengthcm", "sepalwidthcm", "petallengthcm", "petalwidthcm", "species"]
df = pd.read_csv(filenameForIrisData)

# Rename column titles
# https://www.geeksforgeeks.org/python-pandas-dataframe-rename/
iris_df = df.rename(columns = {"sepallength" : "sepallengthcm", "sepalwidth" : "sepalwidthcm", "petallength" : "petallengthcm", "petalwidth" : "petalwidthcm", "class" : "species"})
print (iris_df)

# reference https://www.geeksforgeeks.org/python-basics-of-pandas-using-iris-dataset/

# to get the sum, mean and mediam of a column
# df["column_name"].sum()

sum_df = df["sepallength"].sum()
mean_df = df["sepallength"].mean()
median_df = df["sepallength"].median()
print("Sum:",sum_df, "\nMean:", mean_df, "\nMedian:",median_df)
min_df=df["sepallength"].min()
max_df=df["sepallength"].max()
print("Minimum:",min_df, "\nMaximum:", max_df)

# to show the shape of the dataset : (150, 5) 150 rows, 5 columns
# https://www.c-sharpcorner.com/article/a-first-machine-learning-project-in-python-with-iris-dataset/
print(iris_df.shape)

# Shows the number of instances and the number of attributes in the dataset. 
# There are no null values
print(iris_df.info())

# View how many instances the data frame contains
# https://www.c-sharpcorner.com/article/a-first-machine-learning-project-in-python-with-iris-dataset/
print(iris_df.groupby('species').size())   

print(df.describe())
# https://www.kaggle.com/adityabhat24/iris-data-analysis-and-machine-learning-python
# print(df.groupby('Species').size())


from pandas.plotting import scatter_matrix
# scatter plot matrix
# scatter_matrix(iris_df,figsize=(10,10))

# https://www.kaggle.com/adityabhat24/iris-data-analysis-and-machine-learning-python

# sns.pairplot(iris_df, hue="species")
# plt.show()

'''
Just commenting out for now. Note this works and I want to show it 
sns.pairplot(iris_df, hue="species", diag_kind="kde")
plt.show()
''' 
# https://www.kaggle.com/adityabhat24/iris-data-analysis-and-machine-learning-python


# https://www.c-sharpcorner.com/article/a-first-machine-learning-project-in-python-with-iris-dataset/
plt.figure(figsize=(15,10))    
plt.subplot(2,2,1)    
sns.boxplot(x='species',y='sepallengthcm',data=iris_df)
plt.show()    
plt.subplot(2,2,2)    
sns.boxplot(x='species',y='sepalwidthcm',data=iris_df)   
plt.show() 
plt.subplot(2,2,3)    
sns.boxplot(x='species',y='petallengthcm',data=iris_df)    
plt.show()
plt.subplot(2,2,4)    
sns.boxplot(x='species',y='petalwidthcm',data=iris_df)
plt.show()



''' 
data.isnull()
#if there is data is missing, it will display True else False.
data.isnull.sum()

# https://www.c-sharpcorner.com/article/a-first-machine-learning-project-in-python-with-iris-dataset/
# A first machine learning project in python with Iris dataset
Iris data set is the famous smaller databases for easier visualization and analysis techniques. In this article, we will see a quick view of how to develop machine learning “hello world program”.
 

# For example, if we want to add a column let say "total_values",
# that means if you want to add all the integer value of that particular
# row and get total answer in the new column "total_values".
# first we will extract the columns which have integer values.
cols = data.columns

# it will print the list of column names.
print(cols)

# we will take that columns which have integer values.
cols = cols[1:5]

# we will save it in the new dataframe variable
data1 = data[cols]

# now adding new column "total_values" to dataframe data.
data["total_values"]=data1[cols].sum(axis=1)

# here axis=1 means you are working in rows,
# whereas axis=0 means you are working in columns.


newcols={
"Id":"id",
"SepalLengthCm":"sepallength"
"SepalWidthCm":"sepalwidth"}

data.rename(columns=newcols,inplace=True)

print(data.head())

#this is an example of rendering a datagram,
which is not visualised by any styles.
data.style


# we will here print only the top 10 rows of the dataset,
# if you want to see the result of the whole dataset remove
#.head(10) from the below code

data.head(10).style.highlight_max(color='lightgreen', axis=0)

data.head(10).style.highlight_max(color='lightgreen', axis=1)

data.head(10).style.highlight_max(color='lightgreen', axis=None)

'''