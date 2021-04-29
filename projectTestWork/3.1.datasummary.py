# Data Summary
# Author : Michelle O'Connor

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Notes and references contain in 2.1 

path = ""
filenameForIrisData = path + "iris.csv"
header_list = ["sepallengthcm", "sepalwidthcm", "petallengthcm", "petalwidthcm", "species"]
df = pd.read_csv(filenameForIrisData)
iris_df = df.rename(columns = {"sepallength" : "sepallengthcm", "sepalwidth" : "sepalwidthcm", "petallength" : "petallengthcm", "petalwidth" : "petalwidthcm", "class" : "species"})
# print (iris_df)

''' 
# to show the shape of the dataset : (150, 5) 150 rows, 5 columns
# https://www.c-sharpcorner.com/article/a-first-machine-learning-project-in-python-with-iris-dataset/
print(iris_df.shape)
 
# Shows the number of instances and the number of attributes in the dataset. 
# There are no null values
print(iris_df.info())

# View how many instances the data frame containsby species
# https://www.c-sharpcorner.com/article/a-first-machine-learning-project-in-python-with-iris-dataset/
print(iris_df.groupby('species').size())   

print(iris_df.describe())
# https://www.kaggle.com/adityabhat24/iris-data-analysis-and-machine-learning-python
# print(iris.df.groupby('Species').size())

''' 
print(iris_df.duplicated().sum())

print (iris_df[iris_df.duplicated()])  

plt.title('Species Count')
sns.countplot(iris_df['species'])
plt.show()

setosa = iris_df.loc[iris_df["species"] == "Iris-setosa"]
#virginica = iris_df.loc[iris_df["species"] == "Iris-virginica"])
versicolor = iris_df.loc[iris_df["species"] == "Iris-versicolor"]

print(iris_df['species'].value_counts())

# print (iris_df[setosa].value_counts())

# print = (iris_df(virginica).describe())  # boolean array expected for the condition not object 
# print(iris_df(setosa).describe())
# print(iris_df['virginica'].describe())
# print(iris_df['Iris-veriscolor'].describe())

print(iris_df.describe(include='all'))