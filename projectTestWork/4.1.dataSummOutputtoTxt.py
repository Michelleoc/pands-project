# Data Summary output to text file
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


# to show the shape of the dataset : (150, 5) 150 rows, 5 columns
# https://www.c-sharpcorner.com/article/a-first-machine-learning-project-in-python-with-iris-dataset/
print(iris_df.shape)

# Shows the number of instances and the number of attributes in the dataset. 
# There are no null values
print(iris_df.info())

# View how many instances the data frame containsby species
# https://www.c-sharpcorner.com/article/a-first-machine-learning-project-in-python-with-iris-dataset/
print(iris_df.groupby('species').size())   

print(df.describe())
# https://www.kaggle.com/adityabhat24/iris-data-analysis-and-machine-learning-python
# print(df.groupby('Species').size())



print(iris_df.groupby("species").describe())

# this is to output a summary of each variable to a single text file
# input has to be in string format
# class lectures showed how to export to a newly created txt file
# https://towardsdatascience.com/how-to-use-groupby-and-aggregate-functions-in-pandas-for-quick-data-analysis-c19e7ea76367
with open(".\Variable_Summary.txt", "wt") as i:
    i.write(str(iris_df.groupby("species").describe()))
# print (iris_df.groupby("species").describe())
