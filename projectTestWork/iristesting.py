import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

path = ""
filenameForIrisData = path + "iris.csv"
header_list = ["sepallengthcm", "sepalwidthcm", "petallengthcm", "petalwidthcm", "species"]
df = pd.read_csv(filenameForIrisData)
iris_df = df.rename(columns = {"sepallength" : "sepallengthcm", "sepalwidth" : "sepalwidthcm", "petallength" : "petallengthcm", "petalwidth" : "petalwidthcm", "class" : "species"})
print (iris_df)

print(iris_df.shape)
print(iris_df.info())
print(iris_df.groupby('species').size())   

sepalLength = (iris_df.groupby("sepallengthcm").describe())
sepalWidth = (iris_df.groupby("sepalwidthcm").describe())
petallLength = (iris_df.groupby("petallengthcm").describe())
petalWidth = (iris_df.groupby("petalwidthcm").describe())

# I need this for the final verison, just commenting out for now
# this is to output a summary of each variable to a single text file
'''
# input has to be in string format
# class lectures
# https://towardsdatascience.com/how-to-use-groupby-and-aggregate-functions-in-pandas-for-quick-data-analysis-c19e7ea76367
with open(".\Variable_Summary.txt", "wt") as i:
    i.write(str(iris_df.groupby("species").describe()))
# print (iris_df.groupby("species").describe())
''' 
# Saves a histogram of each variable to png files
# https://www.geeksforgeeks.org/box-plot-and-histogram-exploration-on-iris-data/
plt.figure(figsize = (10, 7))
x = iris_df["sepallengthcm"]
  
plt.hist(x, bins = 20, color = "green")
plt.title("Sepal Length in cm")
plt.xlabel("Sepal Length cm")
plt.ylabel("Count")
plt.show()
# not saving the image correctly = need to work on
# plt.savefig("Sepal_Length")