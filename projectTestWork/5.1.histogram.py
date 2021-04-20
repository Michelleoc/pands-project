# Histograms
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
# Saves a histogram of each variable to png files
# https://www.geeksforgeeks.org/box-plot-and-histogram-exploration-on-iris-data/
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
plt.hist(x, bins =20, color = "blue")
plt.title("Petal Width in cm")
plt.xlabel("Petal Width cm")
plt.ylabel("Count")
# plt.savefig("Petal_Width.png")
plt.show()
'''


# histograms by species 
sns.FacetGrid(iris_df,hue="species",height=5).map(sns.histplot,"petallengthcm").add_legend()
plt.savefig("Petal_Length.png")
sns.FacetGrid(iris_df,hue="species",height=5).map(sns.histplot,"petalwidthcm").add_legend()
plt.savefig("Petal_Width.png")
sns.FacetGrid(iris_df,hue="species",height=5).map(sns.histplot,"sepallengthcm").add_legend()
plt.savefig("Sepal_Length.png")
sns.FacetGrid(iris_df,hue="species",height=5).map(sns.histplot,"sepalwidthcm").add_legend()
plt.savefig("Sepal_Width.png")
plt.show()

