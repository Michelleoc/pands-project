# Scatter Plot for each pair of variables
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


# For each pair of attributes, we can use a scatter plot to visualize their joint distribution
# http://www.cse.msu.edu/~ptan/dmbook/tutorials/tutorial3/tutorial3.html
fig, axes = plt.subplots(3, 2, figsize=(12,12))
index = 0
for i in range(3):
    for j in range(i+1,4):
        ax1 = int(index/2)
        ax2 = index % 2
        axes[ax1][ax2].scatter(iris_df[iris_df.columns[i]], iris_df[iris_df.columns[j]], color='red')
        axes[ax1][ax2].set_xlabel(iris_df.columns[i])
        axes[ax1][ax2].set_ylabel(iris_df.columns[j])
        index = index + 1
plt.show()