# Heatmaps
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

# to show correlation 
print(iris_df.corr())

# Heatmap used to show correlation 
# https://stackabuse.com/ultimate-guide-to-heatmaps-in-seaborn-with-python/

# To show the values on the heatmap, insert "annot = True"
# cmap to pick the colour palette of your choice
plt.figure(figsize=(15,10))
sns.heatmap(iris_df.corr(), annot = True, cmap = 'rocket') 
plt.show()