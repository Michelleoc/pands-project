# Box Plot grouped by species
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

plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='species',y='petallengthcm',data=iris_df)
plt.subplot(2,2,2)
sns.violinplot(x='species',y='petalwidthcm',data=iris_df)
plt.subplot(2,2,3)
sns.violinplot(x='species',y='sepallengthcm',data=iris_df)
plt.subplot(2,2,4)
sns.violinplot(x='species',y='sepalwidthcm',data=iris_df)
plt.show()  