# Pairs Plot 
# Author : Michelle O'Connor

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Notes and references contain in 2.1 

path = ""
filenameForIrisData = path + "iris.csv"
header_list = ["sepallengthcm", "sepalwidthcm", "pcd etallengthcm", "petalwidthcm", "species"]
df = pd.read_csv(filenameForIrisData)
iris_df = df.rename(columns = {"sepallength" : "sepallengthcm", "sepalwidth" : "sepalwidthcm", "petallength" : "petallengthcm", "petalwidth" : "petalwidthcm", "class" : "species"})
# print (iris_df)

# A “pairs plot” is also known as a scatterplot, in which one variable in the same data row is matched with another variable's value,
#  like this: Pairs plots are just elaborations on this showing all variables paired with all the other variables.
# https://www.kaggle.com/biphili/seaborn-matplotlib-iris-data-visualization-code-1
sns.pairplot(iris_df, hue="species", diag_kind="kde")
# sns.pairplot(iris_df, hue="species", diag_kws={Petal, "kde")
plt.show()

from pandas.plotting import scatter_matrix
# scatter plot matrix
# scatter_matrix(iris_df,figsize=(10,10))

# https://www.kaggle.com/adityabhat24/iris-data-analysis-and-machine-learning-python

# sns.pairplot(iris_df, hue="species")
# plt.show()