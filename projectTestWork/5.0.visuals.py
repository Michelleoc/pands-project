# Visuals
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

# also highlight in the project
print(iris_df.duplicated().sum())

print (iris_df[iris_df.duplicated()])

# plt.title('Species Count')
sns.countplot(iris_df['species'])
plt.show()

plt.figure(figsize=(16,9))
plt.title('Comparison between sepal width and length on the basis of species')
sns.scatterplot(iris_df['sepallengthcm'], iris_df['sepalwidthcm'], hue = iris_df['species'], s= 50)
plt.show()


plt.figure(figsize=(16,9))
plt.title('Comparison between petal width and length on the basis of species')
sns.scatterplot(iris_df['petallengthcm'], iris_df['petalwidthcm'], hue = iris_df['species'], s= 50)
plt.show()

# Important
# https://towardsdatascience.com/eda-of-the-iris-dataset-190f6dfd946d



'''

# https://www.kaggle.com/kamrankausar/iris-dataset-ml-and-deep-learning-from-scratch
# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

1. Descriptive statistics

2. Class Distribution (Species counts are balanced or imbalanced)

3. Univariate Plots:- Understand each attribute better.
We start with some univariate plots, that is, plots of each individual variable
Given that the input variables are numeric, we can create box and whisker plots of each.
This gives us a much clearer idea of the distribution of the input attributes:

   3.1 Box Plot - Distribution of attribute through their quartiles & find outlier

We can also create a histogram of each input variable to get an idea of the distribution.
It looks like perhaps two of the input variables have a Gaussian distribution. This is useful to note as we can use algorithms that can exploit this assumption.

   3.2 Histogram - Distrbution of attribute through their bin, we find the distribution of attribute follow Gaussian or other distributions

4. Multivariate Plots :- Understand the relationships between attributes & species better. (Which attributes contributes a lot in classifying species) 4.0 Scatter Plot - Sepal_Length_Width Vs Species.

Now we can look at the interactions between the variables.

First, let’s look at scatterplots of all pairs of attributes. This can be helpful to spot structured relationships between input variables.
Note the diagonal grouping of some pairs of attributes. This suggests a high correlation and a predictable relationship.

    4.1 Scatter Plot - Petal_Length_Width Vs Species.

    4.2 Scatter Plot of all the attributes

    4.3 3D Plot

    4.4 Violinplot - Density of the length and width in the species
'''