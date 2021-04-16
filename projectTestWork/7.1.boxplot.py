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


# https://www.kaggle.com/adityabhat24/iris-data-analysis-and-machine-learning-python

'''
# Boxplot grouped by Species
# https://www.c-sharpcorner.com/article/a-first-machine-learning-project-in-python-with-iris-dataset/ 
# for individual boxplot charts
plt.figure(figsize=(15,10))    
plt.subplot(2,2,1)    
sns.boxplot(x='species',y='sepallengthcm',data=iris_df)
plt.show()
plt.figure(figsize=(15,10))        
plt.subplot(2,2,2)    
sns.boxplot(x='species',y='sepalwidthcm',data=iris_df)   
plt.show()
plt.figure(figsize=(15,10))     
plt.subplot(2,2,3)    
sns.boxplot(x='species',y='petallengthcm',data=iris_df)    
plt.show()
plt.figure(figsize=(15,10))    
plt.subplot(2,2,4)    
sns.boxplot(x='species',y='petalwidthcm',data=iris_df)
plt.show()
'''

# to combine all 4 boxplots charts into one chart output 
plt.figure(figsize=(15,10))    
plt.subplot(2,2,1)    
sns.boxplot(x='species',y='sepallengthcm',data=iris_df)     
plt.subplot(2,2,2)    
sns.boxplot(x='species',y='sepalwidthcm',data=iris_df)     
plt.subplot(2,2,3)    
sns.boxplot(x='species',y='petallengthcm',data=iris_df)     
plt.subplot(2,2,4)    
sns.boxplot(x='species',y='petalwidthcm',data=iris_df)
plt.savefig("Box_plot.png")
plt.show()

# https://aiaspirant.com/box-plot/
# Boxplot represents the minimum, first quartile, median, third quartile and maximun of the dataset. 
# The box shows the between the first and third quartile on the range. The horizontal line that goes through the box is the median.
# The top and bottom lines known as whiskers extend from the ends of the box to the minimun and maximum value. 
# Any data points past the whiskers (represent by a diamon) are considered as outliers.

# Analyising the box plots, while on the sepal length and sepal width the data results are close. 
# However for the petal length and petal width the Iris Setosa is very clearly different from the Veriscolor and Virginica. 
# The maximun value for these 2 attritibutes does not overlap with the Veriscolor and Virginica. 
