# importing data via csv
# Author : Michelle O'Connor

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Import the dataset from iris.csv file 

path = ""
filenameForIrisData = path + "iris.csv"
# header_list = ["sepallengthcm", "sepalwidthcm", "petallengthcm", "petalwidthcm", "species"]
df = pd.read_csv(filenameForIrisData)

# Rename column titles
# https://www.geeksforgeeks.org/python-pandas-dataframe-rename/
iris_df = df.rename(columns = {"sepallength" : "sepallengthcm", "sepalwidth" : "sepalwidthcm", "petallength" : "petallengthcm", "petalwidth" : "petalwidthcm", "class" : "species"})

# https://www.youtube.com/watch?v=Y17Y_8RK6pc Machine Learning in Python: Iris Classification - Part 1

print(iris_df.head(5))   

# print (iris_df)

# https://www.geeksforgeeks.org/python-basics-of-pandas-using-iris-dataset/
