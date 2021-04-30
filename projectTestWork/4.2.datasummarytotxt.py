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

# print(iris_df.info())   


with open("output2.txt", "wt") as f:
    # f.write("blah the blah \n")  
    print ("Summary of the Iris Dataset variables (Features and Species) \n", file = f) 
    print ("Shape of Data \n", str(iris_df.shape),"\n", file = f) # can also use print () option to add/write data into the text file
    print ("Count by Species \n", str(iris_df.groupby('species').size()),"\n", file = f)
    print ("Statistical Data of Dataset by feature \n", str(iris_df.describe()),"\n", file = f)
    print ("Summary of each feature by species \n",str(iris_df.groupby("species").describe()), "\n", file = f)
    # print ("Number of attributes and null values in Dataset \n", str(iris_df.info()),"\n", file = f)