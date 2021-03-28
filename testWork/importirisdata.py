# testing importing iris data
# Author : Michelle O'Connor

import pandas as pd

path = ""
filenameForIrisData = path + "iris.csv"

df = pd.read_csv(filenameForIrisData)

print (df)

# reference https://www.geeksforgeeks.org/python-basics-of-pandas-using-iris-dataset/


# to get the sum, mean and mediam of a column
# df["column_name"].sum()

sum_df = df["sepallength"].sum()
mean_df = df["sepallength"].mean()
median_df = df["sepallength"].median()

print("Sum:",sum_df, "\nMean:", mean_df, "\nMedian:",median_df)

min_df=df["sepallength"].min()
max_df=df["sepallength"].max()

print("Minimum:",min_df, "\nMaximum:", max_df)


# next day change the titles of the columns to include CM in the title

