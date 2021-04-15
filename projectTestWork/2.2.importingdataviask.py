# importing data using sklearn
# Author : Michelle O'Connor

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
print (iris)

# Import Fisher's Iris data set
# Author : Michelle O'Connor

from sklearn.datasets import load_iris

iris = load_iris()

# https://www.youtube.com/watch?v=Y17Y_8RK6pc Machine Learning in Python: Iris Classification - Part 1
print(iris.keys())

print(iris["DESCR"])

print (iris)

print(iris["feature_names"])