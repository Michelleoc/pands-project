import numpy as np
import sklearn 

# import load_iris function from datasets module
# convention is to import modules instead of sklearn as a whole
from sklearn.datasets import load_iris
# save "bunch" object containing iris dataset and its attributes
# the data type is "bunch"
iris = load_iris()
type(iris)
# sklearn.datasets.base.Bunch
# print the iris data
# same data as shown previously
# each row represents each sample
# each column represents the features
print(iris.data)

'''
# print the names of the four features
print iris.feature_names
# print integers representing the species of each observation
# 0, 1, and 2 represent different species
print iris.target
# print the encoding scheme for species: 0 = setosa, 1 = versicolor, 2 = virginica
print iris.target_names
''' 
# check the types of the features and response
print(type(iris.data))
print(type(iris.target))
<class 'np.ndarray'>
<class 'np.ndarray'>
# check the shape of the features (first dimension = number of observations, second dimensions = number of features)
print(iris.data.shape)
(150, 4)
# check the shape of the response (single dimension matching the number of observations)
print(iris.target.shape)
(150,)
# store feature matrix in "X"
X = iris.data

# store response vector in "y"
y = iris.target