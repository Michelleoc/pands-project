# Data Summary using sklearn
# Author : Michelle O'Connor

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets


from sklearn.datasets import load_iris

iris = load_iris()

#data=pd.DataFrame(iris[‘data’],columns=[“Petal length”,”Petal Width”,”Sepal Length”,”Sepal Width”])
#data[‘Species’]=iris[‘target’]
# data[‘Species’]=data[‘Species’].apply(lambda x: iris[‘target_names’][x])


print(iris.data.shape)
print(iris.feature_names)
print(iris.target)
print(iris.target_names)
print(type('iris.data'))
print(type('iris.target'))

from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
iris = load_iris()
# The indices of the features that we are plotting
x_index = 0
y_index = 1
 
# this formatter will label the colorbar with the correct target names
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])
 
plt.figure(figsize=(5, 4))
plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.title('Iris Dataset Scatterplot')
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])
 
plt.tight_layout()
plt.show()