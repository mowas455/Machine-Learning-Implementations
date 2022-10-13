# Libaraies
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

# dataset
iris = datasets.load_iris()
x, y = iris.data, iris.target

# train test split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

# shape of the data
print(x_train.shape)
print(x_test.shape)

# plot show the orginal 
plt.figure()
plt.scatter(x[:,0], x[:, 1], c = y, cmap=cmap,edgecolors='k', s= 20)
plt.show()

# From the scratch model of knn
from knntest import KNN 
clf = KNN(k = 5)
clf.fit(x_train,y_train)
predictions = clf.predict(x_test)

acc = np.sum(predictions == y_test) / len(y_test)
print(acc)

# This is directly the sklearn libaray
from sklearn.neighbors import KNeighborsClassifier
