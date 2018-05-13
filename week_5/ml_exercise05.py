#!/bin/python3

import numpy as np
import pandas as pd
from scipy import io as sio
import matplotlib.pyplot as plt
from numpy.linalg import inv

##### assignment 1

matfile = sio.loadmat('./iris_multiclass.mat')
print('Keys in the matlab file', matfile.keys())

indices_train = matfile['indices_train']-1 # subtract 1, because indexing is not zero-based..
indices_test = matfile['indices_test']-1
species = matfile['species']
measurements = matfile['meas']
# measurements is a 150 x 4 np.array. the four entries are: sepal length, sepal width, petal length and petal width.
# sepal = "kelchblatt", petal = "blütenblatt"

## take indices train / test from the measurements data 
train = measurements[indices_train][0]
train_label_setosa = np.zeros(len(train))
train_label_setosa[(species[indices_train] == 'setosa').flatten()] = 1
train_label_virginica = np.zeros(len(train))
train_label_virginica[(species[indices_train] == 'virginica').flatten()] = 1
train_label_versicolor = np.zeros(len(train))
train_label_versicolor[(species[indices_train] == 'versicolor').flatten()] = 1

# print(train_label_setosa, train_label_versicolor, train_label_virginica)

test = measurements[indices_test][0]
test_label = np.zeros(len(test))
test_label[(species[indices_test] == 'versicolor').flatten()] = 1
test_label[(species[indices_test] == 'virginica').flatten()] = 2


## 1 a)
print('1 a)')
def least_squares(X, Y):
    X_t = X.transpose()
    left = inv(np.dot(X_t, X))
    right = np.dot(X_t, Y)
    return np.dot(left, right)

w_setosa = least_squares(train, train_label_setosa)
w_versicolor = least_squares(train, train_label_versicolor)
w_virginica = least_squares(train, train_label_virginica)

print('Estimate regressions 1-vs-all classes.')
print('w_setosa: {},\nw_versicolor: {},\nw_virginica: {}\n'.format(w_setosa, w_versicolor, w_virginica))
