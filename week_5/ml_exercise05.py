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

## 1 b)
## predict classes for test data, report 0-1-loss

setosa_pred = np.dot(test, w_setosa)
versicolor_pred = np.dot(test, w_versicolor)
virginica_pred = np.dot(test, w_virginica)

# print(setosa_pred, versicolor_pred, virginica_pred)
## iterate all predictions, take max predicted value for that class

pred = np.array([])
for i in range(len(test)):
    entries = np.array([setosa_pred[i], versicolor_pred[i], virginica_pred[i]])
    pred_class = np.argmax(entries) ## gives index, that is our class! (0 = setosa, 1 = versicolor, 2 = virginica)
    pred = np.append(pred, pred_class)

print('Predicted classes for the test data:\n{}\n'.format(pred))

def loss_0_1(Y, Y_pred):
    return int(Y != Y_pred)

print('Verify predicted classes against test_label:\n{}\n'.format(test_label))

losses = np.array([])
for entry in zip(pred, test_label):
    losses = np.append(losses, loss_0_1(entry[0], entry[1]))

print('0/1 Losses for all entries:\n{}\n'.format(losses))