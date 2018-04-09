#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np 

## basic plot labels
def addLabels():
    plt.title('Traffic per Hour')
    plt.ylabel('Traffic')
    plt.xlabel('Hours')

## read file into numpy
file = './traffic_per_hour.csv'
arr = np.fromfile(file, dtype='float', sep='\t').reshape(-1, 2)

## remove nan
arr = arr[~np.isnan(arr).any(axis=1)]

## ex 1: scatter to plot
plt.scatter(arr[:,0], arr[:,1])
addLabels()
plt.savefig('plots/traffic_per_hour.png')
plt.close()

## ex2: fitting and prediction

## change the degree variable to plot different polynoms (linear, quadratic, cubic..)
for degree in range(1,5):
    fit = np.polyfit(arr[:,0], arr[:,1], deg=degree)
    fit_fn = np.poly1d(fit)

    ## predict for y = 10000
    y = 10000
    root = (fit_fn - y).roots
    print("Assuming degree for polynomial fitting {}. Predict time for y=10000: {}".format(degree, root[0]))

    plt.plot(arr[:,0], arr[:,1], 'yo', arr[:,0], fit_fn(arr[:,0]), '--k')
    addLabels()
    plt.savefig('plots/polyfit_degree_' + str(degree) + '.png')
    plt.close()


