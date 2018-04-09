#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np 

## basic plot labels
plt.title('Traffic per Hour')
plt.ylabel('Traffic')
plt.xlabel('Hours')

## read file into numpy
file = './traffic_per_hour.csv'
arr = np.fromfile(file, dtype='float', sep='\t').reshape(-1, 2)

## remove nan
arr = arr[~np.isnan(arr).any(axis=1)]

## ex 1: scatter to plot
# plt.scatter(arr[:,0], arr[:,1])
# plt.savefig('plots/traffic_per_hour.png')


## fit linear
fit = np.polyfit(arr[:,0], arr[:,1], deg=1)
fit_fn = np.poly1d(fit)

#plt.plot(arr[:,0], arr[:,1], 'yo', arr[:,0], fit_fn(arr[:,0]), '--k')
#plt.savefig('plots/linear_polyfit.png')


