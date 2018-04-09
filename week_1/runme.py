#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np 

file = './traffic_per_hour.csv'
arr = np.fromfile(file, dtype='float', sep='\t').reshape(-1, 2)
arr = arr[~np.isnan(arr).any(axis=1)]
print(arr)

plt.scatter(arr[:,0], arr[:,1])
plt.title('Traffic per Hour')
plt.ylabel('Traffic')
plt.xlabel('Hours')
plt.show()