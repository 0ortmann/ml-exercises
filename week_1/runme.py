#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np 

file = './traffic_per_hour.csv'
arr = np.array(np.fromfile(file, dtype='float', sep='\t').reshape(-1, 2))
arr = arr[~np.isnan(arr).any(axis=1)]
print(arr)

#plt.plot([1,2,3,4])
#plt.ylabel('some numbers')
#plt.show()