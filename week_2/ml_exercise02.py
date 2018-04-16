# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 12:25:39 2018

@author: Felix Ortmann (0ortmann) Ina Reis (0reis)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#exercise 2.1 - numpy operations
a = np.arange(10)
    
b = np.full((3,3), True, dtype=bool)

c = a[a%2>0]

d = np.where(a % 2 == 0, -1, a)

e = np.zeros((3,3), int)

np.fill_diagonal(e, 1)


#exercise 2.3 - housing data
housing_data = pd.read_csv("./housing.csv", sep=',')



#printing min and max values with indices
def base_statistics(dataframe):
    for c in dataframe:
        if c == 'ocean_proximity':
            continue
        
        print 'max:', c, dataframe[c].idxmax()
        print 'min:', c, dataframe[c].idxmin()
    #printing max values without indices
    print(dataframe.max())
            
    #printing min values without indices        
    print(dataframe.min())

    #printing mean values
    print 'mean:', dataframe.mean()

#call for full dataset
print '\nfull statistics'
base_statistics(housing_data)
#plotting histograms
'''
for c in housing_data:
    if c == 'ocean_proximity':
        continue
    
    plt.figure()
    plt.title(c)
    housing_data[c].plot(kind='hist')
    plt.savefig('plots/hist_' + c + '.png')
    plt.close()
'''    
'''
the median values income and house value are approximately normally distributed
'''
# scatter map
plt.figure()
plt.title('scatter plot')
housing_data.plot.scatter(x='longitude', y='latitude', c='br', alpha=0.5)
plt.savefig('plots/scatter.png')
plt.close()

#splitting training and test set
#seeding random generator deterministically
np.random.seed(5)

mask = np.random.rand(len(housing_data)) < 0.8

train = housing_data[mask]
test = housing_data[~mask]

#call for training set
print '\n train statistics:'
base_statistics(train)

#call for test set
print '\n test statistics'
base_statistics(test)



