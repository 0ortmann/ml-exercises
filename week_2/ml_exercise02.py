# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 12:25:39 2018

@author: Felix Ortmann (0ortmann) Ina Reis (0reis)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

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


#hilbert matrix
#create hilbert matrix
def hilbert(n):
    v = np.arange(1, n + 1) + np.arange(0, n)[:, np.newaxis]
    return 1. / v
    
#print matrices for n 1-30    
#for n in range(1, 31):
    #print "HilbertMatrix({}) == \n".format(n) , hilbert(n)
    
#solve linear equation
for n in [1, 2, 3, 5, 10, 15, 20, 30, 50, 100]:
    b = np.ones(n)
    h = hilbert(n)
    x = np.linalg.solve(h, b)
    print "hilbert", h
    print "Solution of linear equation of HilbertMatrix({})*(1..1) ==\n {}".format(n,  x)
    print "Verify solution"
    print "{}".format(np.dot(h,x) - b)

#we don't trust the solution! is has marginal differences in the space 10^-8

'''
What is special about this matrices?

    - The values in the hilbert matrix are directly dependent of the indices of the matrix itself (row, column).
    - We are using a quadratic hilbert matrix, thus the matrix is symmetrical along the diagonal.
      The top right value in a hilber matrix of size k would always by 1/k, like the bottom left value.
      The top left value is always 1 and the bottom right value is always 1/(2k-1). 
      A hilbert matrix, only considering one side of the diagonal thus follows the almost exact half of 
      the harmonic series. 
    - values in the verification matrix-dot-product are the exact computations 
      of the hilbertmatrix * solution-vector - target vector. Thus any differences to 0 can 
      be considered as error of the numpy library.
      
'''
#loss functions
def l_1(y_true, y_pred):
    return int(y_true == y_pred)
# l2_loss. This loss does not appear to work well for values > 1   
def l_2(y_true, y_pred):
    return (y_true-y_pred)**2
    
def loss_absolute(y_true, y_pred):
    return abs(y_true-y_pred)
    
#distance functions
def manhattan_distance(x, y):
    return abs(x['latitude']-y['latitude']) + abs(x['longitude']-y['longitude'])

def euclid_distance(x,y):
    delta_lat = abs(x['latitude']-y['latitude'])
    delta_lon = abs(x['longitude']-y['longitude'])
    return math.sqrt(delta_lat**2 + delta_lon**2)

#the kNN Algorithm
# returns the predicted_housing_value and the l2_loss with which the 
# predicted value differs from the exact value
def kNN(dataset, point, k):    
    k_nearest = getKNearest(dataset, point, k)
    predicted_housing_value = sum([x['median_house_value'] for x in k_nearest]) / k
    # uncomment the next line to use square l2_loss    
    # loss = l_2(point['median_house_value'], predicted_housing_value)
    
    # we decided to use the absolute difference between the housing values, because that metric seems to work better
    loss = loss_absolute(point['median_house_value'], predicted_housing_value)
    return predicted_housing_value, loss

# returns the k nearest neighbors of point k in dataset
def getKNearest(dataset, point, k, distance_function=euclid_distance):
    distance_point_map = dict()
    for index, row in dataset.iterrows():
        dist = distance_function(row, point)
        distance_point_map[dist] = row
    indices = sorted(distance_point_map.keys())[:k]
    return [distance_point_map[i]  for i in indices]

for k in [3,5,7,20]:
    print "calculating kNN for k", k
    # note: the following takes some time
    for index, point in test.iterrows():
        knn = kNN(train, point, k)
        print "Checking point: {},{}  : predicted value {}, l2_loss {}".format(point['latitude'], point['longitude'], knn[0], knn[1])

'''
What are your training and test errors?

    The errors (when considering absolute values) are quite low.
    The usual housing values are around 150k-400k. 
    The absolute errors are around 10-150k for k = 3.
    That is quite a huge error, but it seems reasonable because k is low.
    For higher k the absolute error gets better.
    
    Only when using the l2_loss function (which squares the values), the loss appears to be unrealistically high.
'''