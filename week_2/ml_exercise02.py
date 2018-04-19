# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 12:25:39 2018

@author: Felix Ortmann (0ortmann), Ina Reis (0reis)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import math

'''
README:

'''

#2.1 - numpy operations
a = np.arange(10)
#print a    
b = np.full((3,3), True, dtype=bool)
#print b
c = a[a%2>0] 
#print c
d = np.where(a % 2 == 0, -1, a)
#print d
e = np.zeros((3,3), int)
# print e
np.fill_diagonal(e, 1)
#print e

#2.2 - linear Algebra, Hilbert Matrix

#a) function to create hilbert matrix of dimension nxn
def hilbert(n):
    v = np.arange(1, n + 1) + np.arange(0, n)[:, np.newaxis]
    return 1. / v
    
#b) uncomment to print matrices for n = 1-30    
'''
for n in range(1, 31):
    print "HilbertMatrix({}) == \n".format(n) , hilbert(n)
'''
    
#c) solve linear equation
for n in [1, 2, 3, 5, 10, 15, 20, 30, 50, 100]:
    b = np.ones(n)
    h = hilbert(n)
    x = np.linalg.solve(h, b)
    #uncomment to print solutions    
    print "hilbert", h
    print "Solution of linear equation of HilbertMatrix({})*(1..1) ==\n {}".format(n,  x)
    print "Verify solution"
    print "{}".format(np.dot(h,x) - b)

#d) we don't trust the solution! is has marginal differences in the space 10^-8

'''
e) What is special about these matrices?

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


#2.3 - housing data

#a) load dataset
housing_data = pd.read_csv("./housing.csv", sep=',')



#b) printing min and max values with indices
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

#c) plot histograms for all dataset columns except the last (nominal value)
#plots are saved as images
def create_hists(dataset, name):
    for c in dataset:
        if c == 'ocean_proximity':
           continue
    
        plt.figure()
        plt.title(c)
        housing_data[c].plot(kind='hist')
        plt.savefig('plots/hist_' + name + '_' + c + '.png')
        plt.close()
        
create_hists(housing_data, 'full')         

#the median values income and house value are approximately normally distributed

#d) scatter plot
#plot geographical map (plot is saved as image)

colormap = cm.get_cmap('cool')
plt.figure()
plt.title('scatter plot')
housing_data.plot.scatter(x='longitude', y='latitude', c='median_house_value', cmap=colormap, alpha=0.7)
plt.savefig('plots/scatter.png')
plt.close()


#e) splitting training and test set

#seeding random generator deterministically
np.random.seed(5)

#determine ratio of training to test set
#default (as instructed in the assignment) is 0.8
#to run in reasonable time, set 0.999 as a value
mask = np.random.rand(len(housing_data)) < 0.8

train = housing_data[mask]
test = housing_data[~mask]

#calculate histograms and base statistics for training and test set
#print base statistics for training and test sets

print '\n train statistics:'
base_statistics(train)
create_hists(train, "train")

#call for test set
print '\n test statistics'
base_statistics(test)
create_hists(test, "test")

#the test set matches the distribution in the training set.

#2.4 kNN

#a) loss functions

#l1_loss returns 1 if equal, 0 if not
def l_1(y_true, y_pred):
    return int(y_true == y_pred)
    
# l2_loss returns the squared error 
def l_2(y_true, y_pred):
    return (y_true-y_pred)**2

#loss_absolute returns the absolute error 
def loss_absolute(y_true, y_pred):
    return abs(y_true-y_pred)
    
#b) distance functions

#manhattan distance
def manhattan_distance(x, y):
    return abs(x['latitude']-y['latitude']) + abs(x['longitude']-y['longitude'])

#euclidean distance on geographical values
def euclid_distance(x,y):
    delta_lat = abs(x['latitude']-y['latitude'])
    delta_lon = abs(x['longitude']-y['longitude'])
    return math.sqrt(delta_lat**2 + delta_lon**2)
    
#euclidean distance on all features except the last (nominal value)
def euclid_allfeat(x, y):
    x = x.as_matrix()[:-1]
    y = y.as_matrix()[:-1]
    return np.sqrt(np.sum((x-y)**2))

#c) the kNN Algorithm
# returns the predicted_housing_value and the loss with which the 
# predicted value differs from the exact value

def kNN(dataset, point, k, distance_function, loss_function):    
    k_nearest = getKNearest(dataset, point, k, distance_function)
    #print "k_nearest:", k_nearest
    predicted_housing_value = sum([x['median_house_value'] for x in k_nearest]) / k
    
    loss = loss_function(point['median_house_value'], predicted_housing_value)
    
    return predicted_housing_value, loss


# returns the k nearest neighbors of a point in the dataset
def getKNearest(dataset, point, k, distance_function):
    distance_point_map = dict()
    for index, row in dataset.iterrows():
        dist = distance_function(row, point)
        distance_point_map[dist] = row
    indices = sorted(distance_point_map.keys())[:k]
    return [distance_point_map[i]  for i in indices]

#d) predict the housing values for different values of k 
#different loss and distance functions can be used 
k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17]

def calc_kNN(test_k, trainingset, testset, distance_function=euclid_allfeat, loss_function=l_2):
    errors = []
    for k in test_k:
        print "calculating kNN for k =", k
        m_loss = 0
        # note: the following takes some time
        for index, point in testset.iterrows():
            prediction, loss = kNN(trainingset, point, k, distance_function, loss_function)
            #print each prediction and loss 
            print "Checking point: {} : predicted value {}, l2_loss {}".format(index, prediction, loss)
            m_loss = m_loss + loss
        mean_loss = m_loss /len(test)
        errors.append(mean_loss)
        print "mean loss for k = {}: {}".format(k, mean_loss)
    return errors
    

#plot the mean squared errors for different values of k (plot saved as image)
def plot_error(test_k, errors, name):
    plt.figure()
    plt.title('mean squared error')
    plt.plot(test_k, errors, 'ro')
    plt.axis([test_k[0]-1, test_k[-1]+1, 0, max(errors)+0.1*max(errors)])
    plt.xticks(test_k)
    plt.savefig('plots/errors' + name + '.png')
    plt.close()


#predict on training set, calculate training error (uncomment to run)
#run at your own risk, might take a LONG time 
#errs_train = calc_kNN(k_values, train, train)
#plot_error(k_values, errs_train, "_euclideandist_l2loss")

#predict on test set, calculate test error
#we ran this because it finishes in reasonable time to find out about which k to use
errs_test = calc_kNN(k_values, train, test)
plot_error(k_values, errs_test, "_euclideandist_l2loss")


'''
What are your training and test errors?

Since the kNN Algorithm has no classical training phase, the training error in this case would be the error
received when the kNN algorithm is applied to the training set. This means the data point we are classifying will be
considered its own closest neighbour. 
The testing error is the error we receive when the kNN Algorithm classifies data points from a separate test set.
We can use the training error to find a suitable value for k before applying kNN to the test set. 

The squared error is very high as an absolute number, since the predicted values also are high. If considered in relation
to the overall scope of the predicted values, the mean squared error allows us to draw meaningful conclusions, for example
to choose an adequate value for k (with the lowest squared error)

Since our computational power is limited, we used the test set (for its smaller sample size) to get some 
information about the different mean squared errors for values of k.

k = 17 produces the lowest mean squared error when we predict on values from the test set
'''

