#!/bin/python3

import numpy as np
import scipy as sp
from scipy import io as sio
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.metrics import accuracy_score

## assignment 03.1

### random numbers with normal distribution, count n
ranges = [100, 1000, 10000, 100000]
print('Calculate & plot random distributions for different array lengths')
np.random.seed(5)
for n in ranges:
    rand_arr = np.random.randn(n)
    title = 'random_normal_distribution_'+str(n)
    mean, min, max = np.mean(rand_arr), np.amin(rand_arr), np.amax(rand_arr)
    print('n = {} -- mean: {}, min: {}, max: {}'.format(n, mean, min, max))
    plt.figure()
    plt.title(title)
    plt.hist(rand_arr, bins=10)
    plt.savefig('plots/' + title + '.png')
    plt.close()

### random numbers with predefined mean and variance
mean = 1337
spread = 100
print('Calculate & plot random distributions with predefined mean ({}) and variance ({})'.format(mean, spread))
for n in ranges:
    title = 'gaussian_distribution_'+str(n)
    plt.figure()
    plt.title(title)
    rand_arr = np.random.normal(mean, spread, n)
    mean, min, max = np.mean(rand_arr), np.amin(rand_arr), np.amax(rand_arr)
    print('n = {} -- mean: {}, min: {}, max: {}'.format(n, mean, min, max))
    plt.hist(rand_arr, bins=10)
    plt.savefig('plots/' + title + '.png')
    plt.close()

### random numbers with binomial distribution. trying some different probabilities p
print('Calculate & plot random distributions with binomial distribution')
probs = [.3, .5, .7]

for n in ranges:
    for p in probs:
        title = 'binomial_distribution_'+str(n)+'_'+str(p)
        plt.figure()
        plt.title(title)
        rand_arr = np.random.binomial(n, p, n)
        mean, min, max = np.mean(rand_arr), np.amin(rand_arr), np.amax(rand_arr)
        print('n = {} -- mean: {}, min: {}, max: {}'.format(n, mean, min, max))
        plt.hist(rand_arr, bins=10)
        plt.savefig('plots/' + title + '.png')
        plt.close()


### sum up random numbers
### each sum of bins is evaluated for n == 1000 samples
print('Calculate & plot summed normal distributions')
mean = 1
spread = .25
for m in [2, 3, 5, 10, 20]:
    arr = []
    for n in range(100):
        ## sum m random numbers
        arr += [np.sum([np.random.normal(mean, spread, 1) for _ in range(m)])]
    title = 'normal_sum_distribution_hist_M_'+str(m)
    plt.figure()
    plt.title(title)
    plt.hist(arr, bins=10)
    plt.savefig('plots/' + title + '.png')
    plt.close()
    title = 'normal_sum_distribution_scatter_M_'+str(m)
    plt.figure()
    plt.title(title)
    plt.scatter(np.arange(len(arr)), arr)
    plt.savefig('plots/' + title + '.png')
    plt.close()

r = 1
print('Calculate & plot random numbers in a cirlce with radius r = {}'.format(r))
theta = np.linspace(0, 2*np.pi, 1000)
## a, b is the circle with radius r
a, b = r * np.cos(theta), r * np.sin(theta)
rand_arr = np.random.rand(1000)
x, y = rand_arr * np.cos(theta), rand_arr * np.sin(theta)
plt.figure(figsize=(7,7))
plt.plot(a, b, linestyle='-', linewidth=2, label='Circle')
plt.plot(x, y, marker='o', linestyle='None') ## samples
plt.ylim([-r-.5, r+.5])
plt.xlim([-r-.5, r+.5])
plt.grid()
plt.legend(loc='upper right')
plt.savefig('plots/random_values_in_circle.png', block=True)




##### assignment 03.2
print('Assignment 2')

## a) access keys
## gives us a dictionary with variable names as keys, and loaded matrices as values.
matfile = sio.loadmat('./Adot.mat')
print('Keys in the matlab file', matfile.keys())

## b) linear mapping

theta = np.pi/3
V = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
X = matfile['X']
Y = V.dot(X)
plt.figure()
plt.title('linear mapping')
plt.plot(X[0], X[1], color='#ee8d18')
plt.plot(Y[0], Y[1], color='#008d18')
## reference line
k = np.array([[0,100],[0,0]])
plt.plot(k[0], k[1])
plt.plot(V.dot(k)[0], V.dot(k)[1])
plt.savefig('plots/linear_mapping_X_V.png', block=True)
plt.close()

## What does the mapping do? -> it rotates the matrix.
## the orrange plot is the original matrix, the green plot is the rotated matrix
## see also the two lines in the plot. They illustrate the rotation direction

## b2) transpose V and plot again

V_t = np.transpose(V)
Z = V_t.dot(Y)

plt.figure()
plt.title('linear mapping transposed')
plt.plot(Y[0], Y[1], color='#ee8d18')
plt.plot(Z[0], Z[1], color='#008d18')
## reference line
k = V.dot(k) ## update reference line
plt.plot(k[0], k[1])
plt.plot(V_t.dot(k)[0], V_t.dot(k)[1])
plt.savefig('plots/linear_mapping_transposed_Y_V_t.png', block=True)
plt.close()


## What does the mapping do? -> it rotates the matrix back to its original position (bc. transposition of V)
## the orrange plot is the original matrix, the green plot is the rotated matrix
## see also the two lines in the plot. They illustrate the rotation direction

## c) 

## more linear mappings
D1 = np.array([[2, 0], [0, 2]])
D2 = np.array([[2, 0], [0, 1]])

d1 = D1.dot(X)
d2 = D2.dot(X)

plt.figure()
plt.title('D1/D2 linear mappings')
plt.plot(X[0], X[1], color='#ee8d18')
plt.plot(d1[0], d1[1], color='#008d18')
plt.plot(d2[0], d2[1], color='#008dff')
plt.savefig('plots/linear_mapping_D1_D2.png', block=True)
plt.close()

## what do the transformations do?
## both transformations stretch X on the x-axis to the double length
## D1 additionally stretches X on the y-axis to the double length (D2 does not stretch in the y-axis)

## d)

## even more linear mappings:

A = V_t.dot(D2).dot(V)
a = A.dot(X)
plt.figure()
plt.title('A (V1.D2.V) linear mappings')
plt.plot(X[0], X[1], color='#ee8d18')
plt.plot(a[0], a[1], color='#008d18')
plt.savefig('plots/linear_mapping_A.png', block=True)
plt.close()
# what does the mapping do?
# first it applies V_t, thus it rotates the matrix downwards.
# then (D2) it stretches the matrix to double x-axis length
# finally it rotates the matrix back to its original position (V)
# the result is matrix which strechted and rotated while keeping the x-axis direction


###### assignment 03.2

## a) read dataset and convert entries in matrices to double
test_file = sio.loadmat('./usps/usps_test.mat')
test_data = test_file['test_data'].astype('double')
test_label = test_file['test_label'].astype('double')

train_file = sio.loadmat('./usps/usps_train.mat')
train_data = train_file['train_data'].astype('double')
train_label = train_file['train_label'].astype('double')

## b) plot some samples from the data
for i in range(10):
    plt.figure('sample {} handwritten digits'.format(i*1000))
    plt.imshow(train_data[i*1000].reshape(16,16), cmap='gray')
    plt.savefig('plots/sample_handwritten_data_{}.png'.format(i))

## c) kNN

def kNN(k, t_data, t_label, compare_data):
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    knn.fit(t_data, t_label.ravel())
    return knn.predict(compare_data)

## call kNN for different k, plot errors

for k in [1, 3, 5, 7]:
    predicted_test_label = kNN(k, train_data, train_label, test_data)
    #test_error = accuracy_score(predicted_test_label, test_label)

    predicted_train_label = kNN(k, train_data, train_label, train_data)
    #train_error = accuracy_score(predicted_train_label, train_label)

    plt.figure('kNN k={} training errors'.format(k))
    plt.plot(np.arange(len(predicted_train_label)), predicted_train_label, color='#ee0000')
    plt.plot(np.arange(len(train_label)), train_label, color='#000000')
    plt.savefig('plots/kNN_k={}_train_errors'.format(k))

    plt.figure('kNN k={} test errors'.format(k))
    plt.plot(np.arange(len(predicted_test_label)), predicted_test_label, color='#ee0000')
    plt.plot(np.arange(len(test_label)), test_label, color='#000000')
    plt.savefig('plots/kNN_k={}_test_errors'.format(k))


## d) classify specific digits and compare results

train_2 = train_data[1000:2000]
train_label_2 = train_label[1000:2000]
test_2 = test_data[100:200]
test_label_2 = test_label[100:200]

train_3 = train_data[2000:3000]
train_label_3 = train_label[2000:3000]
test_3 = test_data[200:300]
test_label_3 = test_label[200:300]

train_8 = train_data[7000:8000]
train_label_8 = train_label[7000:8000]
test_8 = test_data[700:800]
test_label_8 = test_label[700:800]

train_2_3 = np.concatenate((train_2,train_3))
train_label_2_3 = np.concatenate((train_label_2, train_label_3))
test_2_3 = np.concatenate((test_2, test_3))
test_label_2_3 = np.concatenate((test_label_2, test_label_3))

train_3_8 = np.concatenate((train_3, train_8))
train_label_3_8 = np.concatenate((train_label_3, train_label_8))
test_3_8 = np.concatenate((test_3, test_8))
test_label_3_8 = np.concatenate((test_label_3, test_label_8))

for k in [1, 3, 5, 7]:
    predicted_test_label_2_3 = kNN(k, train_2_3, train_label_2_3, test_2_3)
    predicted_test_label_3_8 = kNN(k, train_3_8, train_label_3_8, test_3_8)

    predicted_train_label_2_3 = kNN(k, train_2_3, train_label_2_3, train_2_3)
    predicted_train_label_3_8 = kNN(k, train_3_8, train_label_3_8, train_3_8)

    plt.figure('kNN 2 & 3 with k={} training errors'.format(k))
    plt.plot(np.arange(len(predicted_train_label_2_3)), predicted_train_label_2_3, color='#ee0000')
    plt.plot(np.arange(len(train_label_2_3)), train_label_2_3, color='#000000')
    plt.savefig('plots/kNN_k={}_on_2_3_train_errors'.format(k))

    plt.figure('kNN 2 & 3 with k={} test errors'.format(k))
    plt.plot(np.arange(len(predicted_test_label_2_3)), predicted_test_label_2_3, color='#ee0000')
    plt.plot(np.arange(len(test_label_2_3)), test_label_2_3, color='#000000')
    plt.savefig('plots/kNN_k={}_on_2_3_test_errors'.format(k))

    plt.figure('kNN 3 & 8 with k={} training errors'.format(k))
    plt.plot(np.arange(len(predicted_train_label_3_8)), predicted_train_label_3_8, color='#ee0000')
    plt.plot(np.arange(len(train_label_3_8)), train_label_3_8, color='#000000')
    plt.savefig('plots/kNN_k={}_on_3_8_train_errors'.format(k))

    plt.figure('kNN 3 & 8 with k={} test errors'.format(k))
    plt.plot(np.arange(len(predicted_test_label_3_8)), predicted_test_label_3_8, color='#ee0000')
    plt.plot(np.arange(len(test_label_3_8)), test_label_3_8, color='#000000')
    plt.savefig('plots/kNN_k={}_on_3_8_test_errors'.format(k))