#!/bin/python3

import numpy as np
import matplotlib.pyplot as plt

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
plt.figure(figsize=(7,6))
plt.plot(a, b, linestyle='-', linewidth=2, label='Circle')
plt.plot(x, y, marker='o', linestyle='None') ## samples
plt.ylim([-r-.5, r+.5])
plt.xlim([-r-.5, r+.5])
plt.grid()
plt.legend(loc='upper right')
plt.savefig('plots/random_values_in_circle.png', block=True)
