# -*- coding: utf-8 -*-
"""
Created on Fri May 25 13:56:54 2018

@author: Felix Ortmann, Ina Reis
"""

from sklearn import svm
from sklearn import metrics
from scipy import io as sio
import numpy as np
import matplotlib.pyplot as plt

'''
6.1: 
configuring the SVM: 
to configure our SVM, we set the respective parameters when calling the class (see code for example)
training the SVM:
to train the SVM, we fit it to our training data and labels (see code for example)
classifying a new test point: 
to classify a new test point, we use the predict function (see code for example)
accessing the support vectors: the different svm class implementations provide certain member functions to
access the support vectors: to access the vectors, use support_vectors_, to access the indices, use support_, to 
get the number of vectors, use n_support
multi-class classification: to configure our SVM to do a multiclass classification, we have to define the shape 
of the decision function in order to incorporate the results of the one against all classification into one 
classification output
'''
matfile = sio.loadmat('./cancer-data.mat')

cancer_input_train = matfile['cancerInput_train']
cancer_target_train = matfile['cancerTarget_train']

cancer_input_test = matfile['cancerInput_test']
cancer_target_test = matfile['cancerTarget_test']

C = [0.01, 0.1, 0.5, 1, 5, 10, 50]

def svc_predict(cancer_input_train, cancer_input_test, cancer_target_train, cancer_target_test):
    test_losses = []
    train_losses = []
    for c in C:
        clf = svm.SVC(C=c) #calling SVC class, configuring the SVM
        clf.fit(cancer_input_train, cancer_target_train) #training the SVM
        y_pred_test = clf.predict(cancer_input_test) #classify the test data
        y_pred_train = clf.predict(cancer_input_train) #classify the training data
        test_losses.append(metrics.zero_one_loss(cancer_target_test, y_pred_test)) #compute 0-1 test loss 
        train_losses.append(metrics.zero_one_loss(cancer_target_train, y_pred_train)) #compute 0-1 train loss 
    return test_losses, train_losses
    
def plot_losses(test_losses, train_losses, name, title):
    plt.figure()
    plt.title(title)
    plt.plot(C, test_losses)
    plt.plot(C, train_losses)
    plt.xticks(C)    
    plt.xscale('log')
    plt.legend(('test_losses', 'train_losses'))
    plt.savefig('plots/' + name + '.png')
    plt.close()
    
test_losses, train_losses = svc_predict(cancer_input_train, cancer_input_test, cancer_target_train, cancer_target_test)
plot_losses(test_losses, train_losses, 'losses', 'losses for different values of C (0.01-50, log scaled)')

'''
What is the effect of choosing a large cost?
The loss is reduced by increasing C, but after a certain threshold (around 0.1) the improvement of the losses 
is marginal. We can see clearly from the plot that the losses for a C of 1 and a C of 50 are almost identical.
We expected the algortihm to overfit the training data, but the test loss shows no indication of this. 
'''