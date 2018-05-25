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
from sklearn.model_selection import cross_val_score


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

#6.2
matfile = sio.loadmat('./cancer-data.mat')

cancer_input_train = matfile['cancerInput_train']
cancer_target_train = matfile['cancerTarget_train']

cancer_input_test = matfile['cancerInput_test']
cancer_target_test = matfile['cancerTarget_test']

C = [0.01, 0.1, 0.5, 1, 5, 10, 50]

#6.2 a) impact of C
def svc_predict(cancer_input_train, cancer_input_test, cancer_target_train, cancer_target_test, kernel, degree=3, gamma='auto', coef0=0.0):
    test_losses = []
    train_losses = []
    scores = []
    for c in C:
        clf = svm.SVC(c, kernel, degree, gamma, coef0) #calling SVC class, configuring the SVM
        clf.fit(cancer_input_train, cancer_target_train) #training the SVM
        y_pred_test = clf.predict(cancer_input_test) #classify the test data
        y_pred_train = clf.predict(cancer_input_train) #classify the training data
        test_losses.append(metrics.zero_one_loss(cancer_target_test, y_pred_test)) #compute 0-1 test loss 
        train_losses.append(metrics.zero_one_loss(cancer_target_train, y_pred_train)) #compute 0-1 train loss 
        score = cross_val_score(clf, cancer_input_train, cancer_target_train, cv=5)
        scores.append(score.mean())
    return test_losses, train_losses, scores
    
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
    
def plot_score(score, name, title):
    plt.figure()
    plt.title(title)
    plt.xticks(C)    
    plt.xscale('log')
    plt.plot(C, score)
    plt.savefig('plots/' + name + '.png')
    plt.close()
    
test_losses, train_losses, score = svc_predict(cancer_input_train, cancer_input_test, cancer_target_train, cancer_target_test, 'rbf')
plot_losses(test_losses, train_losses, 'C_losses', 'losses for different values of C (0.01-50, log scaled)')
plot_score(score, 'C_scores', 'Accuracy for different values of C')
'''
What is the effect of choosing a large cost?
The loss is reduced by increasing C, but after a certain threshold (around 0.1) the improvement of the losses 
is marginal. We can see clearly from the plot that the losses for a C of 1 and a C of 50 are almost identical.
We expected the algortihm to overfit the training data, but the test loss shows no indication of this. 
'''
#6.2 b) different kernels


kernels =  ['linear', 'poly', 'rbf', 'sigmoid']
def predict_kernels(cancer_input_train, cancer_input_test, cancer_target_train, cancer_target_test):
    for k in kernels:
        kernel_test_losses, kernel_train_losses, score = svc_predict(cancer_input_train, cancer_input_test, cancer_target_train, cancer_target_test, k)
        plot_losses(kernel_test_losses, kernel_train_losses, k + '_losses', 'losses for the ' + k + ' kernel function: ')
        plot_score(score, k + '_score', 'Accuracy of kernel ' + k)        
        
predict_kernels(cancer_input_train, cancer_input_test, cancer_target_train, cancer_target_test)

#evaluating different parameters
degrees = np.arange(6)
gammas = np.arange(0.01, 0.1, 0.01)
coef0s = np.arange(0.0, 2.0, 0.2)

def eval_params(kernel):
        for g in gammas:
                test_loss, train_loss, score = svc_predict(cancer_input_train, cancer_input_test, cancer_target_train, cancer_target_test, kernel, 3, g)
                plot_losses(test_loss, train_loss, kernel + '_losses' + 'g' + str(g), 'losses for kernel ' + kernel  + 'g' + str(g) )
                plot_score(score, 'score_g' + str(g), 'score for gamma ' + str(g))
eval_params('rbf')
#eval_params('poly')
#eval_params('sigmoid')   


'''
Evaluation results: 
the kernel that performs best on the data is the rbf kernel, closely followed by the sigmoid and poly (d=1) kernels. 
When picking the gamma parameter, it is evident the results get better the closer we get to the default value 1/n_features
For the poly kernel, the kernel of degree 1 gives the best results with regards to loss and accuracy
'''


#6.2 c) swap train and test sets
test_losses, train_losses, score = svc_predict(cancer_input_test, cancer_input_train, cancer_target_test, cancer_target_train, 'rbf')
plot_losses(test_losses, train_losses, 'swapped_losses', 'losses when train and test sets are swapped')
plot_score(score, 'swapped_scores', 'Accuracy when train and test sets are swapped')

'''
Swapping the test and train sets results in a slightly higher loss and slightly lowered accuracy. Overall the results were better than expected. For sets
with larger size differences, the decrease in accuracy would be more noticeable.
'''