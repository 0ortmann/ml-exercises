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

matfile = sio.loadmat('./cancer-data.mat')

cancer_input_train = matfile['cancerInput_train']
cancer_target_train = matfile['cancerTarget_train']

cancer_input_test = matfile['cancerInput_test']
cancer_target_test = matfile['cancerTarget_test']

def svc_predict(cancer_input_train, cancer_input_test, cancer_target_train, cancer_target_test):
    test_losses = []
    train_losses = []
    for c in [0.01, 0.1, 0.5, 1, 5, 10, 50]:
        clf = svm.SVC(C=c)
        clf.fit(cancer_input_train, cancer_target_train)
        y_pred_test = clf.predict(cancer_input_test)
        y_pred_train = clf.predict(cancer_input_train)
        test_losses.append(metrics.zero_one_loss(cancer_target_test, y_pred_test))
        train_losses.append(metrics.zero_one_loss(cancer_target_train, y_pred_train))
    return test_losses, train_losses
    
def plot_losses(test_losses, train_losses):
    plt.figure()
    plt.plot([0.01, 0.1, 0.5, 1, 5, 10, 50], test_losses)
    plt.plot([0.01, 0.1, 0.5, 1, 5, 10, 50], train_losses)
    plt.xticks([0.01, 0.1, 0.5, 1, 5, 10, 50])    
    plt.xscale('log')
    plt.show()
    plt.close()
    
test_losses, train_losses = svc_predict(cancer_input_train, cancer_input_test, cancer_target_train, cancer_target_test)
plot_losses(test_losses, train_losses)
