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


print("assignment 6.1, sklearn SVM documentation")

print('''
    - configuring the SVM:
        to configure our SVM, we set the respective parameters when calling the class (see code for example)
    - training the SVM:
        to train the SVM, we fit it to our training data and labels (see code for example)
    - classifying a new test point:
        to classify a new test point, we use the predict function (see code for example)
    - accessing the support vectors:
        the different svm class implementations provide certain member functions to access the support vectors:
            - 'support_vectors_': access the vectors directly
            - 'support_': access the indices
            - 'n_support': get the number of vectors
    - multi-class classification:
        to configure our SVM to do a multiclass classification, we have to define the shape of the decision function in order to incorporate the results of the one against all classification into one classification output.
''')

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
        clf.fit(cancer_input_train, cancer_target_train.ravel()) #training the SVM
        y_pred_test = clf.predict(cancer_input_test) #classify the test data
        y_pred_train = clf.predict(cancer_input_train) #classify the training data
        test_losses.append(metrics.zero_one_loss(cancer_target_test, y_pred_test)) #compute 0-1 test loss 
        train_losses.append(metrics.zero_one_loss(cancer_target_train, y_pred_train)) #compute 0-1 train loss 
        score = cross_val_score(clf, cancer_input_train, cancer_target_train.ravel(), cv=5)
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


print('''
Evaluation results: 
The kernel that performs best on the data is the rbf kernel, closely followed by the sigmoid and poly (d=1) kernels.
When picking the gamma parameter, it is evident the results get better the closer we get to the default value 1/n_features
For the poly kernel, the kernel of degree 1 gives the best results with regards to loss and accuracy
''')

#6.2 c) swap train and test sets
test_losses, train_losses, score = svc_predict(cancer_input_test, cancer_input_train, cancer_target_test, cancer_target_train, 'rbf')
plot_losses(test_losses, train_losses, 'swapped_losses', 'losses when train and test sets are swapped')
plot_score(score, 'swapped_scores', 'Accuracy when train and test sets are swapped')

'''
Swapping the test and train sets results in a slightly higher loss and slightly lowered accuracy. Overall the results were better than expected. For sets
with larger size differences, the decrease in accuracy would be more noticeable.
'''


## assignment 6.3
print("assignment 6.3 complexities")

print('''a) complexity of predictions:
    a.1) linear least squares (d dimensions, n training points):
        Training is mainly influenced by matrix inversion, which is in O(d^3) and matrix multiplication, which is in O(d^2 * N). So the complexity depends which variable is bigger in size, N or d. So the complexity is O(max(N, d) * d^2)
        Testing is only a dot product with the weight vector.
    a.2) d-kNN:
        Training is in O(1) (data is already given)
        Testing: In case the distances are not precomputed for k, each iteration has to calculate them for each checked point. That would result in O(n*d*k).
    ''')

print('''b) complexity of predicting with a already fitted SVM:
    It strongly depends on the SVM kernel. We consider linear kernels and non-linear kernels:
    - Linear kernel:
        Once the training is done, the fitted hyperplane is the only source of truth that is needed. The prediction can be performed as a simple dot product between the test vector and the learned weight vector.
        For linear kernels, the prediction time is independent of the number of support vectors. Of course this also depends on how the SVM works internally.
    - non-linear kernels:
        The separating hyperplane can have multiple (even infinite) dimensions, eg with RBF kernel. During training time the support vectors are selected. At test time, the complexity is then linear to the number of support vectors m and linear on the number dimensions -> O(m * d)
    ''')

print('''c) space complexity:
    - Linear least squares: once the weight vector is computed, its size terminates the needed space.
    - kNN: space is in O(n * d), given n training points and d dimensions
    - SVM: space is in O(n * d') (assuming that the kernel transformed d dimensions into d' dimensions in kernel space.)
    ''')

print('''d) specific data sample
    - Calculate linear least squares vector: X is matrix(n, d) = X(10000, 256).
        X' dot X -> n*d^2 ops, produces d,d matrix
        inversion of d,d matrix -> d^3 ops
        X' dot Y -> n*d^2 ops, produces d,d matrix 
        multiplication of 2 d,d matrics -> d^3 ops
        => 1344274432 ops (2*(n*d**2)+2*d**3)
        => final space is just a d-vector
    - kNN: for each 10000 datapoints, check 256 dimensions and get k nearest points to a test point:
        => 25600000 ops
        => n * d space
    - 1000 support vectors:
        => after training: O(m*d) ~ 256000 ops
        => space 1000*256 ~ 2560000''')


# You are right, the prediction time does not depend on the data for a linear SVM. This is because the predictor is ust a dot product between a test vector and the learned weight vector.

