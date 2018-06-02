#!/usr/bin/python3

print('''assignment 7.02
    a) If a single eigenvector covered 0.9 of the data, that would mean that the longest possible, maximized variance describes most of the data very well. The variance in all other dimensions does not describe the data well (0.1). If one would now use the first principle component to reconstruct the original data, one would get out 90 percent, losing 10 percent in other dimensions. 90 percent of all data points are closely situated next to this eigenvector (closer than to any other eigenvector).
    b) If (0,1,1,1) was an eigenvector with 0.85 cover on the data: along the three dimensions of time spent with gaming, facebook and sports is the most information covered (the most spread / variance), but not with age.
    ''')


print('''assignment 7.03
    a) Downloaded script from https://scikit-learn.org/0.15/_downloads/face_recognition.py
       We had to change line 108, class_weight parameter in call to SVC. We set the parameter to "balanced", before it was "auto", but that is no longer an option with current versions of sklearn.
    b) The script calls the built-in function "fetch_lfw_people" from sklearn. It downloads a built-in dataset, called "labeled faces in the wild". Properties (taken from the documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html)
        - pictures centered on single face
        - pixel in rgb
        - pixel value range: 0.0 - 1.0 (encoded as floats)
        - image size 62 x 47 pixels
        - n_samples: 1288
        - n_features: 1850
        - n_classes: 7

       Plot some images for each class (person)... (see the ./plots folder)''')

## plot some images of each of the 7 classes (== plot some photos of each person)

from sklearn.datasets import fetch_lfw_people
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

images = lfw_people.images
n_samples, h, w = lfw_people.images.shape
y = lfw_people.target
target_names = lfw_people.target_names

person_pics = defaultdict(list)
for index, person in enumerate(y):
    person_pics[person] += [index]

for person in person_pics:
    ## plot first 3 images per person
    for i in range(3):
        pic = person_pics[person][i]
        title = target_names[person] + '_' + str(i)
        plt.figure(title)
        plt.imshow(images[pic].reshape((h, w)), cmap=plt.cm.gray)
        plt.savefig('plots/' + title + '.png')
        plt.close()

print('''d) Try different 'n_components'
    We tried values for 'n_components' (nc) from 15 to 300, in steps of 50. Those are the results:
    ============ nc = 15 =====================================

                   precision    recall  f1-score   support

     Ariel Sharon       0.67      0.50      0.57        24
     Colin Powell       0.57      0.73      0.64        51
  Donald Rumsfeld       0.62      0.37      0.46        35
    George W Bush       0.70      0.78      0.74       125
Gerhard Schroeder       0.29      0.24      0.26        25
      Hugo Chavez       0.57      0.67      0.62        12
       Tony Blair       0.56      0.48      0.52        50

      avg / total       0.61      0.61      0.60       322

[[12  6  2  4  0  0  0]
 [ 2 37  1  8  0  0  3]
 [ 1  6 13 12  1  0  2]
 [ 2  9  3 98  7  2  4]
 [ 0  0  2  6  6  2  9]
 [ 0  0  0  1  2  8  1]
 [ 1  7  0 11  5  2 24]]

============ nc = 25 =====================================

                   precision    recall  f1-score   support

     Ariel Sharon       0.61      0.74      0.67        19
     Colin Powell       0.78      0.65      0.71        72
  Donald Rumsfeld       0.59      0.71      0.65        28
    George W Bush       0.77      0.82      0.79       120
Gerhard Schroeder       0.59      0.61      0.60        28
      Hugo Chavez       0.83      0.71      0.77        14
       Tony Blair       0.61      0.54      0.57        41

      avg / total       0.71      0.71      0.71       322

[[14  1  3  1  0  0  0]
 [ 4 47  1 12  2  0  6]
 [ 2  0 20  6  0  0  0]
 [ 3  7  9 98  2  0  1]
 [ 0  0  1  4 17  1  5]
 [ 0  1  0  0  1 10  2]
 [ 0  4  0  7  7  1 22]]

============ nc = 30 =====================================
                   precision    recall  f1-score   support

     Ariel Sharon       0.75      0.82      0.78        22
     Colin Powell       0.76      0.85      0.80        60
  Donald Rumsfeld       0.74      0.68      0.71        25
    George W Bush       0.83      0.90      0.86       116
Gerhard Schroeder       0.64      0.39      0.48        41
      Hugo Chavez       0.62      0.62      0.62        16
       Tony Blair       0.61      0.60      0.60        42

      avg / total       0.74      0.75      0.74       322

[[ 18   3   0   1   0   0   0]
 [  2  51   0   4   2   0   1]
 [  0   2  17   3   2   0   1]
 [  0   4   4 104   1   2   1]
 [  2   3   1   7  16   1  11]
 [  0   0   1   1   2  10   2]
 [  2   4   0   6   2   3  25]]


============ nc = 50 =====================================
                   precision    recall  f1-score   support

     Ariel Sharon       0.73      0.52      0.61        21
     Colin Powell       0.85      0.90      0.87        58
  Donald Rumsfeld       0.76      0.79      0.78        33
    George W Bush       0.92      0.90      0.91       135
Gerhard Schroeder       0.70      0.68      0.69        28
      Hugo Chavez       0.80      0.57      0.67        21
       Tony Blair       0.59      0.85      0.70        26

      avg / total       0.83      0.82      0.82       322

[[ 11   4   0   5   1   0   0]
 [  0  52   1   1   2   0   2]
 [  2   0  26   3   1   0   1]
 [  2   2   5 122   2   1   1]
 [  0   0   1   1  19   1   6]
 [  0   3   0   1   0  12   5]
 [  0   0   1   0   2   1  22]]


============ nc = 100 =====================================

                    precision    recall  f1-score   support

     Ariel Sharon       0.72      0.81      0.76        16
     Colin Powell       0.82      0.80      0.81        69
  Donald Rumsfeld       0.94      0.87      0.90        38
    George W Bush       0.89      0.97      0.93       124
Gerhard Schroeder       0.83      0.86      0.84        22
      Hugo Chavez       0.88      0.82      0.85        17
       Tony Blair       0.96      0.75      0.84        36

      avg / total       0.88      0.87      0.87       322

[[ 13   0   2   0   1   0   0]
 [  4  55   0   8   0   2   0]
 [  0   3  33   1   0   0   1]
 [  1   3   0 120   0   0   0]
 [  0   2   0   1  19   0   0]
 [  0   1   0   1   1  14   0]
 [  0   3   0   4   2   0  27]]


============ nc = 150 ====================================
                   precision    recall  f1-score   support

     Ariel Sharon       1.00      0.60      0.75        20
     Colin Powell       0.83      0.98      0.90        56
  Donald Rumsfeld       0.92      0.92      0.92        25
    George W Bush       0.85      0.95      0.90       127
Gerhard Schroeder       1.00      0.62      0.77        40
      Hugo Chavez       1.00      0.87      0.93        15
       Tony Blair       0.92      0.92      0.92        39

      avg / total       0.90      0.89      0.88       322

[[ 12   3   0   5   0   0   0]
 [  0  55   0   1   0   0   0]
 [  0   0  23   2   0   0   0]
 [  0   4   2 121   0   0   0]
 [  0   3   0   9  25   0   3]
 [  0   0   0   2   0  13   0]
 [  0   1   0   2   0   0  36]]

============ nc = 200 ====================================

                   precision    recall  f1-score   support

     Ariel Sharon       0.75      0.80      0.77        15
     Colin Powell       0.67      0.85      0.75        55
  Donald Rumsfeld       0.81      0.76      0.79        34
    George W Bush       0.83      0.86      0.84       134
Gerhard Schroeder       0.88      0.81      0.85        27
      Hugo Chavez       1.00      0.58      0.73        19
       Tony Blair       0.90      0.68      0.78        38

      avg / total       0.82      0.80      0.80       322

[[ 12   1   0   2   0   0   0]
 [  1  47   0   7   0   0   0]
 [  1   2  26   5   0   0   0]
 [  2  10   4 115   1   0   2]
 [  0   1   0   4  22   0   0]
 [  0   5   0   1   1  11   1]
 [  0   4   2   5   1   0  26]]

============ nc = 250 ====================================

                   precision    recall  f1-score   support

     Ariel Sharon       0.87      0.68      0.76        19
     Colin Powell       0.78      0.82      0.80        61
  Donald Rumsfeld       0.89      0.81      0.85        31
    George W Bush       0.84      0.91      0.88       135
Gerhard Schroeder       0.83      0.68      0.75        22
      Hugo Chavez       0.75      0.56      0.64        16
       Tony Blair       0.87      0.89      0.88        38

      avg / total       0.84      0.84      0.83       322

[[ 13   5   0   1   0   0   0]
 [  0  50   0  11   0   0   0]
 [  2   0  25   3   1   0   0]
 [  0   6   2 123   0   2   2]
 [  0   0   0   4  15   1   2]
 [  0   2   0   3   1   9   1]
 [  0   1   1   1   1   0  34]]

============ nc = 300 ====================================

                   precision    recall  f1-score   support

     Ariel Sharon       0.83      0.77      0.80        13
     Colin Powell       0.79      0.94      0.86        52
  Donald Rumsfeld       0.81      0.75      0.78        28
    George W Bush       0.86      0.91      0.88       150
Gerhard Schroeder       0.87      0.54      0.67        24
      Hugo Chavez       0.85      0.69      0.76        16
       Tony Blair       0.91      0.79      0.85        39

      avg / total       0.85      0.84      0.84       322

[[ 10   2   0   1   0   0   0]
 [  0  49   0   3   0   0   0]
 [  0   1  21   6   0   0   0]
 [  0   7   4 137   0   1   1]
 [  0   2   1   6  13   1   1]
 [  1   0   0   2   1  11   1]
 [  1   1   0   5   1   0  31]]

    The results are ok-ish for nc >= 30. For 25 or even 15 the precision went far too low to make adequate predictions. The best average precision is found when nc is approx. 150. With lower or higher number of components, the precision sinks. This can also be seen in the confusion matrices.
    ''')


print('''e) Eigenface algorithm in automatic passport control?
    I don't think it is a good idea to use the (raw) eigenface algorithm in passport control. The idea of the algorithm is to use PCA to decompose face images into a smaller set of images with very charactereristic features. The recognition is done by projecting the test-face to all those eigenfaces and check the distance to each of them.
    This algorithm may be very useful given a large training data set. But in the passport there is only one foto and one face to test against (the real person). There is not enough data in the passport fotograph to successfully train eg. a SVM.
    In general, the eigenface algorithm is very robust against illumination or face orientation. But it still requires training data, which is not given for the passports with single fotos that we have nowadays.
    ''')