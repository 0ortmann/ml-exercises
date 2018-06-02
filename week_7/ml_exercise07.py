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


print('''e) Eigenface algorithm in automatic passport control?
    I don't think it is a good idea to use the (raw) eigenface algorithm in passport control. The idea of the algorithm is to use PCA to decompose face images into a smaller set of images with very charactereristic features. The recognition is done by projecting the test-face to all those eigenfaces and check the distance to each of them.
    This algorithm may be very useful given a large training data set. But in the passport there is only one foto and one face to test against (the real person). There is not enough data in the passport fotograph to successfully train eg. a SVM.
    In general, the eigenface algorithm is very robust against illumination or face orientation. But it still requires training data, which is not given for the passports with single fotos that we have nowadays.
    ''')