#!/bin/python3

import numpy as np
import pandas as pd
from scipy import io as sio
import matplotlib.pyplot as plt
from numpy.linalg import inv

##### assignment 1

## a) read dataset + plot basics
vaccination_data = pd.read_csv("./vaccination.csv", sep=',')

## helper to plot stuff
def plot_column(col, range_to, file_name):
    counts = vaccination_data[col].value_counts()
    frame = pd.DataFrame({})
    for i, col_props in enumerate(range_to):
        frame[col_props[1]] = [counts[col_props[0]]]
    ax = frame.plot.bar(title=col)
    ax.set_xticks([])
    for p in ax.patches:
        ax.text(p.get_x(), p.get_height()+1, str(p.get_height()))
    plt.savefig(file_name)

plot_column('gender', [(0, 'female'), (1, 'male')], 'plots/gender_counts.png')
plot_column('age', [(1, '0-2 years'), (2, '3-6 years'), (3, '7-10 years'), (4, '11-13 years'), (5, '14-17 years')], 'plots/age_groups.png')
plot_column('olderSiblings', [(0, 'no siblings'), (1, '1 sibling'), (2, '2 siblings'), (3, '3 siblings'), (4, '4 siblings')], 'plots/older_siblings.png')



x = vaccination_data['diseaseX']
y = vaccination_data['diseaseY']
z = vaccination_data['diseaseZ']
age = vaccination_data['age']
weight = vaccination_data['weight']
height = vaccination_data['height']
vacX = vaccination_data['vacX']
residence = vaccination_data['residence']
olderSiblings = vaccination_data['olderSiblings']
bike = vaccination_data['knowsToRideABike']


frame_length = len(y)


## b) marginal probs
print('marginal probabilities')
prob_vac_x = (vacX == 1).sum() / frame_length
print('Probability to have a vaccination against X:', prob_vac_x)

prob_country_side = (residence == 1).sum() / frame_length
print('Probability to live on the country side:', prob_country_side)

prob_older_sib = (olderSiblings != 0).sum() / frame_length
print('Probability to have at least one older sibling:', prob_older_sib)

## c) more marginal probs

prob_taller_1m = (height > 100).sum() / frame_length
print('Probability to be taller than 1m:', prob_taller_1m)

prob_weight_40kg = (weight > 40).sum() / frame_length
print('Probability to weigh more than 40kg:', prob_weight_40kg)

prob_had_Y_Z = ((y == 1) | (z == 1)).sum() / frame_length
print('Probability to have had diseaseY or diseaseZ:', prob_had_Y_Z)
print('')
## d) relative probabilities

print('relative probabilities')
print('')
prob_had_X = (x == 1).sum() / frame_length
prob_had_X_not_vac = (x[ vacX==0 ] == 1).sum() / frame_length
prob_had_X_with_vac = (x[ vacX==1 ] == 1).sum() / frame_length
print('P(diseaseX)', prob_had_X)
print('P(diseaseX | vacX = 0)', prob_had_X_not_vac)
print('P(diseaseX | vacX = 1)', prob_had_X_with_vac)
print('')

prob_vac_X = (vacX == 1).sum() / frame_length
prob_vac_X_had_X = (vacX[ x==1 ] == 1).sum() / frame_length
prob_vac_X_had_not_X = (vacX[ x==0 ] == 1).sum() / frame_length
print('P(vacX)', prob_vac_X)
print('P(vacX | diseaseX = 0)', prob_vac_X_had_not_X)
print('P(vacX | diseaseX = 1)', prob_vac_X_had_X)
print('')


prob_had_Y = (y == 1).sum() / frame_length
prob_had_Y_age_1 = (y[ age==1 ] == 1).sum() / frame_length
prob_had_Y_age_2 = (y[ age==2 ] == 1).sum() / frame_length
prob_had_Y_age_3 = (y[ age==3 ] == 1).sum() / frame_length
prob_had_Y_age_4 = (y[ age==4 ] == 1).sum() / frame_length
prob_had_Y_age_5 = (y[ age==5 ] == 1).sum() / frame_length
print('P(diseaseY)', prob_had_Y)
print('P(diseaseY | age = 1)', prob_had_Y_age_1)
print('P(diseaseY | age = 2)', prob_had_Y_age_2)
print('P(diseaseY | age = 3)', prob_had_Y_age_3)
print('P(diseaseY | age = 4)', prob_had_Y_age_4)
print('P(diseaseY | age = 5)', prob_had_Y_age_5)

plt.figure('P(diseaseY | age = 1/2/3/4)')
probs_y = [prob_had_Y_age_1, prob_had_Y_age_2, prob_had_Y_age_3, prob_had_Y_age_4]
plt.plot([1, 2, 3, 4], probs_y)
plt.savefig('./plots/conditional_prob_diseaseY_age_1234.png')
print('')

prob_vacX_age_1 = (vacX[ age==1 ] == 1).sum() / frame_length
prob_vacX_age_2 = (vacX[ age==2 ] == 1).sum() / frame_length
prob_vacX_age_3 = (vacX[ age==3 ] == 1).sum() / frame_length
prob_vacX_age_4 = (vacX[ age==4 ] == 1).sum() / frame_length
prob_vacX_age_5 = (vacX[ age==5 ] == 1).sum() / frame_length
print('P(vacX)', prob_vac_x)
print('P(vacX | age = 1)', prob_vacX_age_1)
print('P(vacX | age = 2)', prob_vacX_age_2)
print('P(vacX | age = 3):', prob_vacX_age_3)
print('P(vacX | age = 4):', prob_vacX_age_4)
print('P(vacX | age = 5):', prob_vacX_age_5)

plt.figure('P(vacX | age = 1/2/3/4)')
probs_vac_x = [prob_vacX_age_1, prob_vacX_age_2, prob_vacX_age_3, prob_vacX_age_4]
plt.plot([1, 2, 3, 4], probs_vac_x)
plt.savefig('./plots/conditional_prob_vacX_age_1234.png')
print('')

prob_bike = (bike == 1).sum() / frame_length
prob_bike_not_vac = (bike[ vacX==0 ] == 1).sum() / frame_length
prob_bike_vac = (bike[ vacX==1 ] == 1).sum() / frame_length
print('P(knowsToRideABike)', prob_bike)
print('P(knowsToRideABike | vacX = 1)', prob_bike_vac)
print('P(knowsToRideABike | vacX = 0)', prob_bike_not_vac)


print('we can conclude from the plots: it is more likely to have had disease Y or to be vaccinated against X, for older children. With rising age, the probability to have been infected rises and the probability to be vaccinated rises.')
print('')

## e) more conditional probabilites

print('Compare P(diseaseYZ | vacX = 0/1) with P(diseaseX | vacX = 0/1)')
prob_y_z_vac_x = ((y[ vacX ==1 ] == 1) | (z[ vacX ==1 ] == 1)).sum() / frame_length
prob_y_z_not_vac_x = ((y[ vacX ==0 ] == 1) | (z[ vacX ==0 ] == 1)).sum() / frame_length
print('P(diseaseYZ | vacX = 1)', prob_y_z_vac_x)
print('P(diseaseYZ | vacX = 0)', prob_y_z_not_vac_x)
print('Conclusion: it is much more likely to get infected with Y or Z when a child is vaccinated against X. Compared to P(diseaseX | vacX = 0/1), it appears to be not a good idea to vaccinate against X. It brings more YZ infections, while it does not lower X infections very much.')
print('')

print('Calculate P(diseaseYZ | vacX = 0/1, age = 1/2/3/4)')
prob_y_z_vac_x_age_1 = ((y[ (vacX ==1) & (age==1) ] == 1) | (z[ (vacX ==1) & (age==1) ] == 1)).sum() / frame_length
prob_y_z_vac_x_age_2 = ((y[ (vacX ==1) & (age==2) ] == 1) | (z[ (vacX ==1) & (age==2) ] == 1)).sum() / frame_length
prob_y_z_vac_x_age_3 = ((y[ (vacX ==1) & (age==3) ] == 1) | (z[ (vacX ==1) & (age==3) ] == 1)).sum() / frame_length
prob_y_z_vac_x_age_4 = ((y[ (vacX ==1) & (age==4) ] == 1) | (z[ (vacX ==1) & (age==4) ] == 1)).sum() / frame_length
prob_y_z_vac_x_age_5 = ((y[ (vacX ==1) & (age==5) ] == 1) | (z[ (vacX ==1) & (age==5) ] == 1)).sum() / frame_length
prob_y_z_not_vac_x_age_1 = ((y[ (vacX ==0) & (age==1) ] == 1) | (z[ (vacX ==0) & (age==1) ] == 1)).sum() / frame_length
prob_y_z_not_vac_x_age_2 = ((y[ (vacX ==0) & (age==2) ] == 1) | (z[ (vacX ==0) & (age==2) ] == 1)).sum() / frame_length
prob_y_z_not_vac_x_age_3 = ((y[ (vacX ==0) & (age==3) ] == 1) | (z[ (vacX ==0) & (age==3) ] == 1)).sum() / frame_length
prob_y_z_not_vac_x_age_4 = ((y[ (vacX ==0) & (age==4) ] == 1) | (z[ (vacX ==0) & (age==4) ] == 1)).sum() / frame_length
prob_y_z_not_vac_x_age_5 = ((y[ (vacX ==0) & (age==5) ] == 1) | (z[ (vacX ==0) & (age==5) ] == 1)).sum() / frame_length


print('P(diseaseYZ | vacX = 1, age = 1)', prob_y_z_vac_x_age_1)
print('P(diseaseYZ | vacX = 1, age = 2)', prob_y_z_vac_x_age_2)
print('P(diseaseYZ | vacX = 1, age = 3)', prob_y_z_vac_x_age_3)
print('P(diseaseYZ | vacX = 1, age = 4)', prob_y_z_vac_x_age_4)
#print('P(diseaseYZ | vacX = 1, age = 5)', prob_y_z_vac_x_age_5)
print('P(diseaseYZ | vacX = 0, age = 1)', prob_y_z_not_vac_x_age_1)
print('P(diseaseYZ | vacX = 0, age = 2)', prob_y_z_not_vac_x_age_2)
print('P(diseaseYZ | vacX = 0, age = 3)', prob_y_z_not_vac_x_age_3)
print('P(diseaseYZ | vacX = 0, age = 4)', prob_y_z_not_vac_x_age_4)
#print('P(diseaseYZ | vacX = 0, age = 5)', prob_y_z_not_vac_x_age_5)
print('As we calculate the probabilities in a deterministic manner, those results are very accurate.')
print('')

plt.figure('P(diseaseYZ | vacX = 0/1, age = 1/2/3/4)')
plt.plot([1,2,3,4], [prob_y_z_vac_x_age_1,prob_y_z_vac_x_age_2,prob_y_z_vac_x_age_3,prob_y_z_vac_x_age_4])
plt.plot([1,2,3,4], [prob_y_z_not_vac_x_age_1,prob_y_z_not_vac_x_age_2,prob_y_z_not_vac_x_age_3,prob_y_z_not_vac_x_age_4])
plt.savefig('./plots/conditional_prob_diseaseYZ_vacX_age_1234.png')

print('The plot shows a correlation between vaccination against X and the probability to suffer from desease Y or Z. Not being vaccinated seems to be healthier.')



##### assignment 2
print('assignment 2')

## a) data loading and preprocessing
matfile = sio.loadmat('./reg1d.mat')
print('Keys in the matlab file', matfile.keys())

X_train = matfile['X_train']
Y_train = matfile['Y_train']
X_test = matfile['X_test']
Y_test = matfile['Y_test']

plt.figure('Raw training data')
plt.scatter(X_train, Y_train)
plt.savefig('./plots/raw_training_data')

plt.figure('Raw testing data')
plt.scatter(X_test, Y_test)
plt.savefig('./plots/raw_testing_data')

X_train = np.insert(X_train, 1, 1, axis=1)
X_test = np.insert(X_test, 1, 1, axis=1)

## b) implement least_squares and apply to training data

def least_squares(X, Y):
    X_t = X.transpose()
    left = inv(np.dot(X_t, X))
    right = np.dot(X_t, Y)
    return np.dot(left, right)

w = least_squares(X_train, Y_train)
print('Calculate solution vector `w` by using own function `least_squares`. `w` = ', w)

## c) squared loss

def lossL2(point, point_pred):
    return (point-point_pred)**2

def average_loss(Y, Y_pred):
    total_loss = 0
    for entry in zip(Y, Y_pred):
        total_loss = total_loss + lossL2(entry[0], entry[1])
    return total_loss / len(Y)


Y_pred = np.dot(X_test, w)
avg_loss = average_loss(Y_test, Y_pred)

print('The average lossL2 is:', avg_loss)
print()

## d) more dimensions in the data

print('Add quadratic dimension. Fixed value for all points: 0.1')

X_quadratic_train = np.insert(X_train, 1, 0.1, axis=1)
X_quadratic_test = np.insert(X_test, 1, 0.1, axis=1)

w_quadratic = least_squares(X_quadratic_train, Y_train) # learn
Y_quadratic_pred = np.dot(X_quadratic_test, w_quadratic) # predict

avg_quadratic_loss = average_loss(Y_test, Y_quadratic_pred)
print('Quadratic dimension added. Average loss is:', avg_quadratic_loss)
print()


print('Add cubic dimension. Fixed value for all points: 0.85')
X_cubic_train = np.insert(X_quadratic_train, 2, 0.85, axis=1)
X_cubic_test = np.insert(X_quadratic_test, 2, 0.85, axis=1)
w_cubic = least_squares(X_cubic_train, Y_train) # learn
Y_cubic_pred = np.dot(X_cubic_test, w_cubic) # predict

avg_cubic_loss = average_loss(Y_test, Y_cubic_pred)
print('Cubic dimension added. Average loss is:', avg_cubic_loss)
print()


## e) extreme outliers, impact on least_squares

print('Add extreme outliers')
X_train = np.concatenate((X_train, [[1.05, 1]]))
Y_train = np.append(Y_train, -10)
w = least_squares(X_train, Y_train)
Y_pred = np.dot(X_test, w)
avg_loss = average_loss(Y_test, Y_pred)

print('The average lossL2 with extreme outliers is:', avg_loss)
print()
