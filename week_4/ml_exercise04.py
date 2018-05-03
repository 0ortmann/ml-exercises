#!/bin/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

## b) marginal probs
print('marginal probabilities')
prob_vac_x = (vaccination_data['vacX'] == 1).sum() / len(vaccination_data['vacX'])
print('Probability to have a vaccination against X:', prob_vac_x)

prob_country_side = (vaccination_data['residence'] == 1).sum() / len(vaccination_data['residence'])
print('Probability to live on the country side:', prob_country_side)

prob_older_sib = (vaccination_data['olderSiblings'] != 0).sum() / len(vaccination_data['olderSiblings'])
print('Probability to have at least one older sibling:', prob_older_sib)

## c) more marginal probs

prob_taller_1m = (vaccination_data['height'] > 100).sum() / len(vaccination_data['height'])
print('Probability to be taller than 1m:', prob_taller_1m)

prob_weight_40kg = (vaccination_data['weight'] > 40).sum() / len(vaccination_data['weight'])
print('Probability to weigh more than 40kg:', prob_weight_40kg)

prob_had_Y_Z = ((vaccination_data['diseaseY'] == 1).sum() + (vaccination_data['diseaseZ'] == 1).sum()) / len(vaccination_data['weight'])
print('Probability to have had diseaseY or diseaseZ:', prob_had_Y_Z)

## d) relative probabilities

print('relative probabilities')

x = vaccination_data['diseaseX']
vacX = vaccination_data['vacX']
prob_had_X = (x == 1).sum() / len(x)
prob_had_X_not_vac = (x[ vacX==0 ] == 1).sum() / len(x)
prob_had_X_with_vac = (x[ vacX==1 ] == 1).sum() / len(x)
print('Probability to have had diseaseX in general:', prob_had_X)
print('Probability to have had diseaseX without vaccination:', prob_had_X_not_vac)
print('Probability to have had diseaseX with vaccination:', prob_had_X_with_vac)

prob_vac_X = (vacX == 1).sum() / len(x)
prob_vac_X_had_X = (vacX[ x==1 ] == 1).sum() / len(vacX)
prob_vac_X_had_not_X = (vacX[ x==0 ] == 1).sum() / len(vacX)
print('Probability to be vaccinated against X:', prob_vac_X)
print('Probability to be vaccinated and had X:', prob_vac_X_had_X)
print('Probability to be vaccinated and not had X:', prob_vac_X_had_not_X)

y = vaccination_data['diseaseY']
age = vaccination_data['age']

prob_had_Y = (y == 1).sum() / len(y)
prob_had_Y_age_1 = (y[ age==1 ] == 1).sum() / len(y)
prob_had_Y_age_2 = (y[ age==2 ] == 1).sum() / len(y)
prob_had_Y_age_3 = (y[ age==3 ] == 1).sum() / len(y)
prob_had_Y_age_4 = (y[ age==4 ] == 1).sum() / len(y)
prob_had_Y_age_5 = (y[ age==5 ] == 1).sum() / len(y)
print('Probability to have had Y:', prob_had_Y)
print('Probability to have had Y from years 0-2:', prob_had_Y_age_1)
print('Probability to have had Y from years 3-6:', prob_had_Y_age_2)
print('Probability to have had Y from years 7-10:', prob_had_Y_age_3)
print('Probability to have had Y from years 11-13:', prob_had_Y_age_4)
print('Probability to have had Y from years 14-17:', prob_had_Y_age_5)

plt.figure('Disease Y | age {1 2 3 4}')
probs_y = [prob_had_Y_age_1, prob_had_Y_age_2, prob_had_Y_age_3, prob_had_Y_age_4]
plt.plot([1, 2, 3, 4], probs_y)
plt.savefig('./plots/conditional_prob_diseaseY_age_1234.png')

prob_vacX_age_1 = (vacX[ age==1 ] == 1).sum() / len(vacX)
prob_vacX_age_2 = (vacX[ age==2 ] == 1).sum() / len(vacX)
prob_vacX_age_3 = (vacX[ age==3 ] == 1).sum() / len(vacX)
prob_vacX_age_4 = (vacX[ age==4 ] == 1).sum() / len(vacX)
prob_vacX_age_5 = (vacX[ age==5 ] == 1).sum() / len(vacX)
print('Probability to be vaccinated', prob_vac_x)
print('Probability to be vaccinated against X from years 0-2:', prob_vacX_age_1)
print('Probability to be vaccinated against X from years 3-6:', prob_vacX_age_2)
print('Probability to be vaccinated against X from years 7-10:', prob_vacX_age_3)
print('Probability to be vaccinated against X from years 11-13:', prob_vacX_age_4)
print('Probability to be vaccinated against X from years 14-17:', prob_vacX_age_5)

plt.figure('Vaccinated X | age {1 2 3 4}')
probs_vac_x = [prob_vacX_age_1, prob_vacX_age_2, prob_vacX_age_3, prob_vacX_age_4]
plt.plot([1, 2, 3, 4], probs_vac_x)
plt.savefig('./plots/conditional_prob_vacX_age_1234.png')

bike = vaccination_data['knowsToRideABike']
prob_bike = (bike == 1).sum() / len(bike)
prob_bike_not_vac = (bike[ vacX==0 ] == 1).sum() / len(bike)
prob_bike_vac = (bike[ vacX==1 ] == 1).sum() / len(bike)
print('Probability to know how to ride a bike:', prob_bike)
print('Probability to to know how to ride a bike and be vaccinated', prob_bike_vac)
print('Probability to to know how to ride a bike and not to be vaccinated', prob_bike_not_vac)


## we can conclude from the plots, that it is more likely, to have had disease Y or to be vaccinated against X, the older a child is.