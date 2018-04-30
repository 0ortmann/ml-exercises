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