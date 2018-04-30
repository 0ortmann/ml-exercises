#!/bin/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


## assignment 1

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