import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import math


train = pd.read_csv("./data/train.csv", sep=',')

## b) base stats

def base_statistics(df):
    print('Maximum values:\n', df.max(numeric_only=True), '\n')
    print('Minimum values:\n', df.min(numeric_only=True), '\n')
    print('Mean:\n', df.mean(numeric_only=True), '\n')
    print('Median:\n', df.median(), '\n')
    print('Variance:\n', df.var(numeric_only=True), '\n')

## base_statistics(train)

# normalize quality related data fields to have a numerical scale:
qs = {'Ex': 10, 'Gd': 8, 'TA': 6, 'Fa': 4, 'Po': 2, 'NA': 0}
qualityAttrs = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']

for q in qualityAttrs:
    train[q] = train[q].replace(qs)

base_statistics(train)

### test some correlations
#### house size and sale prize
#### sales prize, pools