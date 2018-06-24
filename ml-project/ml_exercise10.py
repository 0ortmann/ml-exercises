import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy
import scipy.cluster.hierarchy as cls_h

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
train = train.fillna(0) # replace all NaN values with 0
for q in qualityAttrs:
    train[q] = train[q].replace(qs)
    train[q] = train[q].astype('int32')

train = train.select_dtypes(['number', np.number]) # strip all columns we currently cannot use

base_statistics(train)

## c) identify features of interest
def plot_corr(corr, filename, size=10):
    plt.figure()
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90);
    plt.yticks(range(len(corr.columns)), corr.columns);
    cbar = fig.colorbar(cax, ticks=[-1, 0, 1], aspect=40, shrink=.8)
    plt.savefig(filename)

corr = train.corr()
plot_corr(corr, 'plots/feature_correlation.png')

interest_threshold = 0.6

for i, x in enumerate(list(corr.columns)):
    for j in list(corr.columns)[i:]:
        if (corr.ix[x,j] > interest_threshold or corr.ix[x,j] < -1 * interest_threshold) and corr.ix[x,j] != 1:
            print(x, ' ', j, ' ', corr.ix[x,j])


X = corr.values
dist = cls_h.distance.pdist(X)
link = cls_h.linkage(dist, method='complete')
ind = cls_h.fcluster(link, interest_threshold*dist.max(), 'distance')
cols = [train.columns.tolist()[i] for i in list((np.argsort(ind)))]
rearranged = train.reindex(cols, axis=1)

plot_corr(rearranged.corr(), 'plots/clustered_features.png')