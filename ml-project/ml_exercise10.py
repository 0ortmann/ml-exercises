import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy
from matplotlib import cm
import scipy.cluster.hierarchy as cls_h
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, make_scorer
from collections import defaultdict

train = pd.read_csv('./data/train.csv', sep=',')
test = pd.read_csv('./data/test.csv', sep=',')

## b) base stats
# normalize quality related data fields to have a numerical scale:
qs = {'Ex': 10, 'Gd': 8, 'TA': 6, 'Fa': 4, 'Po': 2, 'NA': 0}
qualityAttrs = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
train = train.fillna(0) # replace all NaN values with 0
test = test.fillna(0) # replace all NaN values with 0
for q in qualityAttrs:
    train[q] = train[q].replace(qs)
    train[q] = train[q].astype('int32')
    test[q] = test[q].replace(qs)
    test[q] = test[q].astype('int32')

train = train.select_dtypes(['number', np.number]) # strip all columns we currently cannot use
test = test.select_dtypes(['number', np.number]) # strip all columns we currently cannot use
# basic data analysis & plots for subgroup of features

train_small = train[['SalePrice', 'LotArea', 'OverallQual', 'OverallCond', 'GrLivArea']]

train_small.plot(kind='box', subplots=True)
plt.tight_layout()
plt.savefig('plots/boxplot_selected_features.png')

train_small.hist()
plt.tight_layout()
plt.savefig('plots/hist_selected_features.png')

print('Basic statistics about features of interest (manually selected):')
print(train_small.describe())
print()

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
corr_small = train_small.corr()
plot_corr(corr, 'plots/feature_correlation.png')
plot_corr(corr_small, 'plots/selected_features_correlation')

interest_threshold = 0.6
print('Per column correlation, threshold of interest >= {}'.format(interest_threshold))
for i, x in enumerate(list(corr.columns)):
    for j in list(corr.columns)[i:]:
        if (corr.ix[x,j] > interest_threshold or corr.ix[x,j] < -1 * interest_threshold) and corr.ix[x,j] != 1:
            print(x, ' ', j, ' ', corr.ix[x,j])
print()
def rearrange(corr, df, interest_threshold=0.6):
    X = corr.values
    dist = cls_h.distance.pdist(X)
    link = cls_h.linkage(dist, method='complete')
    ind = cls_h.fcluster(link, interest_threshold*dist.max(), 'distance')
    cols = [df.columns.tolist()[i] for i in list((np.argsort(ind)))]
    return df.reindex(cols, axis=1)
    
rearranged = rearrange(corr, train)
rearranged_small = rearrange(corr_small, train_small)    

plot_corr(rearranged.corr(), 'plots/clustered_features.png')
plot_corr(rearranged_small.corr(), 'plots/clustered_selected_features.png')

print('Plots of the pairwise correlation are to be found in ./plots\n')

## d) clustering (we decided to use k-means)

K = [2, 3, 4]
# only cluster for selected quality related columns
qa = train[['OverallQual', 'ExterQual', 'HeatingQC', 'KitchenQual']]

print('Clustering via KMeans')
def kmeans(df, name):
    for k in K:
        print('Using {} clusters'.format(k))
        kmeans = KMeans(n_clusters=k, random_state=42).fit(df)
        ## now for each cluster determine the some attributes of the SalePrice feature distribution
        cluster = defaultdict(list)
        for i, l in enumerate(kmeans.labels_):
            iPrice = train['SalePrice'][i]
            cluster[l] += [iPrice]
        #print('Divided dataset into {} clusters based on quality attributes. Cluster centroids: {}. Basic statistics of "SalePrice" value:'.format(k, kmeans.cluster_centers_))
        plt.figure()
        for c, houses in cluster.items():
            mean, mi, ma = sum(houses)/float(len(houses),), min(houses), max(houses)
            print('cluster {}: Salesprice mean {}, min {}, max {}'.format(c, mean, mi, ma))
            plt.plot((c, c), (mi, ma), 'k-')
            plt.plot(c, mean, 'ro')
            plt.xlabel('Clusters built by quality ratings')
            plt.ylabel('min/max/mean "SalePrice" per cluster')
            plt.savefig('plots/' + name + 'k-means-saleprice_k={}'.format(k))

print('KMeans on all quality attributes:')
kmeans(qa, 'quality')
print('KMeans on few meaningful features:')
kmeans(train_small, 'selectedfeatures')
print()

## e) dimensionality reduction (we decided to use PCA)

print('Reducing dimensionality via PCA:')


def reduce_dim(data, n_components):
    pca = PCA(n_components=n_components, svd_solver='randomized').fit(data) ## fit train
    return pd.DataFrame(pca.transform(data)) # transform data to fitted model

num_components = [4, 8, 16, 32]
# remove price from training data
labels = train['SalePrice']
train_cleaned = train.drop("SalePrice", 1)
for nc in num_components:
    clf = LinearRegression()
    pca_data = reduce_dim(train_cleaned, nc) ## use full train data set
    cv = KFold(n_splits=5, shuffle=True ,random_state=42)
    r2 = make_scorer(r2_score)
    train_score = cross_val_score(clf, pca_data, labels, cv=cv, scoring=r2)
    print('Reduced dimensionality to {}. Prediction accuracy (linear regression): {}'.format(nc, train_score.mean()))

