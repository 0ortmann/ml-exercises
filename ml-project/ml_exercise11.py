import numpy as np
import pandas as pd

import seaborn
from scipy import stats
from scipy.stats import norm, skew
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.metrics import mean_squared_error

# load train data, split to train and test sets 
train = pd.read_csv('./data/train.csv', sep=',')
test = pd.read_csv('./data/test.csv', sep=',')

# plot distribution of saleprice variable compared to normal distribution and plot probability distribution compared to linear distributed quantiles.
def plot_sale_price_dist_and_prob(label_dist, label_prob):
    plt.figure()
    seaborn.distplot(train['SalePrice'] , fit=norm);
    plt.savefig(label_dist)
    plt.figure()
    stats.probplot(train['SalePrice'], plot=plt)
    plt.savefig(label_prob)

plot_sale_price_dist_and_prob('./plots/saleprice_distribution_orig.png', './plots/saleprice_probability.png')

# check the plots -> saleprice is right-skewed. apply log1p to pull-in values
train['SalePrice'] = np.log1p(train['SalePrice'])
plot_sale_price_dist_and_prob('./plots/saleprice_distribution_log_scaled.png', './plots/saleprice_probability_log_scaled.png')

# normalize quality related data fields to have a numerical scale:
qs = {'Ex': 10, 'Gd': 8, 'TA': 6, 'Fa': 4, 'Po': 2, 'NA': 0}
qualityAttrs = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']

 # replace all NaN values with 0
train = train.fillna(0)
test = test.fillna(0)
for q in qualityAttrs:
    train[q] = train[q].replace(qs)
    train[q] = train[q].astype('int32')
    test[q] = test[q].replace(qs)
    test[q] = test[q].astype('int32')

# strip all remaining columns with categorical features 
train = train.select_dtypes(['number', np.number])
test = test.select_dtypes(['number', np.number])

# store ID columns, then drop them
train_ID = train['Id']
train.drop('Id', axis = 1, inplace = True)
test_ID = test['Id']
test.drop('Id', axis = 1, inplace = True)

y_train = train.SalePrice.values
train.drop(['SalePrice'], axis=1, inplace=True)
print('\nSalePrice after log-scaling.\n  min: {}\n  max: {}\n'.format(np.min(y_train), np.max(y_train)))

### evaluate models

def cross_val(model):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_score(model, train.values, y_train, scoring='neg_mean_squared_error', cv = kf)


