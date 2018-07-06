import numpy as np
import pandas as pd

import seaborn
from scipy import stats
from scipy.stats import norm, skew
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import KFold, cross_val_score

from sklearn.linear_model import Lasso, Ridge, ElasticNet, BayesianRidge
from sklearn.svm import SVR
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
print('\nSalePrice after log-scaling.\n  min: {}\n  max: {}\n'.format(np.min(train['SalePrice']), np.max(train['SalePrice'])))
plot_sale_price_dist_and_prob('./plots/saleprice_distribution_log_scaled.png', './plots/saleprice_probability_log_scaled.png')

# store target variable and test IDs, then drop from original data frames:
y_train = train.SalePrice.values
train.drop('SalePrice', axis=1, inplace=True)
train.drop('Id', axis=1, inplace=True)
test_ID = test['Id']
test.drop('Id', axis=1, inplace=True)

merged = pd.concat((train, test)).reset_index(drop=True)
merged = merged.fillna(0) # replace all NaN values with 0

# normalize quality related data fields to have a numerical scale:
qs = {'Ex': 10, 'Gd': 8, 'TA': 6, 'Fa': 4, 'Po': 2, 'NA': 0}
qualityAttrs = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']

for q in qualityAttrs:
    merged[q] = merged[q].replace(qs)
    merged[q] = merged[q].astype('int8')

# one-hot encoding for all remaining categorical values:
merged = pd.get_dummies(merged)

train = merged[:train.shape[0]] # split up again
test = merged[train.shape[0]:]

### evaluate models

def cross_val(model):
    kf = KFold(n_splits=5, shuffle=True, random_state=42).get_n_splits(train.values)
    return -cross_val_score(model, train.values, y_train, scoring='neg_mean_squared_error', cv=kf)

def print_score(model, name):
    score = cross_val(model)
    print('  {}: {:.4f} {:.4f}'.format(name, score.mean(), score.std()))

def print_mse(y, pred, name):
    mse = mean_squared_error(y, pred)
    print('  {}: {:.8f}'.format(name, mse))

lasso = make_pipeline(RobustScaler(), Lasso(alpha = 0.001, random_state=23)) # we found that small alpha performs better (default is 1)
ridge = make_pipeline(RobustScaler(), Ridge(alpha=0.0001, copy_X=True, fit_intercept=True,random_state=42, solver='auto', tol=0.001))
bayesian_ridge = BayesianRidge(copy_X=True)
elastic_net = make_pipeline(RobustScaler(), ElasticNet(alpha=0.001, l1_ratio=1, random_state=23))
svr = make_pipeline(RobustScaler(), SVR(C=100, epsilon=0.01, shrinking=False))

print('\nTesting different regression algorithms, scores:')
print_score(lasso, 'Lasso')
print_score(ridge, 'Ridge Regression')
print_score(bayesian_ridge, 'Bayesian Ridge Regression')
print_score(elastic_net, 'Elastic Net')
print_score(svr, 'Support Vector Regressor')

# fit train data to all models, predict train and test, print mean_squared_error for trainings data
lasso.fit(train, y_train)
lasso_train_pred = lasso.predict(train)
lasso_pred = lasso.predict(test)

ridge.fit(train, y_train)
ridge_train_pred = ridge.predict(train)
ridge_pred = ridge.predict(test)

bayesian_ridge.fit(train, y_train)
bayesian_ridge_train_pred = bayesian_ridge.predict(train)
bayesian_ridge_pred = bayesian_ridge.predict(test)

elastic_net.fit(train, y_train)
elastic_net_train_pred = elastic_net.predict(train)
elastic_net_pred = elastic_net.predict(test)

svr.fit(train, y_train)
svr_train_pred = svr.predict(train)
svr_pred = svr.predict(test)

print('\nMean squared error on training data:')
print_mse(y_train, lasso_train_pred, 'Lasso')
print_mse(y_train, ridge_train_pred, 'Ridge Regression')
print_mse(y_train, bayesian_ridge_train_pred, 'Bayesian Ridge Regression')
print_mse(y_train, elastic_net_train_pred, 'Elastic Net')
print_mse(y_train, svr_train_pred, 'Support Vector Regressor')

## the prints show that ridge performed best on the trainings data. Also the variance of ridge was lowest among all scores.

# inverse to np.log1p:

def make_submission(y_pred):
    y_pred_real_values = np.expm1(y_pred)
    sub = pd.DataFrame()
    sub['Id'] = test_ID
    sub['SalePrice'] = y_pred_real_values
    sub.to_csv('./data/submission.csv', index=False)

make_submission(lasso_pred)

## lasso and elastic net scored exactly the same, bayesian ridge is ok-ish. ridge overfits a bit. svr greatly overfits.