import numpy as np
import pandas as pd

import seaborn
from scipy.stats import norm, probplot
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import KFold, cross_val_score

from sklearn.linear_model import Lasso, Ridge, ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

from sklearn.decomposition import PCA

# load train data, split to train and test sets 
train = pd.read_csv('./data/train.csv', sep=',')
test = pd.read_csv('./data/test.csv', sep=',')

# from assignment 10 we found some features to be highly correlated with SalePrice.
# take a detailed look at those:

sp_corr_features = ['GrLivArea', 'TotalBsmtSF', 'LotArea']
for corr in sp_corr_features:
    plt.figure()
    view = pd.concat([train['SalePrice'], train[corr]], axis=1)
    view.plot.scatter(x=corr, y='SalePrice')
    plt.savefig('./plots/scatter_correlation_{}_saleprice.png'.format(corr))

# remove outliers based on the findings of the previous plots
train = train.drop(train[ train['GrLivArea'] > 4000 ].index)
train = train.drop(train[ train['TotalBsmtSF'] > 6000 ].index)
train = train.drop(train[ train['LotArea'] > 100000 ].index)

# plot distribution of a variable compared to normal distribution and plot probability distribution compared to linear distributed quantiles.
def plot_dist_and_prob(var, label, df):
    plt.figure()
    seaborn.distplot(df[var] , fit=norm);
    plt.savefig('./plots/{}_distribution_{}.png'.format(var, label))
    plt.figure()
    probplot(df[var], plot=plt)
    plt.savefig('./plots/{}_prob_plot_{}.png'.format(var, label))

plot_dist_and_prob('SalePrice', 'orig', train)

# check the plots -> saleprice is right skewed. apply log1p to pull-in values
train['SalePrice'] = np.log1p(train['SalePrice'])
#print('\nSalePrice after log-scaling.\n  min: {}\n  max: {}'.format(np.min(train['SalePrice']), np.max(train['SalePrice'])))
plot_dist_and_prob('SalePrice', 'scaled_log', train)

# store target variable and test IDs, then drop from original data frames:
y_train = train.SalePrice.values
train.drop('SalePrice', axis=1, inplace=True)
train.drop('Id', axis=1, inplace=True)
test_ID = test['Id']
test.drop('Id', axis=1, inplace=True)

merged = pd.concat((train, test)).reset_index(drop=True)

# correct more skewness in data that is directly correlated to SalePrice:
plot_dist_and_prob('GrLivArea', 'orig', merged)
merged['GrLivArea'] = np.log1p(merged['GrLivArea'])
plot_dist_and_prob('GrLivArea', 'scaled_log', merged)

plot_dist_and_prob('LotArea', 'orig', merged)
merged['LotArea'] = np.log1p(merged['LotArea'])
plot_dist_and_prob('LotArea', 'scaled_log', merged)

# replace NA values
replace_na = ['MiscFeature', 'Alley', 'Fence', 'GarageType', 'GarageFinish', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'MSSubClass']
for repl in replace_na:
    merged[repl] = merged[repl].fillna('None')
merged = merged.fillna(0) # replace all remaining NA values with 0

# normalize quality related data fields to have a numerical scale:
q_replace = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
quality_attrs = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']

for q in quality_attrs:
    merged[q] = merged[q].replace(q_replace)
    merged[q] = merged[q].astype('int8')

# transform date-related numerical values to categorical values, then label encode them
num_to_cat = ['MSSubClass', 'YrSold', 'MoSold', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt']
for col in num_to_cat:
    merged[col] = merged[col].apply(str)
    merged[col] = LabelEncoder().fit_transform(list(merged[col].values))

# one-hot encoding for all remaining categorical values:
merged = pd.get_dummies(merged)

train = merged[:train.shape[0]] # split up again
test = merged[train.shape[0]:]

### evaluate models

def cross_val(model, data=train):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    return -cross_val_score(model, data.values, y_train, scoring='neg_mean_squared_error', cv=kf)

def print_score(model, name, data=train):
    score = cross_val(model, data)
    print('  {}: {:.5f} {:.5f}'.format(name, score.mean(), score.std()))

def print_mse(y, pred, name):
    mse = mean_squared_error(y, pred)
    print('  {}: {:.8f}'.format(name, mse))

lasso = make_pipeline(RobustScaler(), Lasso(alpha = 0.00055))
ridge = make_pipeline(RobustScaler(), Ridge(alpha=25, tol=0.00001))
bayesian_ridge = make_pipeline(RobustScaler(), BayesianRidge())
elastic_net = make_pipeline(RobustScaler(), ElasticNet(alpha=0.00055, l1_ratio=0.7))
svr = make_pipeline(RobustScaler(), SVR(C=10, epsilon=0.001, shrinking=False))

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

print('\nReducing Dimensionality with PCA:')

def reduce_dim(data, n_components):
    pca = PCA(n_components=n_components, svd_solver='randomized').fit(data) ## fit train
    return pd.DataFrame(pca.transform(data)) # transform data to fitted model

num_components = [8, 16, 32, 48]
for nc in num_components:
    lasso = make_pipeline(RobustScaler(), Lasso(alpha = 0.0005, random_state=42))
    pca_data = reduce_dim(train, nc) ## use full train data set
    print_score(lasso, 'Lasso (reduced dimensions: {})'.format(nc), pca_data)

def make_submission(y_pred):
    y_pred_real_values = np.expm1(y_pred) # inverse to np.log1p
    sub = pd.DataFrame()
    sub['Id'] = test_ID
    sub['SalePrice'] = y_pred_real_values
    sub.to_csv('./data/submission.csv', index=False)

make_submission(lasso_pred)

## lasso and elastic net scored exactly the same, bayesian ridge is ok-ish. ridge overfits a bit. svr greatly overfits.