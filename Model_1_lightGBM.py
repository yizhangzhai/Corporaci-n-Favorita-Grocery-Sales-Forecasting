import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import date, timedelta
import gc
import lightgbm as lgb
import matplotlib.pyplot as plt
%matplotlib inline

################################################
X_train = pd.read_pickle('X_train.pkl')
y_train = pd.read_pickle('y_train.pkl')
X_val = pd.read_pickle('X_val.pkl')
y_val = pd.read_pickle('y_val.pkl')
X_test = pd.read_pickle('X_test.pkl')

val_pred, test_pred = [], []

n_rounds = 5000
params = {
    'num_leaves': 80,
    'objective': 'regression',
    'min_data_in_leaf': 200,
    'learning_rate': 0.02,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'metric': 'l2',
    'num_threads': 16}

cate_vars = []
for i in range(16):
        print('Predctint Day %s ... ... ' %(i+1))
        dtrain = lgb.Dataset(X_train, label=y_train[:, i],categorical_feature=cate_vars,weight=pd.concat([items["perishable"]] * 6) * 0.25 + 1)
        dval = lgb.Dataset(X_val, label=y_val[:, i], reference=dtrain,categorical_feature=cate_vars,weight=pd.concat([items["perishable"]] * 1) * 0.25 + 1)

        regressor = lgb.train(params, dtrain, num_boost_round=n_rounds, valid_sets=[dtrain, dval], early_stopping_rounds=100, verbose_eval=50)
        val_pred.append(regressor.predict(X_val, num_iteration=regressor.best_iteration))
        test_pred.append(regressor.predict(X_test, num_iteration=regressor.best_iteration))

weight = items["perishable"] * 0.25 + 1
err = (y_val - np.array(val_pred).transpose())**2
err = err.sum(axis=1) * weight
err = np.sqrt(err.sum() / weight.sum() / 16)

importance_df = pd.DataFrame()
importance_df["feature"] = X_train.columns
importance_df["importance"] = regressor.feature_importance('gain')
importance_df = importance_df.sort_values(by='importance', ascending=False)

test_pred.to_csv('test_pred_model_1.csv',index=False)
