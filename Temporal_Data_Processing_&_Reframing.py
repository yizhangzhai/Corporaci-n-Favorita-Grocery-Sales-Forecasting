import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import date, timedelta
import gc
import lightgbm as lgb
import matplotlib.pyplot as plt
%matplotlib inline

train = pd.read_csv('train.csv', parse_dates=['date'])
train = train.loc[train.date>='2017-02-01']
train['unit_sales'] = [np.log1p(x) if x>0 else 0 for x in train.unit_sales.astype(float)]
test = pd.read_csv('test.csv', parse_dates=['date'])
items = pd.read_csv('items.csv')
transactions = pd.read_csv('transactions.csv')
stores = pd.read_csv('stores.csv')
oil = pd.read_csv('oil.csv', parse_dates=['date'])
holidays_events = pd.read_csv('holidays_events.csv', parse_dates=['date'])

le = LabelEncoder()
items['family'] = le.fit_transform(items.family)
stores['city'] = le.fit_transform(stores.city)
stores['state'] = le.fit_transform(stores.state)
stores['type'] = le.fit_transform(stores.type)

train['weekofyear'] = train.date.dt.weekofyear
train['weekday'] = train.date.dt.weekday
train['month'] = train.date.dt.month
test['weekofyear'] = test.date.dt.weekofyear
test['weekday'] = test.date.dt.weekday
test['month'] = test.date.dt.month


############################################################################################
sales_train = train.set_index(['store_nbr','item_nbr','date'])[['unit_sales']].unstack(level=-1)
sales_train.fillna(0,inplace=True)
sales_train.columns = sales_train.columns.get_level_values(1)

promotion_train = train.set_index(['store_nbr','item_nbr','date'])[['onpromotion']].unstack(level=-1)
promotion_train.fillna(False,inplace=True)
promotion_train.columns = promotion_train.columns.get_level_values(1)
promotion_train = promotion_train * 1

promotion_test = test.set_index(['store_nbr','item_nbr','date'])[['onpromotion']].unstack(level=-1)
promotion_test.fillna(False,inplace=True)
promotion_test.columns = promotion_test.columns.get_level_values(1)
promotion_test = promotion_test * 1

promotion_test = promotion_test.reindex(promotion_train.index)
promotion_all = pd.concat([promotion_train,promotion_test],axis=1)

items = items.set_index('item_nbr')
items = items.reindex(sales_train.index.get_level_values(1))
stores = stores.set_index('store_nbr')
stores = stores.reindex(sales_train.index.get_level_values(0))

### intercation data
# stores * item.class
sales_train__store_class = sales_train.reset_index()
sales_train__store_class['item_class'] = items['class'].values
sales_train__store_class___index = sales_train__store_class[['item_class','store_nbr']]
sales_train__store_class = sales_train__store_class.groupby(['item_class','store_nbr'])[sales_train.columns].sum(axis=1)

promotion_train__store_class = promotion_all.reset_index()
promotion_train__store_class['item_class'] = items['class'].values
promotion_train__store_class___index = promotion_train__store_class[['item_class','store_nbr']]
promotion_train__store_class = promotion_train__store_class.groupby(['item_class','store_nbr'])[promotion_all.columns].sum(axis=1)


# items
sales_train__items = sales_train.reset_index()
sales_train__items___index = sales_train__items['item_nbr']
sales_train__items = sales_train__items.groupby(['item_nbr'])[sales_train.columns].sum(axis=1)

promotion_train__items = promotion_all.reset_index()
promotion_train__items___index = promotion_train__items['item_nbr']
promotion_train__items = promotion_train__items.groupby(['item_nbr'])[promotion_all.columns].sum(axis=1)


# items * stores.city
sales_train__item_city = sales_train.reset_index()
sales_train__item_city['city'] = stores['city'].values
sales_train__item_city___index =sales_train__item_city[['city','item_nbr']]
sales_train__item_city = sales_train__item_city.groupby(['city','item_nbr'])[sales_train.columns].sum(axis=1)

promotion_train__item_city = promotion_all.reset_index()
promotion_train__item_city['city'] = stores['city'].values
promotion_train__item_city___index =promotion_train__item_city[['city','item_nbr']]
promotion_train__item_city = promotion_train__item_city.groupby(['city','item_nbr'])[promotion_all.columns].sum(axis=1)


############################################################################################
### training starts from 20170614 to 20170719
### validation starts from 20170726 to 20170810
### testing starts from 20170816 to 20170831

######  Feature Engineering

def data_generator(dat,other_dat,reference_day,n_date_point,date_interval=7,addtional_colnames='',test_data=False):
    X, Y = [], []
    ready={}
    day_delta = [int(x) for x in date_interval*np.arange(n_date_point)]
    addtional_colnames = addtional_colnames+'_'
    for d in day_delta:
        new_day = reference_day + timedelta(d)
        print(new_day)
        time_delta = [3,7,14,30,90]

        for t in time_delta:
            temp = dat[pd.date_range(new_day-timedelta(t),periods=t,freq='D')]
            weights = [1.5 if x in (5,6) else 1 for x in pd.Series(pd.date_range(new_day-timedelta(t),periods=t,freq='D')).dt.weekday.values]
            ready[addtional_colnames+'sales_before_%s_days_mean_weekday_weighted' %t] = (temp * weights).mean(axis=1).values
            ready[addtional_colnames+'sales_before_%s_days_mean' %t] = temp.mean(axis=1).values
            ready[addtional_colnames+'sales_before_%s_days_std' %t] = temp.std(axis=1).values
            ready[addtional_colnames+'sales_before_%s_days_min' %t] = temp.min(axis=1).values
            ready[addtional_colnames+'sales_before_%s_days_max' %t] = temp.max(axis=1).values
            ready[addtional_colnames+'sales_before_%s_days_sum' %t] = temp.sum(axis=1).values
            ready[addtional_colnames+'sales_before_%s_days_mean_of_diff' %t] = temp.diff(axis=1).mean(axis=1).values
            ready[addtional_colnames+'sales_before_%s_days_mean_decay' %t] = (temp * np.power(0.9, np.arange(t)[::-1])).mean(axis=1).values


        ### 1-week ago
        for t in time_delta:
            temp = dat[pd.date_range((new_day-timedelta(7))-timedelta(t),periods=t,freq='D')]
            weights = [1.5 if x in (5,6) else 1 for x in pd.Series(pd.date_range((new_day-timedelta(7))-timedelta(t),periods=t,freq='D')).dt.weekday.values]
            ready[addtional_colnames+'sales_before_%s_days_plus_one_week_mean_weekday_weighted' %t] = (temp * weights).mean(axis=1).values
            ready[addtional_colnames+'sales_before_%s_days_plus_one_week_mean' %t] = temp.mean(axis=1).values
            ready[addtional_colnames+'sales_before_%s_days_plus_one_week_std' %t] = temp.std(axis=1).values
            ready[addtional_colnames+'sales_before_%s_days_plus_one_week_min' %t] = temp.min(axis=1).values
            ready[addtional_colnames+'sales_before_%s_days_plus_one_week_max' %t] = temp.max(axis=1).values
            ready[addtional_colnames+'sales_before_%s_days_plus_one_week_sum' %t] = temp.sum(axis=1).values
            ready[addtional_colnames+'sales_before_%s_days_plus_one_week_mean_of_diff' %t] = temp.diff(axis=1).mean(axis=1).values
            ready[addtional_colnames+'sales_before_%s_days_plus_one_week_mean_decay' %t] = (temp * np.power(0.9, np.arange(t)[::-1])).mean(axis=1).values

        ### 1-month ago
        for t in time_delta:
            temp = dat[pd.date_range((new_day-timedelta(30))-timedelta(t),periods=t,freq='D')]
            weights = [1.5 if x in (5,6) else 1 for x in pd.Series(pd.date_range((new_day-timedelta(30))-timedelta(t),periods=t,freq='D')).dt.weekday.values]
            ready[addtional_colnames+'sales_before_%s_days_plus_one_month_mean_weekday_weighted' %t] = (temp * weights).mean(axis=1).values
            ready[addtional_colnames+'sales_before_%s_days_plus_one_month_mean' %t] = temp.mean(axis=1).values
            ready[addtional_colnames+'sales_before_%s_days_plus_one_month_std' %t] = temp.std(axis=1).values
            ready[addtional_colnames+'sales_before_%s_days_plus_one_month_min' %t] = temp.min(axis=1).values
            ready[addtional_colnames+'sales_before_%s_days_plus_one_month_max' %t] = temp.max(axis=1).values
            ready[addtional_colnames+'sales_before_%s_days_plus_one_month_sum' %t] = temp.sum(axis=1).values
            ready[addtional_colnames+'sales_before_%s_days_plus_one_month_mean_of_diff' %t] = temp.diff(axis=1).mean(axis=1).values
            ready[addtional_colnames+'sales_before_%s_days_plus_one_month_mean_decay' %t] = (temp * np.power(0.9, np.arange(t)[::-1])).mean(axis=1).values

        ### sales CNT
        for t in time_delta:
            temp = (dat[pd.date_range(new_day-timedelta(t),periods=t,freq='D')] >0) * 1
            weights = [1.5 if x in (5,6) else 1 for x in pd.Series(pd.date_range(new_day-timedelta(t),periods=t,freq='D')).dt.weekday.values]
            ready[addtional_colnames+'sales_CNT_before_%s_days_mean_weekday_weighted' %t] = (temp * weights).mean(axis=1).values
            ready[addtional_colnames+'sales_CNT_before_%s_days_mean' %t] = temp.mean(axis=1).values
            ready[addtional_colnames+'sales_CNT_before_%s_days_std' %t] = temp.std(axis=1).values
            ready[addtional_colnames+'sales_CNT_before_%s_days_min' %t] = temp.min(axis=1).values
            ready[addtional_colnames+'sales_CNT_before_%s_days_max' %t] = temp.max(axis=1).values
            ready[addtional_colnames+'sales_CNT_before_%s_days_sum' %t] = temp.sum(axis=1).values
            ready[addtional_colnames+'sales_CNT_before_%s_days_mean_of_diff' %t] = temp.diff(axis=1).mean(axis=1).values
            ready[addtional_colnames+'sales_CNT_before_%s_days_mean_decay' %t] = (temp * np.power(0.9, np.arange(t)[::-1])).mean(axis=1).values

        ### promotion
        for t in time_delta:
            temp = other_dat[0][pd.date_range(new_day-timedelta(t),periods=t,freq='D')]
            ready[addtional_colnames+'promotion_CNT_before_%s_days_mean' %t] = temp.mean(axis=1).values
            ready[addtional_colnames+'promotion_CNT_before_%s_days_sum' %t] = temp.sum(axis=1).values
            if t<16:
                temp = other_dat[0][pd.date_range(new_day,periods=t,freq='D')]
                ready[addtional_colnames+'promotion_CNT_after_%s_days_mean' %t] = temp.mean(axis=1).values
                ready[addtional_colnames+'promotion_CNT_after_%s_days_sum' %t] = temp.sum(axis=1).values

        for t in time_delta:
            temp = (other_dat[0][pd.date_range(new_day-timedelta(t),periods=t,freq='D')]>0)*1
            ready[addtional_colnames+'promotion_TRUE_before_%s_days_mean' %t] = temp.mean(axis=1).values.astype(int).reshape(-1)
            ready[addtional_colnames+'promotion_TRUE_before_%s_days_sum' %t] = temp.sum(axis=1).values.astype(int).reshape(-1)
            if t<16:
                temp = (other_dat[0][pd.date_range(new_day,periods=t,freq='D')]>0)*1
                ready[addtional_colnames+'promotion_TRUE_after_%s_days_mean' %t] = temp.mean(axis=1).values.astype(int).reshape(-1)
                ready[addtional_colnames+'promotion_TRUE_after_%s_days_sum' %t] = temp.sum(axis=1).values.astype(int).reshape(-1)

        for t in time_delta:
            temp = dat[pd.date_range(new_day-timedelta(t),periods=t,freq='D')] * other_dat[0][pd.date_range(new_day-timedelta(t),periods=t,freq='D')]
            ready[addtional_colnames+'promoted_sales_before_%s_days_plus_one_month_mean' %t] = temp.mean(axis=1).values
            ready[addtional_colnames+'promoted_sales_before_%s_days_plus_one_month_std' %t] = temp.std(axis=1).values
            ready[addtional_colnames+'promoted_sales_before_%s_days_plus_one_month_min' %t] = temp.min(axis=1).values
            ready[addtional_colnames+'promoted_sales_before_%s_days_plus_one_month_max' %t] = temp.max(axis=1).values
            ready[addtional_colnames+'promoted_sales_before_%s_days_plus_one_month_sum' %t] = temp.sum(axis=1).values
            ready[addtional_colnames+'promoted_sales_before_%s_days_plus_one_month_mean_of_diff' %t] = temp.diff(axis=1).mean(axis=1).values
            ready[addtional_colnames+'promoted_sales_before_%s_days_plus_one_month_mean_decay' %t] = (temp * np.power(0.9, np.arange(t)[::-1])).mean(axis=1).values

        ### Single day
        for i in range(1,8):
            temp = dat[pd.date_range(new_day-timedelta(i*7),periods=1,freq='7D')]
            weights = [1.5 if x in (5,6) else 1 for x in pd.Series(pd.date_range(new_day-timedelta(i*7),periods=1,freq='7D')).dt.weekday.values]
            ready[addtional_colnames+'sales_on_the_day_before_%s_weeks_mean_weekday_weighted' %t] = (temp * weights).mean(axis=1).values
            ready[addtional_colnames+'sales_on_the_day_before_%s_weeks_raw' %i] = temp.values.astype(int).reshape(-1)
            ready[addtional_colnames+'sales_TRUE_on_the_day_before_%s_weeks_raw' %i] = ((temp.values>0)*1).reshape(-1)

            temp = other_dat[0][pd.date_range(new_day-timedelta(i*7),periods=1,freq='7D')]
            ready[addtional_colnames+'promotion_on_the_day_before_%s_weeks_raw' %i] = temp.values.astype(int).reshape(-1)
            ready[addtional_colnames+'promotion_TRUE_on_the_day_before_%s_weeks_raw' %i] = ((temp.values>0)*1).reshape(-1)

        if test_data:
            y = []
        else:
            y = dat[pd.date_range(new_day, periods=16)].values


        X.append(pd.DataFrame(ready))
        Y.append(y)

    X = pd.concat(X,axis=0)
    Y = np.concatenate(Y,axis=0)

    return X,Y

### Train data
reference_day = date(2017,6,14)
X_train,y_train = data_generator(sales_train,other_dat=[promotion_all],reference_day=reference_day,n_date_point=6,date_interval=7)
#  b.
X_train_store_item,_ = data_generator(sales_train__store_class,other_dat=[promotion_train__store_class],reference_day=reference_day,n_date_point=1,date_interval=7,addtional_colnames='store_class')
X_train_store_item.index = sales_train__store_class.index
X_train_store_item = X_train_store_item.reindex(sales_train__store_class___index).reset_index(drop=True)

#  c.
X_train_store_item,_ = data_generator(sales_train__items,other_dat=[promotion_train__items],reference_day=reference_day,n_date_point=1,date_interval=7,addtional_colnames='items')
X_train_store_item.index = sales_train__items.index
X_train_store_item = X_train_store_item.reindex(sales_train__items___index).reset_index(drop=True)

#  d.
X_train_item_city,_ = data_generator(sales_train__item_city,other_dat=[promotion_train__item_city],reference_day=reference_day,n_date_point=1,date_interval=7,addtional_colnames='item_city')
X_train_item_city.index = sales_train__item_city.index
X_train_item_city = X_train_item_city.reindex(sales_train__item_city___index).reset_index(drop=True)

# combine
X_train = pd.concat([X_train,X_train_store_item,X_train_store_item,X_train_item_city,items,stores],axis=1)

### -------------------------------------------- Val data
reference_day = date(2017,7,26)
#  a.
X_val,y_val = data_generator(sales_train,other_dat=[promotion_all],reference_day=reference_day,n_date_point=1,date_interval=7,addtional_colnames='main')

#  b.
X_val_store_item,_ = data_generator(sales_train__store_class,other_dat=[promotion_train__store_class],reference_day=reference_day,n_date_point=1,date_interval=7,addtional_colnames='store_class')
X_val_store_item.index = sales_train__store_class.index
X_val_store_item = X_val_store_item.reindex(sales_train__store_class___index).reset_index(drop=True)

#  c.
X_val_store_item,_ = data_generator(sales_train__items,other_dat=[promotion_train__items],reference_day=reference_day,n_date_point=1,date_interval=7,addtional_colnames='items')
X_val_store_item.index = sales_train__items.index
X_val_store_item = X_val_store_item.reindex(sales_train__items___index).reset_index(drop=True)

#  d.
X_val_item_city,_ = data_generator(sales_train__item_city,other_dat=[promotion_train__item_city],reference_day=reference_day,n_date_point=1,date_interval=7,addtional_colnames='item_city')
X_val_item_city.index = sales_train__item_city.index
X_val_item_city = X_val_item_city.reindex(sales_train__item_city___index).reset_index(drop=True)

# combine
X_val = pd.concat([X_val,X_val_store_item,X_val_store_item,X_val_item_city,items,stores],axis=1)

### --------------------------------------------- Test data
reference_day = date(2017,8,16)
# a.
X_test, _ = data_generator(sales_train,other_dat=[promotion_all],reference_day=reference_day,n_date_point=1,date_interval=7,test_data=True)

#  b.
X_test_store_item,_ = data_generator(sales_train__store_class,other_dat=[promotion_train__store_class],reference_day=reference_day,n_date_point=1,date_interval=7)
X_val_store_item.index = sales_train__store_class.index
X_test_store_item = X_val_store_item.reindex(sales_train__store_class___index).reset_index(drop=True)

#  c.
X_test_store_item,_ = data_generator(sales_train__items,other_dat=[promotion_train__items],reference_day=reference_day,n_date_point=1,date_interval=7)
X_test_store_item.index = sales_train__items.index
X_test_store_item = X_val_store_item.reindex(sales_train__items___index).reset_index(drop=True)

#  d.
X_test_item_city,_ = data_generator(sales_train__item_city,other_dat=[promotion_train__item_city],reference_day=reference_day,n_date_point=1,date_interval=7)
X_test_item_city.index = sales_train__item_city.index
X_test_item_city = X_val_item_city.reindex(sales_train__item_city___index).reset_index(drop=True)

# combine
X_test = pd.concat([X_test,X_test_store_item,X_test_store_item,X_test_item_city,items,stores],axis=0)

#################################################################################################
X_train.to_pickle('X_train.pkl')
y_train.to_pickle('y_train.pkl')
X_val.to_pickle('X_val.pkl')
y_val.to_pickle('y_val.pkl')
X_test.to_pickle('X_test.pkl')
