import sys
import time

import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
import random

import lightgbm as lgb


start_time = time.time()

df_calendar = pd.read_csv("calendar.csv")

# print(df_calendar.head(5))

df_sales = pd.read_csv("sales_train_validation.csv")

output_rows = list(df_sales['id'])
num_of_time_series = len(df_sales)
output_rows_eval = [item_id[:-10] + 'evaluation' for item_id in output_rows]
# print( output_rows_eval[:15])
output_rows = output_rows + output_rows_eval

df_out = pd.DataFrame(index = np.arange(len(output_rows)), columns = ['id'] + [f'F{i}' for i in range(1, 29)])
df_out['id'] = output_rows

n = len(df_sales['id'])
output_index = np.arange(n)


#encoding events
df_calendar['event_name_1'].fillna('unknown', inplace=True)
event_encoder = preprocessing.LabelEncoder()
df_calendar['event_name_1'] = event_encoder.fit_transform(df_calendar['event_name_1'])


# form dataframe for prediction
columns_for_means = ['id'] + [f'd_{i}' for i in range(1863, 1914)]
df_means_for_prediction = df_sales[columns_for_means]
df_means_for_prediction = pd.melt(df_means_for_prediction, id_vars=['id'],
                   var_name='day', value_name='sales')
df_means_for_prediction = pd.merge(df_means_for_prediction, df_calendar, how = 'inner',
                                   left_on = ['day'], right_on = ['d'])

for i in range(4):
    df_means_for_prediction[f'prev_week_{i}'] = \
        df_means_for_prediction.groupby(['id'])['sales'].transform(lambda x: x.shift(i*7).rolling(7).mean())

df_means_for_prediction = df_means_for_prediction[df_means_for_prediction['day'] == 'd_1913']
df_means_for_prediction.reset_index(inplace=True)

df_means_for_prediction.drop(columns=['index', 'id', 'day', 'sales', 'date',
                                      'wm_yr_wk', 'weekday', 'wday', 'month', 'year', 'd',
                                        'event_type_1', 'event_name_2',
                                        'event_type_2', 'snap_CA', 'snap_TX', 'snap_WI'], inplace=True)


#form dataset for learning
list_series_to_choose = sorted(random.sample([i for i in range(len(df_sales))], 2000))
df_sales = df_sales.iloc[list_series_to_choose, :]

df_sales = pd.melt(df_sales, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
                   var_name='day', value_name='sales')
df_sales = pd.merge(df_sales, df_calendar, how = 'inner', left_on = ['day'], right_on = ['d'])

df_sales.drop(['item_id', 'dept_id', 'cat_id', 'store_id',
               'state_id', 'day', 'date', 'wm_yr_wk', 'weekday', 'd', 'event_type_1',
               'event_name_2', 'event_type_2', 'snap_CA', 'snap_TX', 'snap_WI'], inplace = True, axis = 1)

for k in range(28 * 2):
    print(f'k={k}')

    df_for_train = df_sales.copy()
    for i in range(4):
        df_for_train[f'prev_week_{i}'] = \
            df_for_train.groupby(['id'])['sales'].transform(lambda x: x.shift(k + i * 7 + 1).rolling(7).mean())

    df_for_train.dropna(inplace=True)

    params = {'num_leaves': 555,
              'min_child_weight': 0.034,
              'feature_fraction': 0.379,
              'bagging_fraction': 0.418,
              'min_data_in_leaf': 106,
              'objective': 'regression',
              'max_depth': -1,
              'learning_rate': 0.005,
              "boosting_type": "gbdt",
              "bagging_seed": 11,
              "metric": 'rmse',
              "verbosity": -1,
              'reg_alpha': 0.3899,
              'reg_lambda': 0.648,
              'random_state': 222,
             }

    n = len(df_for_train)

    m = int(0.75 * n)

    df_train = df_for_train.iloc[:m, :]
    df_valid = df_for_train.iloc[m:, :]

    columns = ['wday', 'month', 'year', 'event_name_1'] + \
                [f'prev_week_{i}' for i in range(4)]

    dtrain = lgb.Dataset(df_train[columns], label=df_train['sales'])
    dvalid = lgb.Dataset(df_valid[columns], label=df_valid['sales'])

    model = lgb.train(params, dtrain, 3000, valid_sets=[dtrain, dvalid],
                      early_stopping_rounds=50, verbose_eval=50)

    # df_for_prediction = pd.DataFrame(index = output_index, columns = columns)
    df_for_prediction = df_means_for_prediction.copy()

    output_wday = df_calendar.loc[len(df_calendar)-28 * 2 + k, 'wday']
    output_month = df_calendar.loc[len(df_calendar)-28 * 2 + k, 'month']
    output_year = df_calendar.loc[len(df_calendar)-28 * 2 + k, 'year']
    output_event_name_1 = df_calendar.loc[len(df_calendar)-28 * 2 + k, 'event_name_1']

    df_for_prediction['wday'] = output_wday
    df_for_prediction['month'] = output_month
    df_for_prediction['year'] = output_year
    df_for_prediction['event_name_1'] = output_event_name_1

    predictions = model.predict(df_for_prediction)

    if k < 28:
        df_out.iloc[:num_of_time_series, k + 1] = predictions
    else:
        df_out.iloc[num_of_time_series:, k - 28 + 1] = predictions

df_out.to_csv("output_lgbm_06.csv", index=False)

end_time = time.time()
seconds = end_time - start_time
print(f"Program took {seconds/60} minutes")

