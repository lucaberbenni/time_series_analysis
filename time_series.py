import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

import warnings
warnings.simplefilter('ignore')

'''
import and clean data
'''
df_raw = pd.read_csv(
    'data/TG_STAID002759.txt', 
    sep=',', 
    header=14
)
df = df_raw.drop(
    ' SOUID', 
    axis=1
)
df['date'] = pd.to_datetime(
    df['    DATE'], 
    format='%Y%m%d'
)
df = df.drop(
    '    DATE', 
    axis=1
)
df.columns = [
    'temperature', 
    'quality', 
    'date'
]
df.set_index(
    'date', 
    inplace=True, 
    drop=True
)
df['temperature'] = df['temperature'] / 10

'''
build and train a model on data after WW2
'''
df_test = df.loc['1945-12-01':]
df_test['year'] = df_test.index.year
df_test['month'] = df_test.index.month
df_test['day'] = df_test.index.day
df_test = df_test.drop('quality', axis=1)

X = df_test.drop('temperature', axis=1)
y= df_test['temperature']

transformer = make_pipeline(ColumnTransformer(transformers=[('cat', 
                                                             PolynomialFeatures(degree=4, 
                                                                           interaction_only= False, 
                                                                           include_bias=False), 
                                                                           ['year', 
                                                                            'month', 
                                                                            'day'])]))

X_t = transformer.fit_transform(X)
m = LinearRegression()
m.fit(X_t, y)
pred = m.predict(X_t)
score = m.score(X_t, y)

pred_df = pd.DataFrame(pred)
pred_df.index = y.index

'''
predict null values
'''
df_null =  df[df['quality'] == 9]
df_null['year'] = df_null.index.year
df_null['month'] = df_null.index.month
df_null['day'] = df_null.index.day
df_null = df_null.drop('quality', axis=1)

X_null = df_null.drop('temperature', axis=1)
X_null_t = transformer.transform(X_null)

y_null = df_null['temperature']
pred_null = m.predict(X_null_t)

pred_df_null = pd.DataFrame(pred_null)
pred_df_null.index = y_null.index

'''
train another module on datas before WW2
'''
df_past =  df.loc[:'1945-04-24']
df_past['year'] = df_past.index.year
df_past['month'] = df_past.index.month
df_past['day'] = df_past.index.day
df_past = df_past.drop('quality', axis=1)

X_past = df_past.drop('temperature', axis=1)
X_past_t = transformer.fit_transform(X_past)

y_past = df_past['temperature']
m_past = LinearRegression()

m_past.fit(X_past_t, y_past)
pred_past = m_past.predict(X_past_t)

pred_df_past = pd.DataFrame(pred_past)
pred_df_past.index = y_past.index

'''
use this second model to predict datas during WW2 and compare them
'''
X_null_t = transformer.transform(X_null)
pred_null_past = m_past.predict(X_null_t)

pred_df_null_past = pd.DataFrame(pred_null_past)
pred_df_null_past.index = y_null.index

'''
merge all datas
'''
merged_df_null = pd.merge(pred_df_null_past, pred_df_null, left_index=True, right_index=True)
merged_df_null['temperature'] = (merged_df_null['0_x'] + merged_df_null['0_y']) / 2
df_war = merged_df_null.drop('0_x', axis=1)
df_war = df_war.drop('0_y', axis=1)

df_war['year'] = df_war.index.year
df_war['month'] = df_war.index.month
df_war['day'] = df_war.index.day

complete_df = pd.concat([df_past, df_war, df_test], join= 'outer')

'''
time series analysis
'''
df_ts = complete_df.drop(['year', 'month', 'day'], axis=1)

df_ts['timestep'] = range(len(df_ts))
X_ts = df_ts[['timestep']]
y_ts = df_ts['temperature']

m_ts = LinearRegression()
m_ts.fit(X_ts, y_ts)

df_ts['trend'] = m_ts.predict(X_ts)

# df_ts[['temperature', 'trend']].plot()
# plt.show()

'''
monthly seasonality
'''
monthly_seasonal_dummies = pd.get_dummies(df_ts.index.month, drop_first=True, prefix='month').set_index(df_ts.index)
df_ts_monthly = df_ts.join(monthly_seasonal_dummies)

X_season_monthly = df_ts_monthly.drop(['temperature', 'trend'], axis = 1)
y_season_monthly = df_ts_monthly['temperature']

m_ts.fit(X_season_monthly, y_season_monthly)
df_ts_monthly['trend_season'] = m_ts.predict(X_season_monthly)

# df_ts_monthly[['temperature', 'trend_season', 'trend']].plot()
# plt.show()

'''
monthly remainder
'''
df_ts_monthly['remainder'] = df_ts_monthly['temperature'] - df_ts_monthly['trend_season']

'''
evaluating forecast
'''
df_full = df_ts_monthly.drop('trend', axis = 1)
df_full['lag1'] = df_full['remainder'].shift(1)
df_full.dropna(inplace=True)

df_full_test = df_full.loc['2022-01-31':]
df_full_train = df_full.drop(pd.date_range(start ='2022-01-31', end= '2023-01-31'))

X_full = df_full_train.drop(['temperature', 
                       'trend_season', 
                       'remainder'], 
                       axis = 1)
y_full = df_full_train['temperature']

m_full = LinearRegression()
m_full.fit(X_full, y_full)

df_full_train['full_pred'] = m_full.predict(X_full)

# df_full_train[['temperature', 
#          'trend_season', 
#          'full_pred']].plot()
# plt.show()

df_full_train['remainder_full_pred'] = df_full_train['temperature'] - df_full_train['full_pred']

# df_full_train[['remainder', 'remainder_full_pred']].plot()
# plt.show()

df_full_train = df_full_train.drop('remainder_full_pred', axis=1)

'''
evaluating training set
'''
ts_split = TimeSeriesSplit(n_splits=5)
time_series_split = ts_split.split(X_full, y_full)
result = cross_val_score(estimator = m_full, 
                         X = X_full, 
                         y = y_full, 
                         cv = time_series_split)

# print(result)

'''
evaluating on test set
'''
df_full_test = df_full_test.drop(['trend_season', 
                                 'remainder', 
                                 'lag1'], 
                                 axis=1)

X_notfull_test = df_full_test.drop('temperature', 
                                axis=1)

df_full_test['trend_season'] = m_ts.predict(X_notfull_test)

# df_full_test[['temperature', 
#               'trend_season']].plot()
# plt.show()

df_full_test['remainder'] = df_full_test['temperature'] - df_full_test['trend_season']
df_full_test['lag1'] = df_full_test['remainder'].shift(1)
df_full_test.loc['2022-01-31', 'lag1'] = df_full_train.loc['2022-01-30', 'remainder']

X_full_test = df_full_test.drop(['temperature', 
                                 'trend_season', 
                                 'remainder'], 
                                 axis = 1)

df_full_test['full_pred'] = m_full.predict(X_full_test)

# print(m_full.score(X_full_test, df_full_test['temperature']))

df_full_train_test = df_full_train[['temperature', 
                                    'trend_season', 
                                    'full_pred']].append(df_full_test[['temperature', 
                                                                       'trend_season', 
                                                                       'full_pred']])

'''
future prediction
'''
df_merged = df_full_train.append(df_full_test)
X_merged = df_merged.drop(['temperature', 
                           'trend_season', 
                           'remainder', 
                           'full_pred'], 
                           axis = 1)
y_merged = df_merged['temperature']

m_merged = LinearRegression()
m_merged.fit(X_merged, y_merged)

timestep = df_merged['timestep'].max() + 1
months = [0] * 11
months[0] = 1
lag = df_merged.loc['2023-01-31', 'remainder']

X_future1 = []
X_future1.append(timestep)
X_future1.extend(months)
X_future1.append(lag)

X_future1 = pd.DataFrame([X_future1], columns = X_merged.columns)

# print(m_merged.predict(X_future1))