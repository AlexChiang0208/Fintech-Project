### Feature engineering ###

import numpy as np
import pandas as pd

path1 = "/Users/alex_chiang/Documents/Fin_tech/AI基金/classified/asia_bond.csv"
df = pd.read_csv(path1, parse_dates=True, index_col='Datetime')
df = df.loc[:'2019-12-31']

path2 = '/Users/alex_chiang/Documents/Fin_tech/AI基金/data_ML/asia_bond/'
fund_id = df.columns
for i in fund_id:
    ret = df[[i]].pct_change().dropna()
    
    X_mean = ret.rolling(240).mean()
    X_mean.rename(columns = {i:'X_mean'}, inplace = True)
    
    X_std = ret.rolling(240).std()
    X_std.rename(columns = {i:'X_std'}, inplace = True)
    
    X_skew = ret.rolling(240).skew()
    X_skew.rename(columns = {i:'X_skew'}, inplace = True)
    
    X_kurt = ret.rolling(240).kurt()
    X_kurt.rename(columns = {i:'X_kurt'}, inplace = True)
    
    exp_ret = (df[[i]].shift(240) - df[[i]]) / df[[i]]
    std_dev = ret.rolling(240).std() * np.sqrt(240)
    Y_sharpe = exp_ret / std_dev
    Y_sharpe = Y_sharpe.shift(-240)
    Y_sharpe.rename(columns = {i:'Y_sharpe'}, inplace = True)
    
    df_ML = pd.concat([X_mean, X_std, X_skew, X_kurt, Y_sharpe], axis=1).dropna() 
    df_ML.to_csv(path2 + 'asia_bond_' + i + '.csv')
#%%

### LinearRegression ###

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

path1 = "/Users/alex_chiang/Documents/Fin_tech/AI基金/classified/asia_bond.csv"
path2 = '/Users/alex_chiang/Documents/Fin_tech/AI基金/data_ML/asia_bond/'

df = pd.read_csv(path1, parse_dates=True, index_col='Datetime')
fund_id = df.columns

df_predict = pd.DataFrame()
for i in fund_id:

    df_XY = pd.read_csv(path2 + 'asia_bond_' + fund_id[2] + '.csv', parse_dates=True, index_col='Datetime')
     
    # Train Test Split 
    X_train = df_XY.iloc[:-10, :4]
    y_train = df_XY.iloc[:-10, 4:]
    X_test = df_XY.iloc[-10:, :4]
    y_test = df_XY.iloc[-10:, 4:]
    
    # Modeling
    model = Lasso(alpha = 0.00001)
    #model = Ridge()
    #model = LinearRegression()
    model.fit(X_train, y_train)
    
    model.predict(X_test)
    
    # Result
    y_test_pred = model.predict(X_test)[0,0]
    df_each = pd.DataFrame([y_test_pred, y_test.iloc[0,0]], 
                            columns = [i],
                            index = ['Y_predict', 'Y_True']).transpose()
    
    df_predict = pd.concat([df_predict, df_each], axis=0)

# 選出前五名的基金
nice_fund = df_predict.nlargest(5, 'Y_predict').index

# 畫出 in-sample, out-sample
df_price = df.loc['2019-01-04':, nice_fund]

df_ret = np.log(df_price/df_price.shift(1))
df_ret.fillna(0, inplace = True)
df_ret = df_ret.cumsum()

# 增加對照組（前 1/2 的平均）
num = int(len(df.columns) / 2)
df_all = df.loc['2019-01-04':]
benchmark_id = ((df_all.iloc[-1] - df_all.iloc[0]) / df_all.iloc[0]).nlargest(num).index

df_benchmark = df.loc['2019-01-04':, benchmark_id]
df_benchmark_ret = np.log(df_benchmark/df_benchmark.shift(1))
df_benchmark_ret.fillna(0, inplace = True)
df_benchmark_ret = df_benchmark_ret.cumsum()
df_benchmark_ret = df_benchmark_ret.mean(axis = 1).to_frame()
df_benchmark_ret.rename(columns = {0:'benchmark'}, inplace = True)

df_result = pd.concat([df_ret, df_benchmark_ret], axis = 1)
df_result.plot()   
#%%

### DNN ###

import numpy as np
import pandas as pd
import keras
from keras import layers
from keras.models import Sequential

path1 = "/Users/alex_chiang/Documents/Fin_tech/AI基金/classified/asia_bond.csv"
path2 = '/Users/alex_chiang/Documents/Fin_tech/AI基金/data_ML/asia_bond/'

df = pd.read_csv(path1, parse_dates=True, index_col='Datetime')
fund_id = df.columns

df_predict = pd.DataFrame()
for i in fund_id:

    df_XY = pd.read_csv(path2 + 'asia_bond_' + i + '.csv', parse_dates=True, index_col='Datetime')
       
    # Train Test Split 
    X_train = df_XY.iloc[:-1, :4]
    y_train = df_XY.iloc[:-1, 4:]
    X_test = df_XY.iloc[-1:, :4]
    y_test = df_XY.iloc[-1:, 4:]
    
    # Modeling
    model = Sequential()
    model.add(layers.Dense(64, activation='relu',input_dim=(4)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    
    # Change an optimizers' learning rate
    adam = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=adam, loss='mean_squared_error')
    model.fit(X_train, y_train, 
              epochs=150, batch_size=30, validation_split=0.15, shuffle=False)
    
    # Result
    y_test_pred = model.predict(X_test)[0,0]
    df_each = pd.DataFrame([y_test_pred, y_test.iloc[0,0]], 
                            columns = [i],
                            index = ['Y_predict', 'Y_True']).transpose()
    
    df_predict = pd.concat([df_predict, df_each], axis=0)
    
# 選出前五名的基金
nice_fund = df_predict.nlargest(5, 'Y_predict').index

# 畫出 in-sample, out-sample
df_price = df.loc['2019-01-04':, nice_fund]

df_ret = np.log(df_price/df_price.shift(1))
df_ret.fillna(0, inplace = True)
df_ret = df_ret.cumsum()

# 增加對照組（前 1/2 的平均）
num = int(len(df.columns) / 2)
df_all = df.loc['2019-01-04':]
benchmark_id = ((df_all.iloc[-1] - df_all.iloc[0]) / df_all.iloc[0]).nlargest(num).index

df_benchmark = df.loc['2019-01-04':, benchmark_id]
df_benchmark_ret = np.log(df_benchmark/df_benchmark.shift(1))
df_benchmark_ret.fillna(0, inplace = True)
df_benchmark_ret = df_benchmark_ret.cumsum()
df_benchmark_ret = df_benchmark_ret.mean(axis = 1).to_frame()
df_benchmark_ret.rename(columns = {0:'benchmark'}, inplace = True)

df_result = pd.concat([df_ret, df_benchmark_ret], axis = 1)
df_result.plot()   
