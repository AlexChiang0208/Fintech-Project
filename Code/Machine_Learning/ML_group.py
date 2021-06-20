### Feature engineering ###

import numpy as np
import pandas as pd
import glob
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

path = "/Users/alex_chiang/Documents/Fin_tech/AI基金/"
files = sorted(glob.glob(path + '/classified/*.csv'))
file_name = [i.split('/')[-1].split('.csv')[0] for i in files]

for i,j in zip(files, file_name):
    
    df = pd.read_csv(i, parse_dates=True, index_col='Datetime')
    df = df.loc[:'2019-12-31']
    fund_id = df.columns
    
    # 計算市場平均報酬
    mkt_ret = df.pct_change().mean(axis = 1).to_frame().dropna()
    mkt_ret = mkt_ret.rolling(240).mean()

    df_ML = pd.DataFrame()
    for i in fund_id:
        ret = df[[i]].pct_change().dropna()
        neg_ret = ret.applymap(lambda x: 0 if x > 0 else x)
        
        # x1: 報酬率的平均數
        X_mean = ret.rolling(240).mean()
        X_mean.rename(columns = {i:'X_mean'}, inplace = True)
        
        # x2: 報酬率的標準差
        X_std = ret.rolling(240).std()
        X_std.rename(columns = {i:'X_std'}, inplace = True)
        
        # x3: 報酬率的負標準差
        X_neg_std = neg_ret.rolling(240).std()
        X_neg_std.rename(columns = {i:'X_neg_std'}, inplace = True)
        
        # x4: 報酬率的偏態係數
        X_skew = ret.rolling(240).skew()
        X_skew.rename(columns = {i:'X_skew'}, inplace = True)
        
        # x5: 報酬率的峰度係數
        X_kurt = ret.rolling(240).kurt()
        X_kurt.rename(columns = {i:'X_kurt'}, inplace = True)
        
        # x6: 平均報酬率漲跌幅
        X_return_growth = (X_mean - X_mean.shift(1)) / abs(X_mean.shift(1))
        X_return_growth.rename(columns = {'X_mean':'X_return_growth'}, inplace = True)
        
        # x7: 基金報酬率相對於市場報酬率的比率
        X_fund_to_market_return = (X_mean['X_mean'] / mkt_ret[0]).to_frame()
        X_fund_to_market_return.rename(columns = {0:'X_fund_to_market_return'}, inplace = True)
        
        # x8: sharpe ratio
        X_sharpe_ratio = (X_mean['X_mean'] / X_std['X_std']).to_frame()
        X_sharpe_ratio.rename(columns = {0:'X_sharpe_ratio'}, inplace = True)
        
        # x9: sortino ratio
        X_sortino_ratio = (X_mean['X_mean'] / X_neg_std['X_neg_std']).to_frame()
        X_sortino_ratio.rename(columns = {0:'X_sortino_ratio'}, inplace = True)
        
        # x10: maximum drawdown
        log_ret = np.log(df[[i]]/df[[i]].shift(1)).dropna()
        roll_cumsum = log_ret.rolling(min_periods = 1, window = 240).sum()
        roll_max = roll_cumsum.rolling(min_periods = 1, window = 240).max()
        X_max_drawdown = abs(roll_cumsum - roll_max)
        X_max_drawdown.rename(columns = {i:'X_max_drawdown'}, inplace = True)
        
        # x11: calmar ratio
        X_calmar_ratio = (X_mean['X_mean'] / X_max_drawdown['X_max_drawdown']).to_frame()
        X_calmar_ratio = X_calmar_ratio.applymap(lambda x: np.nan if (x == np.Inf or x == -np.Inf) else x)
        X_calmar_ratio = X_calmar_ratio.fillna(method='ffill').fillna(method='bfill')
        X_calmar_ratio.rename(columns = {0:'X_calmar_ratio'}, inplace = True)
        
        # x12, x13: alpha, beta
        X = sm.add_constant(mkt_ret)
        rolling = RollingOLS(endog=X_mean, exog=X, window=240)
        X_alpha = rolling.fit().params.iloc[:,0].to_frame()
        X_alpha.rename(columns = {'const':'X_alpha'}, inplace = True)
        X_beta = rolling.fit().params.iloc[:,1].to_frame()
        X_beta.rename(columns = {0:'X_beta'}, inplace = True)
        
        # Y: 240天後報酬率
        Y_ret = (df[[i]].shift(240) - df[[i]]) / df[[i]]
        Y_ret = Y_ret.shift(-240)
        Y_ret.rename(columns = {i:'Y_ret'}, inplace = True)
        
        df_each = pd.concat([X_mean, X_std, X_neg_std, 
                             X_skew, X_kurt, X_return_growth,
                             X_fund_to_market_return, X_sharpe_ratio,
                             X_sortino_ratio, X_max_drawdown,
                             X_calmar_ratio, X_alpha,
                             X_beta, Y_ret], axis=1).dropna() 
        
        df_each['fund_id'] = i
        df_ML = pd.concat([df_ML, df_each], axis = 0)
    
    df_ML.to_csv(path + 'data_ML/ML_' + j + '.csv')   
#%%

### 模組化 ###

# 選出的五檔基金計算結果
def Calculate_funds(df, funds):

    global Drawdown, earning_strategy, earning, result
    
    # Benchmark（全部的平均）
    bench = df.loc['2019-01-04':'2020-12-31']
    bench_ret = np.log(bench/bench.shift(1))
    bench_ret = bench_ret.mean(axis = 1).to_frame()
    bench_ret.rename(columns = {0:'benchmark'}, inplace = True)

    # Result
    strategy = df.loc['2019-01-04':'2020-12-31'][funds]
    ret_strategy = np.log(strategy/strategy.shift(1))
    ret_strategy['benchmark'] = bench_ret
    ret_strategy.fillna(0, inplace = True)
    earning_strategy = ret_strategy.cumsum()
    earning = 100 * (1 + earning_strategy)
    
    # Drawdown
    Drawdown = pd.DataFrame()
    for k in earning_strategy.columns:
        strategy_ret = earning_strategy[[k]]
        li = []
        for i in range(len(strategy_ret)):
            li.append(strategy_ret.iloc[i,0] - strategy_ret.iloc[:i,0].max())
        li[0] = 0
        each_drawdown = pd.DataFrame(li, index=strategy_ret.index)
        each_drawdown.rename(columns={0:k}, inplace = True)
        for j in range(len(each_drawdown)):
            if each_drawdown.iloc[j,0] > 0:
                each_drawdown.iloc[j] = 0
        Drawdown = pd.concat([Drawdown, each_drawdown], axis = 1)
    
    # Calculate
    max_drawdown = abs(Drawdown.min())
    accumulation_return = earning_strategy.iloc[-1]
    accumulation_return.name = None
    annual_return = ret_strategy.mean() * 365
    annual_volatility = ret_strategy.std() * np.sqrt(365)
    neg_annual_volatility = ret_strategy.applymap(lambda x: 0 if x > 0 else x).std() * np.sqrt(365)
    sharpe_ratio = annual_return / annual_volatility
    sortino_ratio = annual_return / neg_annual_volatility
    calmar_ratio = annual_return / max_drawdown
    
    result = pd.DataFrame([accumulation_return, annual_return, 
                           annual_volatility, neg_annual_volatility,
                           max_drawdown, sharpe_ratio,
                           sortino_ratio, calmar_ratio],  
                          index = ['Accumulation Return', 'Annual Return',
                                   'Annual Volatility', 'Negative Annual Volatility',
                                   'Max Drawdown', 'Sharpe Ratio',
                                   'Sortino Ratio', 'Calmar Ratio']).transpose()
    return

# 將圖片及數據寫入Excel
def save_excel(sheet_name):   
    sheet = workbook.sheets.add()
    sheet.name = sheet_name
    
    sheet.range('A1').value = result
    sheet.range('A10').value = earning
    
    ax = earning_strategy.plot(title = sheet_name)
    fig1 = ax.get_figure()
    plt.grid(True)
    sheet.pictures.add(fig1, name='MyPlot1', update=True, 
                       left=sheet.range('K1').left,
                       top=sheet.range('K1').top) 
    
    plt.cla() # 畫新圖前，要先清空畫本
    ax1 = Drawdown['benchmark'].plot.area(stacked=False, color = 'red', alpha=0.45, title = 'Drawdown')
    ax2 = Drawdown.drop(columns = 'benchmark').plot(ax=ax1, linewidth = 1.5)
    fig2 = ax2.get_figure()
    sheet.pictures.add(fig2, name='MyPlot2', update=True, 
                       left=sheet.range('K22').left, 
                       top=sheet.range('K22').top)
    return
#%%

### Ridge ###

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
import xlwings as xw
import glob
from sklearn.metrics import mean_squared_error

path = "/Users/alex_chiang/Documents/Fin_tech/AI基金/"

files_value = sorted(glob.glob(path + 'classified/*.csv'))
files_ML = sorted(glob.glob(path + 'data_ML/*.csv'))
file_name = [i.split('/')[-1].split('.csv')[0] for i in files_ML]

workbook = xw.Book()

for i,j,k in zip(files_value, files_ML, file_name):

    df = pd.read_csv(i, parse_dates = True, index_col = 'Datetime')
    fund_id = df.columns
    
    df_XY = pd.read_csv(j, parse_dates = True, index_col = ['Datetime', 'fund_id'])
    df_XY.sort_index(level = 0, inplace = True)
    
    # 資料切割
    X_train = df_XY.loc[:'2019-01-03'].drop(columns = 'Y_ret')
    y_train = df_XY.loc[:'2019-01-03'][['Y_ret']]
    X_test = df_XY.loc['2019-01-04'].drop(columns = 'Y_ret')
    
    # 將資料做 PCA
    pca = PCA(n_components = 3)
    pca_trans = pca.fit(X_train)
    X_train_pca = pca_trans.transform(X_train)
    X_train_pca = pd.DataFrame(X_train_pca, index=X_train.index)
    
    X_test_pca = pca_trans.transform(X_test)
    X_test_pca = pd.DataFrame(X_test_pca, index=X_test.index)
    
    # 模型建立
    model = Ridge(alpha = 0)
    model.fit(X_train_pca, y_train)
    MSE = mean_squared_error(df_XY.loc['2019-01-04'][['Y_ret']], model.predict(X_test_pca))
    print(k, MSE)

    # 選出前五名的基金
    df_predict = pd.DataFrame(model.predict(X_test_pca), index=X_test.xs('2019-01-04').index)
    funds = df_predict.nlargest(5, 0).index
    
    # 前五名的基金＝》計算結果
    Calculate_funds(df = df, funds = funds)

    # 寫入Excel  
    save_excel(sheet_name = k)
    
workbook.save(path + 'ML_result/ML_Ridge_result.xlsx')
workbook.close()


### 真槍實戰選基金 ###

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

path = "/Users/alex_chiang/Documents/Fin_tech/AI基金/classified/"
MLpath = "/Users/alex_chiang/Documents/Fin_tech/AI基金/data_ML/"

china_equity = pd.read_csv(path + 'china_equity.csv', parse_dates=True, index_col='Datetime')
tech_equity = pd.read_csv(path + 'tech_equity.csv', parse_dates=True, index_col='Datetime')
hybrid_bond = pd.read_csv(path + 'hybrid_bond.csv', parse_dates=True, index_col='Datetime')

ML_china_equity = pd.read_csv(MLpath+'ML_china_equity.csv', parse_dates = True, index_col = ['Datetime', 'fund_id'])
ML_china_equity.sort_index(level = 0, inplace = True)

ML_tech_equity = pd.read_csv(MLpath+'ML_tech_equity.csv', parse_dates = True, index_col = ['Datetime', 'fund_id'])
ML_tech_equity.sort_index(level = 0, inplace = True)

ML_hybrid_bond = pd.read_csv(MLpath+'ML_hybrid_bond.csv', parse_dates = True, index_col = ['Datetime', 'fund_id'])
ML_hybrid_bond.sort_index(level = 0, inplace = True)

def invest_funds(df, df_XY):

    df = df.iloc[-242:]
    fund_id = df.columns
    
    # 計算市場平均報酬
    mkt_ret = df.pct_change().mean(axis = 1).to_frame().dropna()
    mkt_ret = mkt_ret.rolling(240).mean()
    
    df_ML = pd.DataFrame()
    for i in fund_id:
        ret = df[[i]].pct_change().dropna()
        neg_ret = ret.applymap(lambda x: 0 if x > 0 else x)
        
        # x1: 報酬率的平均數
        X_mean = ret.rolling(240).mean()
        X_mean.rename(columns = {i:'X_mean'}, inplace = True)
        
        # x2: 報酬率的標準差
        X_std = ret.rolling(240).std()
        X_std.rename(columns = {i:'X_std'}, inplace = True)
        
        # x3: 報酬率的負標準差
        X_neg_std = neg_ret.rolling(240).std()
        X_neg_std.rename(columns = {i:'X_neg_std'}, inplace = True)
        
        # x4: 報酬率的偏態係數
        X_skew = ret.rolling(240).skew()
        X_skew.rename(columns = {i:'X_skew'}, inplace = True)
        
        # x5: 報酬率的峰度係數
        X_kurt = ret.rolling(240).kurt()
        X_kurt.rename(columns = {i:'X_kurt'}, inplace = True)
        
        # x6: 平均報酬率漲跌幅
        X_return_growth = (X_mean - X_mean.shift(1)) / abs(X_mean.shift(1))
        X_return_growth.rename(columns = {'X_mean':'X_return_growth'}, inplace = True)
        
        # x7: 基金報酬率相對於市場報酬率的比率
        X_fund_to_market_return = (X_mean['X_mean'] / mkt_ret[0]).to_frame()
        X_fund_to_market_return.rename(columns = {0:'X_fund_to_market_return'}, inplace = True)
        
        # x8: sharpe ratio
        X_sharpe_ratio = (X_mean['X_mean'] / X_std['X_std']).to_frame()
        X_sharpe_ratio.rename(columns = {0:'X_sharpe_ratio'}, inplace = True)
        
        # x9: sortino ratio
        X_sortino_ratio = (X_mean['X_mean'] / X_neg_std['X_neg_std']).to_frame()
        X_sortino_ratio.rename(columns = {0:'X_sortino_ratio'}, inplace = True)
        
        # x10: maximum drawdown
        log_ret = np.log(df[[i]]/df[[i]].shift(1)).dropna()
        roll_cumsum = log_ret.rolling(min_periods = 1, window = 240).sum()
        roll_max = roll_cumsum.rolling(min_periods = 1, window = 240).max()
        X_max_drawdown = abs(roll_cumsum - roll_max)
        X_max_drawdown.rename(columns = {i:'X_max_drawdown'}, inplace = True)
        
        # x11: calmar ratio
        X_calmar_ratio = (X_mean['X_mean'] / X_max_drawdown['X_max_drawdown']).to_frame()
        X_calmar_ratio = X_calmar_ratio.applymap(lambda x: np.nan if (x == np.Inf or x == -np.Inf) else x)
        X_calmar_ratio = X_calmar_ratio.fillna(method='ffill').fillna(method='bfill')
        X_calmar_ratio.rename(columns = {0:'X_calmar_ratio'}, inplace = True)
        
        # x12, x13: alpha, beta
        X = sm.add_constant(mkt_ret)
        rolling = RollingOLS(endog=X_mean, exog=X, window=240)
        X_alpha = rolling.fit().params.iloc[:,0].to_frame()
        X_alpha.rename(columns = {'const':'X_alpha'}, inplace = True)
        X_beta = rolling.fit().params.iloc[:,1].to_frame()
        X_beta.rename(columns = {0:'X_beta'}, inplace = True)
    
        df_each = pd.concat([X_mean, X_std, X_neg_std, 
                             X_skew, X_kurt, X_return_growth,
                             X_fund_to_market_return, X_sharpe_ratio,
                             X_sortino_ratio, X_max_drawdown,
                             X_calmar_ratio, X_alpha, X_beta], axis=1).dropna() 
        
        df_each['fund_id'] = i
        df_ML = pd.concat([df_ML, df_each], axis = 0)
    
    df_ML.set_index('fund_id', append = True, inplace = True)
    df_ML.sort_index(level = 0, inplace = True)
    
    # 資料切割
    X_train = df_XY.loc[:'2019-01-03'].drop(columns = 'Y_ret')
    y_train = df_XY.loc[:'2019-01-03'][['Y_ret']]
    X_test = df_ML
    
    # 將資料做 PCA
    pca = PCA(n_components = 3)
    pca_trans = pca.fit(X_train)
    X_train_pca = pca_trans.transform(X_train)
    X_train_pca = pd.DataFrame(X_train_pca, index=X_train.index)
    
    X_test_pca = pca_trans.transform(X_test)
    X_test_pca = pd.DataFrame(X_test_pca, index=X_test.index)
    
    # 模型建立
    model = Ridge(alpha = 0)
    model.fit(X_train_pca, y_train)
    
    # 選出前五名的基金
    df_predict = pd.DataFrame(model.predict(X_test_pca), index=fund_id)
    
    return df_predict.nlargest(5, 0).index
    
# china_equity
invest_funds(df = china_equity, df_XY = ML_china_equity)

# tech_equity
invest_funds(df = tech_equity, df_XY = ML_tech_equity)

# hybrid_bond
invest_funds(df = hybrid_bond, df_XY = ML_hybrid_bond)
#%%

### Lasso ###

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
import xlwings as xw
import glob

path = "/Users/alex_chiang/Documents/Fin_tech/AI基金/"

files_value = sorted(glob.glob(path + 'classified/*.csv'))
files_ML = sorted(glob.glob(path + 'data_ML/*.csv'))
file_name = [i.split('/')[-1].split('.csv')[0] for i in files_ML]

workbook = xw.Book()

for i,j,k in zip(files_value, files_ML, file_name):

    df = pd.read_csv(i, parse_dates = True, index_col = 'Datetime')
    fund_id = df.columns
    
    df_XY = pd.read_csv(j, parse_dates = True, index_col = ['Datetime', 'fund_id'])
    df_XY.sort_index(level = 0, inplace = True)
    
    # 資料切割
    X_train = df_XY.loc[:'2019-01-03'].drop(columns = 'Y_ret')
    y_train = df_XY.loc[:'2019-01-03'][['Y_ret']]
    X_test = df_XY.loc['2019-01-04'].drop(columns = 'Y_ret')
    
    # 將資料做 PCA
    pca = PCA(n_components = 3)
    pca_trans = pca.fit(X_train)
    X_train_pca = pca_trans.transform(X_train)
    X_train_pca = pd.DataFrame(X_train_pca, index=X_train.index)
    
    X_test_pca = pca_trans.transform(X_test)
    X_test_pca = pd.DataFrame(X_test_pca, index=X_test.index)
    
    # 模型建立
    model = Lasso(alpha = 0.001)
    model.fit(X_train_pca, y_train)

    # 選出前五名的基金
    df_predict = pd.DataFrame(model.predict(X_test_pca), index=X_test.xs('2019-01-04').index)
    funds = df_predict.nlargest(5, 0).index
    
    # 前五名的基金＝》計算結果
    Calculate_funds(df = df, funds = funds)

    # 寫入Excel  
    save_excel(sheet_name = k)
    
workbook.save(path + 'ML_result/ML_Lasso_result.xlsx')
workbook.close()
#%%

### SVR ###

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.decomposition import PCA
import xlwings as xw
import glob

path = "/Users/alex_chiang/Documents/Fin_tech/AI基金/"

files_value = sorted(glob.glob(path + 'classified/*.csv'))
files_ML = sorted(glob.glob(path + 'data_ML/*.csv'))
file_name = [i.split('/')[-1].split('.csv')[0] for i in files_ML]

workbook = xw.Book()

for i,j,k in zip(files_value, files_ML, file_name):

    df = pd.read_csv(i, parse_dates = True, index_col = 'Datetime')
    fund_id = df.columns
    
    df_XY = pd.read_csv(j, parse_dates = True, index_col = ['Datetime', 'fund_id'])
    df_XY.sort_index(level = 0, inplace = True)
    
    # 資料切割
    X_train = df_XY.loc[:'2019-01-03'].drop(columns = 'Y_ret')
    y_train = df_XY.loc[:'2019-01-03'][['Y_ret']]
    X_test = df_XY.loc['2019-01-04'].drop(columns = 'Y_ret')
    
    # 將資料做 PCA
    pca = PCA(n_components = 3)
    pca_trans = pca.fit(X_train)
    X_train_pca = pca_trans.transform(X_train)
    X_train_pca = pd.DataFrame(X_train_pca, index=X_train.index)
    
    X_test_pca = pca_trans.transform(X_test)
    X_test_pca = pd.DataFrame(X_test_pca, index=X_test.index)
    
    # 模型建立
    model = SVR()
    model.fit(X_train_pca, y_train)
    
    # 選出前五名的基金
    df_predict = pd.DataFrame(model.predict(X_test_pca), index=X_test.xs('2019-01-04').index)
    funds = df_predict.nlargest(5, 0).index
    
   # 前五名的基金＝》計算結果
    Calculate_funds(df = df, funds = funds)

    # 寫入Excel  
    save_excel(sheet_name = k)
    
workbook.save(path + 'ML_result/ML_SVR_result.xlsx')
workbook.close()
#%%

### DNN ###

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import layers
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import keras
import xlwings as xw
import glob

path = "/Users/alex_chiang/Documents/Fin_tech/AI基金/"

files_value = sorted(glob.glob(path + 'classified/*.csv'))
files_ML = sorted(glob.glob(path + 'data_ML/*.csv'))
file_name = [i.split('/')[-1].split('.csv')[0] for i in files_ML]

files_value = files_value[1:3]
files_ML = files_ML[1:3]
file_name = file_name[1:3]

workbook = xw.Book()

for i,j,k in zip(files_value, files_ML, file_name):

    df = pd.read_csv(i, parse_dates = True, index_col = 'Datetime')
    fund_id = df.columns
    
    df_XY = pd.read_csv(j, parse_dates = True, index_col = ['Datetime', 'fund_id'])
    df_XY.sort_index(level = 0, inplace = True)
    
    # 資料切割
    X_train = df_XY.loc[:'2019-01-03'].drop(columns = 'Y_ret')
    y_train = df_XY.loc[:'2019-01-03'][['Y_ret']]
    X_test = df_XY.loc['2019-01-04'].drop(columns = 'Y_ret')

    # 將資料做『標準化』
    scaler = StandardScaler()
    scaler_trainX = scaler.fit(X_train)
    
    X_train_scaled = scaler_trainX.transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, 
                                            index=X_train.index, 
                                            columns=X_train.columns)
    
    X_test_scaled = scaler_trainX.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, 
                                      index=X_test.index, 
                                      columns=X_test.columns)
    
    # 模型建立
    model = Sequential()
    model.add(layers.Dense(64, activation='relu', input_dim=(13)))
    model.add(layers.Dense(64, activation='relu', input_dim=(13)))
    model.add(layers.Dense(32, activation='relu', input_dim=(13)))
    model.add(layers.Dense(32, activation='relu', input_dim=(13)))
    model.add(layers.Dense(1))
    
    adam = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=adam, loss='mean_squared_error')
    
    history = model.fit(X_train_scaled, y_train, 
              epochs=200, batch_size=60, validation_split=0.2, shuffle=False)
    
    # MSE
    MSE = mean_squared_error(df_XY.loc['2019-01-04'][['Y_ret']], model.predict(X_test_scaled))
    print(k, MSE)
    
    # 選出前五名的基金
    df_predict = pd.DataFrame(model.predict(X_test_scaled), index=X_test.xs('2019-01-04').index)
    funds = df_predict.nlargest(5, 0).index
    
    # 計算結果
    Calculate_funds(df = df, funds = funds)

    # 寫入Excel  
    save_excel(sheet_name = k)
    
workbook.save(path + 'ML_result/ML_DNN_result.xlsx')
workbook.close()


### 真槍實戰選基金 ###

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from keras import layers
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
import keras

path = "/Users/alex_chiang/Documents/Fin_tech/AI基金/classified/"
MLpath = "/Users/alex_chiang/Documents/Fin_tech/AI基金/data_ML/"

asia_equity = pd.read_csv(path + 'asia_equity.csv', parse_dates=True, index_col='Datetime')

ML_asia_equity = pd.read_csv(MLpath+'ML_asia_equity.csv', parse_dates = True, index_col = ['Datetime', 'fund_id'])
ML_asia_equity.sort_index(level = 0, inplace = True)

def invest_funds(df, df_XY):

    df = df.iloc[-242:]
    fund_id = df.columns
    
    # 計算市場平均報酬
    mkt_ret = df.pct_change().mean(axis = 1).to_frame().dropna()
    mkt_ret = mkt_ret.rolling(240).mean()
    
    df_ML = pd.DataFrame()
    for i in fund_id:
        ret = df[[i]].pct_change().dropna()
        neg_ret = ret.applymap(lambda x: 0 if x > 0 else x)
        
        # x1: 報酬率的平均數
        X_mean = ret.rolling(240).mean()
        X_mean.rename(columns = {i:'X_mean'}, inplace = True)
        
        # x2: 報酬率的標準差
        X_std = ret.rolling(240).std()
        X_std.rename(columns = {i:'X_std'}, inplace = True)
        
        # x3: 報酬率的負標準差
        X_neg_std = neg_ret.rolling(240).std()
        X_neg_std.rename(columns = {i:'X_neg_std'}, inplace = True)
        
        # x4: 報酬率的偏態係數
        X_skew = ret.rolling(240).skew()
        X_skew.rename(columns = {i:'X_skew'}, inplace = True)
        
        # x5: 報酬率的峰度係數
        X_kurt = ret.rolling(240).kurt()
        X_kurt.rename(columns = {i:'X_kurt'}, inplace = True)
        
        # x6: 平均報酬率漲跌幅
        X_return_growth = (X_mean - X_mean.shift(1)) / abs(X_mean.shift(1))
        X_return_growth.rename(columns = {'X_mean':'X_return_growth'}, inplace = True)
        
        # x7: 基金報酬率相對於市場報酬率的比率
        X_fund_to_market_return = (X_mean['X_mean'] / mkt_ret[0]).to_frame()
        X_fund_to_market_return.rename(columns = {0:'X_fund_to_market_return'}, inplace = True)
        
        # x8: sharpe ratio
        X_sharpe_ratio = (X_mean['X_mean'] / X_std['X_std']).to_frame()
        X_sharpe_ratio.rename(columns = {0:'X_sharpe_ratio'}, inplace = True)
        
        # x9: sortino ratio
        X_sortino_ratio = (X_mean['X_mean'] / X_neg_std['X_neg_std']).to_frame()
        X_sortino_ratio.rename(columns = {0:'X_sortino_ratio'}, inplace = True)
        
        # x10: maximum drawdown
        log_ret = np.log(df[[i]]/df[[i]].shift(1)).dropna()
        roll_cumsum = log_ret.rolling(min_periods = 1, window = 240).sum()
        roll_max = roll_cumsum.rolling(min_periods = 1, window = 240).max()
        X_max_drawdown = abs(roll_cumsum - roll_max)
        X_max_drawdown.rename(columns = {i:'X_max_drawdown'}, inplace = True)
        
        # x11: calmar ratio
        X_calmar_ratio = (X_mean['X_mean'] / X_max_drawdown['X_max_drawdown']).to_frame()
        X_calmar_ratio = X_calmar_ratio.applymap(lambda x: np.nan if (x == np.Inf or x == -np.Inf) else x)
        X_calmar_ratio = X_calmar_ratio.fillna(method='ffill').fillna(method='bfill')
        X_calmar_ratio.rename(columns = {0:'X_calmar_ratio'}, inplace = True)
        
        # x12, x13: alpha, beta
        X = sm.add_constant(mkt_ret)
        rolling = RollingOLS(endog=X_mean, exog=X, window=240)
        X_alpha = rolling.fit().params.iloc[:,0].to_frame()
        X_alpha.rename(columns = {'const':'X_alpha'}, inplace = True)
        X_beta = rolling.fit().params.iloc[:,1].to_frame()
        X_beta.rename(columns = {0:'X_beta'}, inplace = True)
    
        df_each = pd.concat([X_mean, X_std, X_neg_std, 
                             X_skew, X_kurt, X_return_growth,
                             X_fund_to_market_return, X_sharpe_ratio,
                             X_sortino_ratio, X_max_drawdown,
                             X_calmar_ratio, X_alpha, X_beta], axis=1).dropna() 
        
        df_each['fund_id'] = i
        df_ML = pd.concat([df_ML, df_each], axis = 0)
    
    df_ML.set_index('fund_id', append = True, inplace = True)
    df_ML.sort_index(level = 0, inplace = True)
    
    # 資料切割
    X_train = df_XY.loc[:'2019-01-03'].drop(columns = 'Y_ret')
    y_train = df_XY.loc[:'2019-01-03'][['Y_ret']]
    X_test = df_ML

    # 將資料做『標準化』
    scaler = StandardScaler()
    scaler_trainX = scaler.fit(X_train)
    
    X_train_scaled = scaler_trainX.transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, 
                                            index=X_train.index, 
                                            columns=X_train.columns)
    
    X_test_scaled = scaler_trainX.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, 
                                      index=X_test.index, 
                                      columns=X_test.columns)
    
    # 模型建立
    model = Sequential()
    model.add(layers.Dense(64, activation='relu', input_dim=(13)))
    model.add(layers.Dense(64, activation='relu', input_dim=(13)))
    model.add(layers.Dense(32, activation='relu', input_dim=(13)))
    model.add(layers.Dense(32, activation='relu', input_dim=(13)))
    model.add(layers.Dense(1))
    
    adam = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=adam, loss='mean_squared_error')
    
    history = model.fit(X_train_scaled, y_train, 
              epochs=200, batch_size=60, validation_split=0.2, shuffle=False)
    
    # 選出前五名的基金
    df_predict = pd.DataFrame(model.predict(X_test_scaled), index=X_test.xs('2021-03-31').index)
    return df_predict.nlargest(5, 0).index
    
# asia_equity
invest_funds(df = asia_equity, df_XY = ML_asia_equity)
