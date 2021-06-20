### 模組化 ###

# 選出的五檔基金計算結果
def Calculate_funds(df, funds):

    global Drawdown, earning_strategy, earning, result
    
    # Benchmark（全部的平均）
    df_benchmark = df.loc['2019-01-04':'2020-12-31']
    df_bench_ret = np.log(df_benchmark/df_benchmark.shift(1))
    df_bench_ret = df_bench_ret.mean(axis=1).to_frame()
    df_bench_ret.rename(columns = {0:'benchmark'}, inplace = True)
    
    # Result
    strategy = df.loc['2019-01-04':'2020-12-31'][funds]
    ret_strategy = np.log(strategy/strategy.shift(1))
    ret_strategy['benchmark'] = df_bench_ret
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

### Feature engineering ###

import numpy as np
import pandas as pd
import glob
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

path = "C:/Users/User/OneDrive/桌面/classified/"
files = sorted(glob.glob(path + '*.csv'))
file_name = [i.split('\\')[-1].split('.csv')[0] for i in files]


for i,j in zip(files, file_name):

    print(j)
    df = pd.read_csv(i, parse_dates=True, index_col='Datetime')
    df = df.loc[:'2019-12-31']
    fund_id = df.columns
    
    df_ML = pd.DataFrame()

    mk_ret = df.pct_change().mean(axis = 1).to_frame().dropna()
    mk_ret = mk_ret.rolling(240).mean()

    for i in fund_id:
        
        fund = df[[i]]
        fund_return  = fund.pct_change().dropna()
  
      # x1 & x2: 算每日報酬率100days平均與標準差  
        return_av = fund_return.rolling(240).mean() # 100天平均每日報酬率
        return_std = fund_return.rolling(240).std()
        return_av.rename(columns = {i:'x_return_av'}, inplace = True)
        return_std.rename(columns = {i:'x_return_std'}, inplace = True)

  
      # x3: 算承擔每單位風險(標準差)可獲得報酬 
        return_std_ratio = (return_av['x_return_av'] / return_std['x_return_std']).to_frame()
        return_std_ratio.rename(columns = {0:'x_return_std_ratio'}, inplace = True)
        
      # x4: 平均報酬率漲跌幅
        return_growth = (return_av - return_av.shift(1)) / abs(return_av.shift(1))
        return_growth.rename(columns = {i:'x_return_growth'}, inplace = True)
          
      # x5: 算基金報酬率相對於市場報酬率的比率
        fund_to_market_return =  (return_av['x_return_av'] / mk_ret[0]).to_frame()
        fund_to_market_return.rename(columns = {0:'x_fund_to_market_return'}, inplace = True)
    
      # x6: x-skew
        skew = fund_return.rolling(240).skew()
        skew.rename(columns = {i:'x_skew'}, inplace = True)

      # x7: x-kurt
        kurt = fund_return.rolling(240).kurt()
        kurt.rename(columns = {i:'x_kurt'}, inplace = True)

      # x8: neg_std
        neg_ret = fund_return.applymap(lambda x: 0 if x > 0 else x)
        neg_std = neg_ret.rolling(240).std()
        neg_std.rename(columns = {i:'x_neg_std'}, inplace = True)
        
      # x9: sortino ratio
        sortino_ratio = (return_av['x_return_av'] / neg_std['x_neg_std']).to_frame()
        sortino_ratio.rename(columns = {0:'x_sortino_ratio'}, inplace = True)
        
      # x10: maximum drawdown
        log_ret = np.log(df[[i]]/(df[[i]].shift(1))).dropna()
        roll_cumsum = log_ret.rolling(min_periods = 1, window = 240).sum()
        roll_max = roll_cumsum.rolling(min_periods = 1, window = 240).max()
        max_drawdown = abs(roll_cumsum - roll_max)
        max_drawdown.rename(columns = {i:'x_max_drawdown'}, inplace = True)
        
      # x11: calmar ratio
        calmar_ratio = (return_av['x_return_av'] / max_drawdown['x_max_drawdown']).to_frame()
        calmar_ratio = calmar_ratio.applymap(lambda x: np.nan if (x == np.Inf or x == -np.Inf) else x)
        calmar_ratio = calmar_ratio.fillna(method='ffill').fillna(method='bfill')
        calmar_ratio.rename(columns = {0:'x_calmar_ratio'}, inplace = True)

        # x12, x13: alpha, beta
        X = sm.add_constant(mk_ret)
        rolling = RollingOLS(endog=return_av, exog=X, window=240)
        alpha = rolling.fit().params.iloc[:,0].to_frame()
        alpha.rename(columns = {'const':'x_alpha'}, inplace = True)
        beta = rolling.fit().params.iloc[:,1].to_frame()
        beta.rename(columns = {0:'x_beta'}, inplace = True)

      # Y: 算100days報酬率
        y_return = (fund.shift(240) - fund)/fund
        y_return = y_return.shift(-240)
        y_return= y_return.dropna()
        y_return.rename(columns = {i:'Y_ret'}, inplace = True)
        

        df_each = pd.concat([return_av, 
                          return_std,
                          neg_std,
                          return_std_ratio, 
                          return_growth,
                          fund_to_market_return,
                          skew,
                          kurt,
                          sortino_ratio,
                          max_drawdown,
                          calmar_ratio,
                          alpha,
                          beta,
                          y_return], axis=1)

        df_each['fund_id'] = i
        df_ML = pd.concat([df_ML, df_each], axis = 0).dropna()

    df_ML.to_pickle("C:/Users/User/OneDrive/桌面/new_classified/" + j + '.pkl')


#%%

### RidgeCV ###

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import xlwings as xw
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

path = "C:/Users/User/OneDrive/桌面/"

files_value = sorted(glob.glob(path + 'classified/*.csv'))
files_ML = sorted(glob.glob(path + 'new_classified/*.pkl'))
file_name = [i.split('\\')[-1].split('.pkl')[0] for i in files_ML]

workbook = xw.Book()

for i,j,k in zip(files_value, files_ML, file_name):
    print(k)
    df = pd.read_csv(i, parse_dates = True, index_col = 'Datetime')
    fund_id = df.columns

    df_XY = pd.read_pickle(j)
    df_XY.sort_index(level = 0, inplace = True)

    df_total_pred = pd.DataFrame()
    scores = []

    #每個基金跑模型: Ridge
    for fund in fund_id:
        df_fund = df_XY[df_XY['fund_id']== fund]

        #train-test split
        X_train = df_fund.loc[:'2019-01-03'].drop(columns = ['Y_ret', 'fund_id'])
        y_train = df_fund.loc[:'2019-01-03'][['Y_ret']]
        X_test = df_fund.loc['2019-01-04'].to_frame().transpose()
        X_test = X_test.drop(columns = ['Y_ret','fund_id'])
        y_test = df_fund.loc['2019-01-04']['Y_ret']



        # 將資料做 PCA
        pca = PCA(n_components = 3)
        pca_trans = pca.fit(X_train)
        X_train_pca = pca_trans.transform(X_train)
        X_train_pca = pd.DataFrame(X_train_pca, index=X_train.index)

    
        X_test_pca = pca_trans.transform(X_test)
        X_test_pca = pd.DataFrame(X_test_pca, index=X_test.index)
        
        # 模型建立
        model = Ridge(alpha=100)
        #model = LinearRegression()
        #model = SVR(kernel='rbf', gamma=0.1)#, cv=,
          # param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                #       "gamma": np.logspace(-2, 2, 5)}, n_jobs = -1, verbose = 2)
       # model = Lasso(alpha = 0.001)
        model.fit(X_train_pca, y_train)
        Train_y_pred= model.predict(X_train_pca)
        Test_y_pred = model.predict(X_test_pca)
        scores.append(model.score(X_train_pca,y_train))
        
        # 選出前五名的基金
        x = model.predict(X_test_pca)[0]
        df_predict = pd.DataFrame({'ret_predict':x, 'fund_id':fund},index=[0], columns=['ret_predict','fund_id'])
        df_predict.set_index('fund_id',inplace = True)
        df_total_pred =pd.concat([df_total_pred, df_predict], axis = 0)
        
    print(np.mean(scores))
    funds = df_total_pred.nlargest(5, 'ret_predict').index   
    # 前五名的基金＝》計算結果
    Calculate_funds(df = df, funds = funds)

    # 寫入Excel  
    save_excel(sheet_name = k)
    
workbook.save(path + 'ML_Ridge100_result0530.xlsx')
workbook.close()
