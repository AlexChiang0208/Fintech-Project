### 原版 ###

import datetime
import numpy as np
import pandas as pd

path = "/Users/alex_chiang/Documents/Fin_tech/AI基金/data_classified/type_stock.csv"
df = pd.read_csv(path, parse_dates=True, index_col='Datetime')

### 策略製作

insample = '2017-01-01'
data = df.copy() # get funds from past net-value

# Time transformer
insample = pd.to_datetime(insample)
while True:
    if insample in data.index:
        break
    else:
        insample = insample + datetime.timedelta(days=1)

data = data.loc[:insample - datetime.timedelta(days=1)]


'''
2014/1/1 ~ 2016/12/31 使用三年資料做出策略指標，挑選最多五檔基金
2017/1/1 ~ 2018/12/31 In-sample（其實定義上都是Out-sample啦！）
2019/1/1 ~ 2021/03/31 Out-sample
'''

## 選基金策略（範例1）：30日報酬率前五大
funds_1 = data.pct_change(30).iloc[-1].nlargest(5).index


## 選基金策略（範例2）：4433改
ret = data.pct_change()

# 前 1/3, 1/4 的長度
l4 = int(len(data.columns)/4)
l3 = int(len(data.columns)/3)

# 一年績效排名在前1/4
rule1 = ret.iloc[-240:].mean().nlargest(l4).index

# 兩年、三年期及自今年以來基金績效排名在同類型前四分之一者
rule2_1 = ret.iloc[-480:].mean().nlargest(l4).index
rule2_2 = ret.iloc[-720:].mean().nlargest(l4).index
rule2_3 = ret.iloc[-100:].mean().nlargest(l4).index

# 六個月績效排名在同類型前三分之一者
rule3 = ret.iloc[-120:].mean().nlargest(l3).index

# 三個月績效排名在同類型前三分之一者
rule4 = ret.iloc[-60:].mean().nlargest(l3).index

# 篩選出符合條件的基金
filt = [i for i in ret if i in rule1]
filt = [i for i in filt if i in rule2_1]
filt = [i for i in filt if i in rule2_2]
filt = [i for i in filt if i in rule2_3]
filt = [i for i in filt if i in rule3]
filt = [i for i in filt if i in rule4]
len(filt)

# 最後用 sharp ratio 挑出最佳的基金
stock_value = data[filt]
expected_return = (stock_value.iloc[-240:].pct_change().mean()) * 240
standard_dev = (stock_value.iloc[-240:].pct_change().std()) * np.sqrt(240)
sharp = (expected_return - 0.008) / standard_dev

# 得到策略基金
funds_2 = sharp.nlargest(5).index
#%%

### 策略績效

def performance(funds, title):
    # Calculate
    strategy = df.loc[insample:][funds]
    ret_strategy = np.log(strategy/strategy.shift(1))
    earning_strategy = ret_strategy.cumsum()
    earning_strategy.fillna(0, inplace = True)
    accumulation_return = earning_strategy.iloc[-1]
    annual_return = ret_strategy.mean() * 365
    annual_volatility = ret_strategy.std() * np.sqrt(365)
    sharpe_ratio = annual_return / annual_volatility
    
    # Benchmark
    num = int(len(df.columns) / 2)
    total = (df.loc[insample:].iloc[-1] - df.loc[insample:].iloc[0]) / df.loc[insample:].iloc[0]
    benchmark_id = total.nlargest(num).index
    df_benchmark = df.loc[insample:][benchmark_id]
    df_benchmark_ret = np.log(df_benchmark/df_benchmark.shift(1))
    df_benchmark_ret.fillna(0, inplace = True)
    df_benchmark_ret = df_benchmark_ret.cumsum()
    df_benchmark_ret = df_benchmark_ret.mean(axis = 1).to_frame()
    df_benchmark_ret.rename(columns = {0:'benchmark'}, inplace = True)

    # Output
    df_result = pd.concat([earning_strategy, df_benchmark_ret], axis = 1)
    print('Annual Return: \n', annual_return, '\n')
    print('Annual Volatility: \n', annual_volatility, '\n')
    print('Sharpe Ratio: \n', sharpe_ratio, '\n')
    print('Accumulation Return: \n', accumulation_return)
    df_result.plot(title = title)
    return

performance(funds = funds_1, title = 'strategy_1')
performance(funds = funds_2, title = 'strategy_2')
#%%

### 整合到 Excel 檔中

import datetime
import glob
import numpy as np
import pandas as pd
import xlwings as xw
import matplotlib.pyplot as plt

path = "/Users/alex_chiang/Documents/Fin_tech/AI基金/"
files = glob.glob(path + '/classified/*.csv')
file_name = [i.split('/')[-1].split('.csv')[0] for i in files]

workbook = xw.Book()

for i,j in zip(files, file_name):
    df = pd.read_csv(i, parse_dates=True, index_col='Datetime')
    
    insample = '2017-01-01'
    data = df.copy() # get funds from past net-value
    
    # Time transformer
    insample = pd.to_datetime(insample)
    while True:
        if insample in data.index:
            break
        else:
            insample = insample + datetime.timedelta(days=1)
    
    data = data.loc[:insample - datetime.timedelta(days=1)]
    
    ## 選基金策略：4433改
    ret = data.pct_change()
    l4 = int(len(data.columns)/4)
    l3 = int(len(data.columns)/3)
    
    rule1 = ret.iloc[-240:].mean().nlargest(l4).index
    rule2_1 = ret.iloc[-480:].mean().nlargest(l4).index
    rule2_2 = ret.iloc[-720:].mean().nlargest(l4).index
    rule2_3 = ret.iloc[-100:].mean().nlargest(l4).index
    rule3 = ret.iloc[-120:].mean().nlargest(l3).index
    rule4 = ret.iloc[-60:].mean().nlargest(l3).index
    
    # 篩選出符合條件的基金
    filt = [i for i in ret if i in rule1]
    filt = [i for i in filt if i in rule2_1]
    filt = [i for i in filt if i in rule2_2]
    filt = [i for i in filt if i in rule2_3]
    filt = [i for i in filt if i in rule3]
    filt = [i for i in filt if i in rule4]
    
    # 最後用 sharp ratio 挑出最佳的基金
    stock_value = data[filt]
    expected_return = (stock_value.iloc[-240:].pct_change().mean()) * 240
    standard_dev = (stock_value.iloc[-240:].pct_change().std()) * np.sqrt(240)
    sharp = (expected_return - 0.008) / standard_dev
    
    # 得到策略基金
    funds = sharp.nlargest(5).index
    
    # Calculate
    strategy = df.loc[insample:][funds]
    ret_strategy = np.log(strategy/strategy.shift(1))
    earning_strategy = ret_strategy.cumsum()
    earning_strategy.fillna(0, inplace = True)
    accumulation_return = earning_strategy.iloc[-1]
    annual_return = ret_strategy.mean() * 365
    annual_volatility = ret_strategy.std() * np.sqrt(365)
    sharpe_ratio = annual_return / annual_volatility
    
    # Benchmark
    num = int(len(df.columns) / 2)
    total = (df.loc[insample:].iloc[-1] - df.loc[insample:].iloc[0]) / df.loc[insample:].iloc[0]
    benchmark_id = total.nlargest(num).index
    df_benchmark = df.loc[insample:][benchmark_id]
    df_benchmark_ret = np.log(df_benchmark/df_benchmark.shift(1))
    df_benchmark_ret.fillna(0, inplace = True)
    df_benchmark_ret = df_benchmark_ret.cumsum()
    df_benchmark_ret = df_benchmark_ret.mean(axis = 1).to_frame()
    df_benchmark_ret.rename(columns = {0:'benchmark'}, inplace = True)
    
    # Output
    df_result = pd.concat([earning_strategy, df_benchmark_ret], axis = 1)
    sheet = workbook.sheets.add()
    sheet.name = j
    
    sheet.range('A1').value = 'Annual Return'
    sheet.range('A2').value = annual_return
    
    sheet.range('A8').value = 'Annual Volatility'
    sheet.range('A9').value = annual_volatility
    
    sheet.range('A15').value = 'Sharpe Ratio'
    sheet.range('A16').value = sharpe_ratio
    
    sheet.range('A23').value = 'Accumulation Return'
    sheet.range('A24').value = accumulation_return
    
    ax = df_result.plot(title = j)
    fig = ax.get_figure() 
    sheet.pictures.add(fig, name='MyPlot', update=True, left=sheet.range('D1').left) 
 
workbook.save(path + 'RuleBase_4433_result.xlsx')
workbook.close()
