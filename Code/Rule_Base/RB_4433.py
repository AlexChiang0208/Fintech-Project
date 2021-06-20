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
    if len(Drawdown.columns) == 1:
        fig2 = ax1.get_figure()
    else:
        ax2 = Drawdown.drop(columns = 'benchmark').plot(ax=ax1, linewidth = 1.5)
        fig2 = ax2.get_figure()
    sheet.pictures.add(fig2, name='MyPlot2', update=True, 
                       left=sheet.range('K22').left, 
                       top=sheet.range('K22').top)
    return

# 時間調整
def time_trans():
    
    global insample
    
    insample = pd.to_datetime(insample)
    while True:
        if insample in data.index:
            break
        else:
            insample = insample + datetime.timedelta(days=1)
    return
#%%

### Rule Base ###
### 4433 ###

import datetime
import glob
import numpy as np
import pandas as pd
import xlwings as xw
import matplotlib.pyplot as plt

path = "/Users/alex_chiang/Documents/Fin_tech/AI基金/"
files = glob.glob(path + '/classified/*.csv')
file_name = [i.split('/')[-1].split('.csv')[0] for i in files]

insample = '2019-01-03'
workbook = xw.Book()

for i,j in zip(files, file_name):
    
    df = pd.read_csv(i, parse_dates=True, index_col='Datetime')
    data = df.copy()
    
    # Time transformer
    time_trans()
    data = data.loc[:insample]
    
    # 選基金策略
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

    # 前五名的基金＝》計算結果
    Calculate_funds(df = df, funds = funds)

    # 寫入Excel  
    save_excel(sheet_name = j)
    
workbook.save(path + 'RB_result/RuleBase_4433_result.xlsx')
workbook.close()
#%%

### 真槍實戰選基金 ###

import numpy as np
import pandas as pd

path = "/Users/alex_chiang/Documents/Fin_tech/AI基金/classified/"

asia_bond = pd.read_csv(path + 'asia_bond.csv', parse_dates=True, index_col='Datetime')
glo_hy_bond  = pd.read_csv(path + 'glo_hy_bond.csv', parse_dates=True, index_col='Datetime')
tw_equity = pd.read_csv(path + 'tw_equity.csv', parse_dates=True, index_col='Datetime')

def invest_funds(df):
    # 選基金策略
    ret = df.pct_change()
    l4 = int(len(df.columns)/4)
    l3 = int(len(df.columns)/3)
    
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
    stock_value = df[filt]
    expected_return = (stock_value.iloc[-240:].pct_change().mean()) * 240
    standard_dev = (stock_value.iloc[-240:].pct_change().std()) * np.sqrt(240)
    sharp = (expected_return - 0.008) / standard_dev

    return sharp.nlargest(5).index

# asia_bond
invest_funds(df = asia_bond)

# glo_hy_bond
invest_funds(df = glo_hy_bond)

# tw_equity
invest_funds(df = tw_equity)
