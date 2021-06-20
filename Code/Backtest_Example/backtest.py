import datetime
import numpy as np
import pandas as pd

path = '/Users/alex_chiang/Documents/Fin_tech/AI基金/data_classified/type_stock.csv'
df = pd.read_csv(path, parse_dates=True, index_col='Datetime')

# 尚未考慮交易成本！
# 尚未「制式化、模組化」策略寫法

hold_days = 50    # day of holding a fund
start_date = '2019-04-01'
end_date = '2020-11-30'
data = df
window = 60    # day of rolling drawdown

# Time transformer
start_date = pd.to_datetime(start_date)
while True:
    if start_date in data.index:
        break
    else:
        start_date = start_date + datetime.timedelta(days=1)

end_date = pd.to_datetime(end_date)
while True:
    if end_date in data.index:
        break
    else:
        end_date = end_date - datetime.timedelta(days=1)

# 選基金策略：使用 Rule Base 回傳「一檔」基金的 ID
# 範例：三十天報酬率最大者
strategy = data.pct_change(30).loc[start_date].nlargest(1).index[0]


### Start backtest ###
earning = pd.DataFrame()
price = data[strategy].loc[start_date:]
ret = np.log(price/price.shift(1)).dropna().iloc[0:hold_days]
earning = pd.concat([earning, ret], axis=0)

while True:     
    if end_date in earning.index:
        earning = earning.loc[:end_date]
        break
    else: 
        start_date = earning.index[-1]
        strategy = data.pct_change(hold_days).loc[start_date].nlargest(1).index[0]
        price = data[strategy].loc[start_date:]
        ret = np.log(price/price.shift(1)).dropna().iloc[0:hold_days]
        earning = pd.concat([earning, ret], axis=0)

earning.rename(columns={0:'Return'}, inplace = True)
strategy_ret = earning.cumsum()

# Result Calculation
annual_return = np.round(earning.mean()[0] * 365, 4)
annual_volatility = np.round(earning.std()[0] * np.sqrt(365), 4)
sharpe_ratio = np.round(annual_return / annual_volatility, 4)
accumulation_return = np.round(strategy_ret.iloc[-1,0], 4)

# Drawdown
li = []
for i in range(len(strategy_ret)):
    li.append(strategy_ret.iloc[i,0] - strategy_ret.iloc[:i,0].max())

li[0] = 0
max_drawdown = pd.DataFrame(li, index=strategy_ret.index)
max_drawdown.rename(columns={0:'Drawdown'}, inplace = True)
for i in range(len(max_drawdown)):
    if max_drawdown.iloc[i,0] > 0:
        max_drawdown.iloc[i] = 0

# Rolling Drawdown
li_2 = []
for i in range(len(strategy_ret)):
    if window > i:
        li_2.append(strategy_ret.iloc[i,0] - strategy_ret.iloc[:i,0].max())
    else:
        li_2.append(strategy_ret.iloc[i,0] - strategy_ret.iloc[i-window:i,0].max())

li_2[0] = 0
rolling_max_drawdown = pd.DataFrame(li_2, index=strategy_ret.index)
rolling_max_drawdown.rename(columns={0:str(window)+'days_Drawdown'}, inplace = True)
for i in range(len(rolling_max_drawdown)):
    if rolling_max_drawdown.iloc[i,0] > 0:
        rolling_max_drawdown.iloc[i] = 0

# Output
print('Annual Return: ', annual_return)
print('Annual Volatility: ', annual_volatility)
print('Sharpe Ratio: ', sharpe_ratio)
print('Accumulation Return: ', accumulation_return)
strategy_ret.plot()
max_drawdown.plot.area(stacked=False, color = 'red')
rolling_max_drawdown.plot.area(stacked=False, color = 'orangered')
#%%

# 在 jupyter notebook(lab) 開啟會呈現好看的視覺化圖形
# pip install pyfolio

import pyfolio as pf
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

pf.create_returns_tear_sheet(earning.squeeze())
