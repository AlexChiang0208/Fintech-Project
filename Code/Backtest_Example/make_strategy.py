import datetime
import pandas as pd
from pandas_datareader import DataReader

start = datetime.datetime(2019,1,1)
end = datetime.datetime(2021,1,1)

codes=['2330.TW','2317.TW','2303.TW', '2327.TW', '2454.TW']

df = pd.DataFrame()
for m in codes:
    each_stock = DataReader(m, "yahoo", start, end)[['Adj Close']]
    each_stock.rename(columns={'Adj Close':m}, inplace = True)
    df = pd.concat([df, each_stock], axis=1)
#%%

import numpy as np
import matplotlib.pyplot as plt

# 假裝這些資料是基金淨值
# 尚未考慮交易成本！
# 目前沒有動態換基金策略，僅有靜態再平衡（例如：50天自動換一檔基金）
# 策略會執行到最後一天

earning = pd.DataFrame()
hold_days = 50  #幾個交易日後自動再平衡
start_date = '2020-01-02'  #請輸入 DataFrame 中出現的日期作為第一天交易日

# 選基金策略：使用 Rule Base 回傳「一檔」基金的 ID
# 範例：三十天報酬率最大者
strategy = df.pct_change(30).loc[start_date].nlargest(1).index[0]

price = df[strategy].loc[start_date:]
ret = np.log(price/price.shift(1)).dropna().iloc[0:hold_days]
earning = pd.concat([earning, ret], axis=0)

while True:     
    if earning.index[-1] == df.index[-1]:
        break
    else: 
        start_date = earning.index[-1]
        strategy = df.pct_change(hold_days).loc[start_date].nlargest(1).index[0]
        price = df[strategy].loc[start_date:]
        ret = np.log(price/price.shift(1)).dropna().iloc[0:hold_days]
        earning = pd.concat([earning, ret], axis=0)

earning.rename(columns={0:'Return'}, inplace = True)
strategy_ret = earning.cumsum()
strategy_ret.plot()
#%%

# 在 jupyter notebook(lab) 開啟會呈現好看的視覺化圖形
# pip install pyfolio

import pyfolio as pf
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

pf.create_returns_tear_sheet(earning.squeeze())