import datetime
import numpy as np
import pandas as pd

path = "/Users/alex_chiang/Documents/Fin_tech/AI基金/data_classified/type_stock_RR5.csv"
df = pd.read_csv(path, parse_dates=True, index_col='Datetime')
df

#, parse_dates=True, index_col='Datetime'
# 假裝這些資料是基金淨值
# 尚未考慮交易成本！
# 尚未「制式化、模組化」策略寫法

hold_days = 50    # day of holding a fund
start_date = '2017-06-01'
end_date = '2021-03-31'
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


# 選基金策略（範例）：使用 Rule Base 回傳「一檔」基金的 ID
#strategy = data.pct_change(30).loc[start_date].nlargest(1).index[0]


# 選基金策略（4433改）：
'''
1 策略邏輯 --> 從Start date開始算起
2 len(data.columns) 這樣才是基金長度，原本算成日期的長度了
3 選取的地方小錯，已改好。如：[i for i in data1 if i in rule1]
4 Sharpe Ratio 那段，我幫你改了簡單一點的寫法，因為我有點看不懂
5 最終結果是要取出『一檔基金的名稱』，改成.index[0]就可以取出來了
補：太長的程式碼，建議定義需調整的參數一個名稱。如：l4 = int(len(data.columns)/4)
補：我抽離5年報酬這個條件，不然回測資料太少（從第五年後才可以初步購買基金）
補：因為需要三年資料才能購買，起始時間要設定有資料的至少後三年
'''
# Debug部分我在程式碼後面寫 #!
data1=data.pct_change()

# 前 1/3, 1/4 的長度
l4 = int(len(data.columns)/4) #! data.columns才是指基金的程度
l3 = int(len(data.columns)/3)

#一年績效排名在前1/4
rule1= data1.loc[:start_date].iloc[-252:].mean().nlargest(l4).index.tolist()[0:l4] 
#! 試著看懂 loc[:start_date].iloc[-252:] 
#! 每一次的選基金策略，都要讓他知道，從哪個起始時間開始算起（Start date會隨著一次次迴圈不斷增加）

#兩年、三年期及自今年以來基金績效排名在同類型前四分之一者
rule2_1=data1.loc[:start_date].iloc[-504:].mean().nlargest(l4).index.tolist()[0:l4]
rule2_2=data1.loc[:start_date].iloc[-756:].mean().nlargest(l4).index.tolist()[0:l4]
rule2_4=data1.loc[:start_date].iloc[-80:].mean().nlargest(l4).index.tolist()[0:l4]

#六個月績效排名在同類型前三分之一者
rule3=data1.loc[:start_date].iloc[-120:].mean().nlargest(l3).index.tolist()[0:l3]

#三個月績效排名在同類型前三分之一者
rule4=data1.loc[:start_date].iloc[-60:].mean().nlargest(l3).index.tolist()[0:l3]

data2 = [i for i in data1 if i in rule1] #! 我改了一些，執行一次就可以知道哪裡寫錯
data2 = [i for i in data2 if i in rule2_1]
data2 = [i for i in data2 if i in rule2_2]
data2 = [i for i in data2 if i in rule2_4]
data2 = [i for i in data2 if i in rule3]
data2 = [i for i in data2 if i in rule4]
stock_value=data.loc[:start_date][data2] #! 這邊一樣記得 loc[:start_date]

##再用sharp ratio挑出最佳的基金
expected_return = (stock_value.iloc[-252:].pct_change().mean())*252 #! 略改寫法
standard_dev = (stock_value.iloc[-252:].pct_change().std())*np.sqrt(252)
sharp=(expected_return-0.008)/standard_dev

strategy=sharp.nlargest(1).index[0] #! 只需要取出基金代碼 string的格式
print(strategy)

'''
原本寫的策略
##先用四四三三法則挑符合的基金
data1=data.pct_change()  #算報酬率

#一年績效排名在前1/4
rule1= data1.tail(252).mean().nlargest(int(len(data)/4)).index.tolist()[0:int(len(data)/4)]

#兩年、三年、五年期及自今年以來基金績效排名在同類型前四分之一者
rule2_1=data1.tail(504).mean().nlargest(int(len(data)/4)).index.tolist()[0:int(len(data)/4)]

rule2_2=data1.tail(756).mean().nlargest(int(len(data)/4)).index.tolist()[0:int(len(data)/4)]

rule2_3=data1.tail(1260).mean().nlargest(int(len(data)/4)).index.tolist()[0:int(len(data)/4)]

rule2_4=data1.tail(80).mean().nlargest(int(len(data)/4)).index.tolist()[0:int(len(data)/4)]

#六個月績效排名在同類型前三分之一者
rule3=data1.tail(120).mean().nlargest(int(len(data)/3)).index.tolist()[0:int(len(data)/3)]
#三個月績效排名在同類型前三分之一者
rule4=data1.tail(60).mean().nlargest(int(len(data)/3)).index.tolist()[0:int(len(data)/3)]

data2 = [i for i in rule1 if i in rule2_1]
data2 = [i for i in data if i in rule2_2]
data2 = [i for i in data if i in rule2_3]
data2 = [i for i in data if i in rule2_4]
data2 = [i for i in data if i in rule3]
data2 = [i for i in data if i in rule4]
stock_value=data[data2]

##再用sharp ratio挑出最佳的基金
expected_return = stock_value.resample('Y').last()[:-1].pct_change().mean()

standard_dev = stock_value.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250))

sharp=(expected_return-0.008)/standard_dev

strategy=sharp.nlargest(1)
print(strategy)
'''

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
# !pip install pyfolio

import pyfolio as pf
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

pf.create_returns_tear_sheet(earning.squeeze())
