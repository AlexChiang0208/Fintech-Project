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

df.to_csv('/Users/alex_chiang/Documents/Fin_tech/回測架構試做/data.csv')
