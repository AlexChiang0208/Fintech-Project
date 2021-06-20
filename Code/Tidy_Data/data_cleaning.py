import pandas as pd
import numpy as np
import glob

# 抓出所有檔案路徑
path = '/Users/alex_chiang/Documents/Fin_tech/AI基金/基金歷史淨值/'
files = glob.glob(path + "/*.csv")

# 將所有檔案統整到 "data"
data = pd.DataFrame()
for each_file in files:
    df = pd.read_csv(each_file)
    df.rename(columns = {'基金代碼':'Fund_id', '參考日期(淨值)':'Datetime'}, inplace = True)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index(['Fund_id', 'Datetime'])
    df = df.sort_index(level=0)

    fund_id = []
    for i in df.index.get_level_values(0):
        if i not in fund_id:
            fund_id.append(i)

    for j in fund_id:
        df.loc[j]['參考匯率'].fillna(method = 'ffill', inplace=True)

    df['value'] = df['參考淨值'] * df['參考匯率']
    df = df[['value']]
    data = pd.concat([data, df], axis=0)

# 整理統整表
data = data.sort_index()
fund_data = data.unstack(0)['value']
fund_data.columns.name = ''
fund_data = fund_data.applymap(lambda x: np.nan if x <= 0 else x)

# 存檔
address = '/Users/alex_chiang/Documents/Fin_tech/AI基金/'
fund_data.to_pickle(address+'FundData_NTD.pkl')
#%%

import pandas as pd

address = '/Users/alex_chiang/Documents/Fin_tech/AI基金/'
fund = pd.read_pickle(address+'FundData_NTD.pkl')

# 刪除 row 缺失值
def delete_row(percentage):
    '''
    percentage = 0.7, 刪除超過 70% 都是空值的 row
    percentage = 0.1, 刪除超過 10% 都是空值的 row
    '''
    global fund
    percentage  = percentage * 100
    min_count =  int(((100 - percentage) / 100) * fund.shape[1] + 1)
    fund = fund.dropna(axis = 0, thresh = min_count)
    return

# 刪除 columns 缺失值
def delete_column(percentage):
    '''
    percentage = 0.7, 刪除超過 70% 都是空值的 column
    percentage = 0.1, 刪除超過 10% 都是空值的 column
    '''
    global fund
    percentage  = percentage * 100
    min_count =  int(((100 - percentage) / 100) * fund.shape[0] + 1)
    fund = fund.dropna(axis = 1, thresh = min_count)
    return

delete_row(percentage = 0.8)
delete_column(percentage = 0.2)
delete_row(percentage = 0.3)

# 向後填補 20 個
fund = fund.fillna(method='ffill', limit = 20)

# 向前填補 20 個
fund = fund.fillna(method='bfill', limit = 20)

# 刪除還存在缺失值的 columns
fund = fund.dropna(axis=1)

# 檢查是否還有缺失值
fund.isnull().values.any()

'''
原本 [2566 rows x 5048 columns] -> 刪除超過 70% 都是空值的行列
先刪行，再刪列 [1858 rows x 2967 columns]
先刪列，再刪行 [1841 rows x 3230 columns]

刪除超過 70% 都是空值的列，刪除超過 40% 都是空值的行
[1841 rows x 2658 columns]

刪除超過 70% 都是空值的列，刪除超過 15% 都是空值的行
[1841 rows x 2169 columns]

刪除超過 80% 都是空值的列，刪除超過 20% 都是空值的行，刪除超過 30% 都是空值的列
[1779 rows x 2274 columns]
'''

# 存檔
fund.to_pickle(address+'Remove_nan_FundData_NTD.pkl')
fund.to_csv(address+'Remove_nan_FundData_NTD.csv')
fund.to_excel(address+'Remove_nan_FundData_NTD.xlsx')
#%%


ret = ((fund.iloc[-1] - fund.iloc[0]) / fund.iloc[0]).to_frame()


fund = pd.read_pickle(address+'Remove_nan_FundData_NTD.pkl')



a = fund[['100']]


fund_ret = (fund.shift(1) - fund) / fund
fund_ret = fund_ret.dropna()

b = fund_ret[['100']].rolling(100).std()







