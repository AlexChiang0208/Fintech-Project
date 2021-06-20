import pandas as pd
import numpy as np
import csv

stock=pd.read_csv("C:\\Users\\user\\PycharmProjects\\strategy\\stock.csv")

bond=pd.read_csv("C:\\Users\\user\\PycharmProjects\\strategy\\bond.csv")

bond_RR2=pd.read_csv("C:\\Users\\user\\PycharmProjects\\strategy\\bond_RR2.csv")

bond_RR3=pd.read_csv("C:\\Users\\user\\PycharmProjects\\strategy\\bond_RR3.csv")

stock_RR3=pd.read_csv("C:\\Users\\user\\PycharmProjects\\strategy\\stock_RR3.csv")

stock_RR4=pd.read_csv("C:\\Users\\user\\PycharmProjects\\strategy\\stock_RR4.csv")

stock_RR5=pd.read_csv("C:\\Users\\user\\PycharmProjects\\strategy\\stock_RR5.csv")

data_all=pd.read_csv("C:\\Users\\user\\PycharmProjects\\strategy\\data.csv")

list_main=data_all.head(0) #取第一列的值
list_main=list(list_main)
# list_1=stock['fund_id']
# list_1=list(list_1)
# list_2=bond['fund_id']
# list_2=list(list_2)
# list_3=bond_RR2['fund_id']
# list_3=list(list_3)
# list_4=bond_RR3['fund_id']
# list_4=list(list_4)
# list_5=stock_RR3['fund_id']
# list_5=list(list_5)
# list_6=stock_RR4['fund_id']
# list_6=list(list_6)
list_7=stock_RR5['fund_id']
list_7=list(list_7)

# data = [i for i in list_main if i in list_1]
# stock_value=data_all[data]
# print(stock_value)
# address = "C:\\Users\\user\\PycharmProjects\\strategy\\"
# stock_value.to_csv(address+'type_stock.csv')

# data = [i for i in list_main if i in list_2]
# stock_value=data_all[data]
# print(stock_value)
# address = "C:\\Users\\user\\PycharmProjects\\strategy\\"
# stock_value.to_csv(address+'type_bond.csv')

# data = [i for i in list_main if i in list_3]
# stock_value=data_all[data]
# print(stock_value)
# address = "C:\\Users\\user\\PycharmProjects\\strategy\\"
# stock_value.to_csv(address+'type_bond_RR2.csv')

# data = [i for i in list_main if i in list_4]
# stock_value=data_all[data]
# print(stock_value)
# address = "C:\\Users\\user\\PycharmProjects\\strategy\\"
# stock_value.to_csv(address+'type_bond_RR3.csv')

# data = [i for i in list_main if i in list_5]
# stock_value=data_all[data]
# print(stock_value)
# address = "C:\\Users\\user\\PycharmProjects\\strategy\\"
# stock_value.to_csv(address+'type_stock_RR3.csv')

# data = [i for i in list_main if i in list_6]
# stock_value=data_all[data]
# print(stock_value)
# address = "C:\\Users\\user\\PycharmProjects\\strategy\\"
# stock_value.to_csv(address+'type_stock_RR4.csv')

data = [i for i in list_main if i in list_7]
stock_value=data_all[data]
print(stock_value)
address = "C:\\Users\\user\\PycharmProjects\\strategy\\"
stock_value.to_csv(address+'type_stock_RR5.csv')