import numpy as np 
import matplotlib.pyplot as plt 
#!pip install yfinance
import yfinance as yf # https://pypi.org/project/yfinance/
import math
import random
import seaborn as sns
import datetime
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings(action='ignore')

# get data by ticker-name, start-time & end-time
def get_df_data(ticker_name="AAPL", start_time="2022-01-01", end_time="2022-10-09"):
    df_data = yf.download(tickers=ticker_name, start=start_time, end=end_time) 
    df_data = df_data[ ["Open", "High", "Low", "Close", "Volume"] ]
    df_data['Volume_log'] = np.log2(df_data['Volume'])
    df_data['previous_Close'] = df_data['Close'].shift(1)
    df_data['daily_return'] = (df_data['Close']-df_data['previous_Close'])/df_data['previous_Close']
    df_data['overnight_return'] = (df_data['Open']-df_data['previous_Close'])/df_data['previous_Close']
    MA1, MA2 = 5, 20
    df_data['MA1'] = df_data['Close'].rolling(MA1).mean()
    df_data['MA2'] = df_data['Close'].rolling(MA2).mean()
    df_data['weekday'] = df_data.index.weekday
    return df_data

def common_range(left1, right1, left2, right2):
    if right1<left2 or left1>right2:
        return False, -1, -1
    new_left = max(left1, left2)
    new_right = min(right1, right2)
    return True, round(new_left, 1), round(new_right, 1)

def cycle01(df_data, limit_Len=20):
    i = 0
    while i<len(df_data):
        date1 = df_data.index[i]
        it1 = df_data.iloc[i]
        left1, right1 = it1['Low'], it1['High']
        period_list = []
        j = i + 1
        while j<min(len(df_data), i + limit_Len):
            date2 = df_data.index[j]
            it2 = df_data.iloc[j]
            left2, right2 = it2['Low'], it2['High']
            f, new_left, new_right = common_range(left1, right1, left2, right2)
            if f:
                tmp_list = [date1, left1, right1, date2, left2, right2, new_left, new_right]
                period_list.append( tmp_list )                
            j += 1
        print(  )
        i += 1
    return 


#
def divide_df(df_data, label_name='Close'):
    tmp_df = df_data.copy()
    mx_indx, mn_idx = tmp_df[label_name].idxmax(), tmp_df[label_name].idxmin()
    start_idx, end_idx = tmp_df.index[0], tmp_df.index[-1]
    if mx_indx not in [start_idx, end_idx] and mn_idx not in [start_idx, end_idx]:
        if mx_indx < mn_idx:
            p1 = tmp_df.loc[:mx_indx]
            p2 = tmp_df.loc[mx_indx:mn_idx]
            p3 = tmp_df.loc[mn_idx:]
            return [p1, p2, p3]
        else:
            p1 = tmp_df.loc[:mn_idx]
            p2 = tmp_df.loc[mn_idx:mx_indx]
            p3 = tmp_df.loc[mx_indx:]
            return [p1, p2, p3]
    if mx_indx in [start_idx, end_idx]:
        p1 = tmp_df.loc[:mn_idx]
        p2 = tmp_df.loc[mn_idx:]
        return [p1, p2]
    if mn_idx in [start_idx, end_idx]:
        p1 = tmp_df.loc[:mx_indx]
        p2 = tmp_df.loc[mx_indx:]
    return []

def check_df(df_data, label_name='Close'):
    tmp_df = df_data.copy()
    mx_indx, mn_idx = tmp_df[label_name].idxmax(), tmp_df[label_name].idxmin()
    start_idx, end_idx = tmp_df.index[0], tmp_df.index[-1]
    if mx_indx in [start_idx, end_idx] and mn_idx in [start_idx, end_idx]:
        return False
    return True



st, et = "2022-02-01", "2023-02-28"
stock_code = "9988.HK" # 0700
df_data1 = get_df_data(stock_code, st, et)

label_name='Close'
df_list = [df_data1]



new_list = []
for it in df_list:
    print(len(it), "\t", it.index[0], "\t", it.index[-1])
    print(check_df(it, label_name))
print()

