import numpy as np 
import matplotlib.pyplot as plt 
#!pip install yfinance
import yfinance as yf # https://pypi.org/project/yfinance/
import math
import random
import seaborn as sns
import datetime
import pandas as pd
#!pip install sklearn
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings
warnings.filterwarnings(action='ignore')
from bs4 import BeautifulSoup
import requests
import json
import time
import re

def get_realtime_info(stock_code, printing=True):
    URL_link = "https://www.citifirst.com.hk/en/data/json/json_realtimedata/code/"+stock_code
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:80.0) Gecko/20100101 Firefox/80.0'}
    soup = BeautifulSoup(requests.get(URL_link, headers=headers).content, 'html.parser')

    start_index = str(soup).find("{")
    end_index = str(soup).find("}")
    st = str(soup)[start_index:end_index+1]
    json_str = ""
    for ln in st.split(","):
        if "<" not in ln:
            json_str = json_str + ln + ","
    dic = json.loads(json_str[:-1])

    stock_name, current_time, current_price, current_volume_info = dic['code']+".HK", dic['stimeNoformat'], dic['last'], dic['turnover']
    info = []
    dif = round(float(current_price)-float(dic['lastc']), 2)
    ratio = round(dif/float(dic['lastc'])*100, 2)
    info.append( str(dif) + "\t\t" + str(ratio) + "%" )
    info.append( "open\t\t" + dic['open'] )
    info.append( "high\t\t" + dic['high'] )
    info.append( "low\t\t" + dic['low'] )
    info.append( "turnover\t\t" + dic['turnover'] )
    info.append( "last close\t\t" + dic['lastc'] )
    if printing:
        for sub_info in info:
            print( "\t\t", sub_info )
    return stock_name, current_time, current_price, info, current_volume_info

def calculate_week_number(df_data):
    start_weekday = df_data.index[0].weekday()
    start_date = df_data.index[0]
    week_nums = []
    i = 0
    while i<len(df_data.index):
        cur_date = df_data.index[i]
        cur_week_num = ( int((cur_date- start_date).days) + start_weekday ) // 7
        week_nums.append( cur_week_num )
        i += 1
    df_data['week_num'] = week_nums
    return df_data

def get_df_data(ticker_name="AAPL", start_time="2022-01-01", end_time="2022-10-09", MA_type1="Close", MA_type2="Close", MA1=5, MA2=20):
    df_data = yf.download(tickers=ticker_name, start=start_time, end=end_time) 
    real_time_str = "data may late for 15 minutes"
    if ".HK" in ticker_name:
        flag = True        
        try:
            stock_name, current_time, current_price, info, current_volume_info = get_realtime_info(ticker_name, False)            
            today_date = current_time.strip()[:10]
            today_date = datetime.datetime.strptime(today_date, '%Y-%m-%d')
            #print(info, current_price, today_date)

            if today_date == df_data.index[-1]:    
                open_price, high_price, low_price = info[1].split()[1], info[2].split()[1], info[3].split()[1]
                df_data.at[df_data.index[-1], "Open"] = float(open_price)
                df_data.at[df_data.index[-1], "High"] = float(high_price)
                df_data.at[df_data.index[-1], "Low"] = float(low_price)
                df_data.at[df_data.index[-1], "Close"] = float(current_price)
            elif today_date > df_data.index[-1]:
                # ['-0.3\t\t-0.3%', 'open\t\tN/A', 'high\t\tN/A', 'low\t\tN/A', 'turnover\t\t79.32M', 'last close\t\t99.30']
                try:
                    open_price, high_price, low_price = info[1].split()[1], info[2].split()[1], info[3].split()[1]   
                    new_row = [float(open_price), float(high_price), float(low_price), float(current_price), float(current_price), 0]
                except:
                    open_price, high_price, low_price = current_price, current_price, current_price
                    new_row = [float(open_price), float(high_price), float(low_price), float(current_price), float(current_price), 0]
                print(open_price, high_price, low_price)
                col_names = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
                df_data = df_data.append(pd.DataFrame([ new_row ],index=[ today_date ],columns=col_names))
            flag = False
            real_time_str = "real time data"
        except:
            df_data = yf.download(tickers=ticker_name, start=start_time, end=end_time) 
            real_time_str = "data may late for 15 minutes"
    print(ticker_name, ":\t", real_time_str)
    df_data = df_data[ ["Open", "High", "Low", "Close", "Volume"] ]
    df_data['Volume_log'] = np.log2(df_data['Volume'])
    df_data['previous_Close'] = df_data['Close'].shift(1)
    df_data['daily_return'] = (df_data['Close']-df_data['previous_Close'])/df_data['previous_Close']
    #df_data['overnight_return'] = (df_data['Open']-df_data['previous_Close'])/df_data['previous_Close']
    #MA1, MA2 = 5, 20 # for one week and one month
    df_data['MA1'] = df_data[MA_type1].rolling(MA1).mean()
    df_data['MA2'] = df_data[MA_type2].rolling(MA2).mean()
    df_data['weekday'] = df_data.index.weekday
    df_data = calculate_week_number(df_data)   
    return df_data


def return2str(df_data):
    daily_return = list(df_data['daily_return'])
    st = ""
    i = 1
    while i<len(daily_return):
        v = daily_return[i]
        if v > 0:
            st = st + "+1"
        else:
            st = st + "-1"
        i += 1
    return st

def get_col_correlation(col1, col2, df):
    return df[col1].corr(df[col2])

def get_datasets(df, features, label):
    X_data, y_data = [], []
    i = 0
    while i < len(df): # 
        row_data = df.iloc[i]
        x_tmp = []
        for col in features:
            x_tmp.append( row_data[col] )
        y_tmp = row_data[label]
        #print( x_tmp, y_tmp )
        X_data.append( x_tmp )
        y_data.append( y_tmp )
        i += 1
    return X_data, y_data

def get_row_data(df, row_index, features, label):
    row_data = df.iloc[row_index]
    row_x = []
    for col in features:
        row_x.append( row_data[col] )
    row_y = row_data[label]
    return row_x, row_y

from sklearn.linear_model import LinearRegression
def train_model(train_X, train_y, printing=True):
    model = LinearRegression().fit(train_X, train_y)

    r_sq = model.score(train_X, train_y)
    if printing:
        print(f"coefficient of determination: {r_sq}")
        print(f"intercept: {model.intercept_}\tslope: {model.coef_}")
    return model

def basic_info(df):
    print("mean:\t", df['dif'].mean())
    print("std:\t", df['dif'].std())
    print("25%:\t", df['dif'].quantile(0.25))
    print("50%:\t", df['dif'].quantile(0.50))
    print("75%:\t", df['dif'].quantile(0.75))
    return

def error_analyze(train_y, y_pred, printing=True):
    df = pd.DataFrame(columns = ['y_real', 'y_pred'])
    df['y_real'] = train_y
    df['y_pred'] = y_pred
    df['dif'] = (df['y_real'] - df['y_pred'])/df['y_real'] * 100
    df['dif'] = df['dif'].abs()
    if printing:
        basic_info(df)
    return df


    
stocks_info = [
    ('TENCENT', '0700.HK', 1*7.8),
    ('BABA', '9988.HK', 1),
    ('BIDU', '9888.HK', 1),
    ('JD', '9618.HK', 0.5 * 7.8),
    ('MPNGY', '3690.HK', 0.5 * 7.8),
    ('NTES', '9999.HK', 0.2 * 7.8),
	('LI', '2015.HK', 0.5 * 7.8),
	('XPEV', '9868.HK', 0.5*7.8),	
    ('NIO', '9866.HK', 1 * 7.8),
    ('SenseTime', '0020.HK', 1*7.8),
    ('Kuaishou', '1024.HK', 1*7.8),
	('BILI', '9626.HK', 1 * 7.8),
	('TCOM', '9961.HK', 1 * 7.8),
    ('SMIC', '0981.HK', 1*7.8),
    ('Xiaomi', '1810.HK', 1*7.8),
	('YUMC', '9987.HK', 1*7.8),
	('EDU', '9901.HK', 0.1*7.8),
	('ZTO', '2057.HK', 1*7.8),
    ('BEKE', '2423.HK', 0.5*7.8),
    ('ZH', '2390.HK', 3 * 7.8), 
    ('WB', '9898.HK', 1*7.8),
    ('ZLAB', '9688.HK', 0.5*7.8),
    ('MNSO', '9896.HK', 0.5*7.8)
]

data_col = []
for it in stocks_info[:22]:
    #target_stock = '9988.HK'
    target_name, target_stock, _ = it
    st, et = "2022-01-01", "2023-06-30"
    ma1, ma2 = 5, 20

    features = ["MA1", "MA2", "Close", "High", "Low", "Open", "next_Open","Volume_log"]
    label = "next_Close"

    df1 = get_df_data(target_stock, st, et, "Close", "Close", ma1, ma2)
    df1['next_Close'] = df1['Close'].shift(-1)
    df1['next_Open'] = df1['Open'].shift(-1)
    df1

    test_X, test_y = get_row_data(df1, -2, features, label)
    previous_close = test_X[2]
    test_X, test_y

    X_data, y_data = get_datasets(df1[-201:-2], features, label)

    model = train_model(X_data, y_data, printing=True)
    y_pred = model.predict(X_data)
    error_df = error_analyze(y_data, y_pred, True)
    error_mean = error_df['dif'].mean()
    error_median = error_df['dif'].median()
    error_75 = error_df['dif'].quantile(0.75)

    test_X = np.array([test_X])
    test_y_pred = model.predict(test_X)
    print(f"reference:\t{test_X} \t predicted:\t{test_y_pred} \t real:\t{test_y}")
    real_y = test_y
    [pred_y] = test_y_pred

    daily_return = (real_y - previous_close) / previous_close * 100
    data_col.append( [target_name+"_"+target_stock, previous_close, real_y, daily_return, pred_y, error_mean, error_median, error_75] )

col_names = [
            'stock-name', 
            'previous_Close',
            'real_Close',
            'daily_return%',
            'predicted_Close',
            'error_mean%',
            'error_median%',
            'error_75%'
            ]
info_df = pd.DataFrame(data_col, columns=col_names)
info_df['price_change'] = info_df['real_Close'] - info_df['previous_Close']
#info_df['daily_return%'] = (info_df['real_Close'] - info_df['previous_Close'])/info_df['previous_Close'] * 100
round_dic = {'previous_Close':2, 'predicted_Close': 2, 'error_mean%': 2, 'error_median%': 2, 'error_75%':2, 'daily_return%':2, 'price_change':2 }

pd.set_option("display.max_rows", None, "display.max_columns", None)
filename = "prediction_" + str(datetime.datetime.now())[:10] + ".csv"
info_df.round(round_dic).to_csv( 'C:/Users/Admin/Desktop/stocks_analyze_predict/stocks_predict/'+filename)
print( info_df.round(round_dic) )
