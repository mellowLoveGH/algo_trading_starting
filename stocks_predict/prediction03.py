# %%
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

# %%
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

# add week number for the dataframe with date as index
def calculate_week_number(df_data):
    start_weekday = df_data.index[0].weekday() # offset
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

# get data by ticker-name, start-time & end-time
def get_df_data(ticker_name="AAPL", start_time="2022-01-01", end_time="2022-10-09"):
    df_data = yf.download(tickers=ticker_name, start=start_time, end=end_time) 
    real_time_str = "data may late for 15 minutes"
    if ".HK" in ticker_name:
        flag = True        
        try:
            stock_name, current_time, current_price, info, current_volume_info = get_realtime_info(ticker_name, False)
            
            today_date = current_time.strip()[:10]
            today_date = datetime.datetime.strptime(today_date, '%Y-%m-%d')
            #today_date = str(datetime.datetime.now())[:10] #+ ' ' + current_time.split()[1]
            #today_date = datetime.datetime.strptime(today_date, '%Y-%m-%d') #%H:%M
            #print(info, current_price, today_date)

            if today_date == df_data.index[-1]:  
                print("update real time for today")      
                open_price, high_price, low_price = info[1].split()[1], info[2].split()[1], info[3].split()[1]
                df_data.at[df_data.index[-1], "Open"] = float(open_price)
                df_data.at[df_data.index[-1], "High"] = float(high_price)
                df_data.at[df_data.index[-1], "Low"] = float(low_price)
                df_data.at[df_data.index[-1], "Close"] = float(current_price)
            elif today_date > df_data.index[-1]:
                # ['-0.3\t\t-0.3%', 'open\t\tN/A', 'high\t\tN/A', 'low\t\tN/A', 'turnover\t\t79.32M', 'last close\t\t99.30']
                print("add real time for today", today_date)         
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
    df_data['overnight_return'] = (df_data['Open']-df_data['previous_Close'])/df_data['previous_Close']
    MA1, MA2 = 5, 20
    df_data['MA1'] = df_data['Close'].rolling(MA1).mean()
    df_data['MA2'] = df_data['Close'].rolling(MA2).mean()
    
    # add week number
    df_data['weekday'] = df_data.index.weekday
    df_data = calculate_week_number(df_data)   
    return df_data

def weekly_info(df_data, week_num):
    tmp_df = df_data[ df_data['week_num']==week_num ].copy()
    Len = len(tmp_df)
    start_date, end_date = tmp_df.index[0], tmp_df.index[-1]
    open_price, close_price = tmp_df['Open'][0], tmp_df['Close'][-1]

    high_price, high_date = open_price, start_date
    low_price, low_date = open_price, start_date
    i = 0
    while i<Len:
        cur_date = tmp_df.index[i]
        cur_high, cur_low = tmp_df['High'][i], tmp_df['Low'][i]
        if cur_high>=high_price:
            high_price = cur_high
            high_date = cur_date
        if cur_low<=low_price:
            low_price = cur_low
            low_date = cur_date
        i += 1
    
    # rise-fall ratio
    rise_N, fall_N = len( tmp_df[tmp_df['daily_return']>0] ), len( tmp_df[tmp_df['daily_return']<0] )
    # open avg, open std, 
    open_avg, open_std = tmp_df['Open'].mean(), tmp_df['Open'].std()
    # close avg, close std, 
    close_avg, close_std = tmp_df['Close'].mean(), tmp_df['Close'].std()
    # volume avg, volume std
    volume_avg, volume_std = tmp_df['Volume_log'].mean(), tmp_df['Volume_log'].std() # Volume, Volume_log
    # MA
    ma1, ma2 = tmp_df['MA1'].mean(), tmp_df['MA2'].mean()

    it_info = [week_num, start_date, end_date, open_price, close_price, high_price, high_date, low_price, low_date, 
               rise_N, fall_N, open_avg, open_std, close_avg, close_std, volume_avg, volume_std, ma1, ma2]
    
    return it_info

# week num, start date, end date, open price, close price, high price, high date, low price, low date
def generate_weekly_df(df_data):
    data_col = []

    weeknum_list = list( set(list(df_data['week_num'])) )
    weeknum_list = sorted(weeknum_list)
    for i in weeknum_list[1:]:
        wn = i
        it_info = weekly_info(df_data, wn)
        data_col.append( it_info )
    
    col_names = [   
                    'week-num', 'start_date', 'end_date', 'open_price', 'close_price', 
                    'high_price', 'high_date', 'low_price', 'low_date', 
                    'rise_N', 'fall_N',
                    'open_avg', 'open_std',
                    'close_avg', 'close_std',
                    'volume_avg', 'volume_std',
                    'MA1', 'MA2'
                ]
    weekly_df = pd.DataFrame(data_col, columns = col_names)
    # next week features & labels
    weekly_df['nw_open'] = weekly_df['open_price'].shift(-1)
    weekly_df['nw_high'] = weekly_df['high_price'].shift(-1)
    weekly_df['nw_low'] = weekly_df['low_price'].shift(-1)
    weekly_df['nw_close'] = weekly_df['close_price'].shift(-1)
    return weekly_df

def get_datasets(df, features, label):
    X_data, y_data = [], []
    i = 0
    while i < len(df): # 
        row_data = df.iloc[i]
        fg = True
        x_tmp = []
        for col in features:
            v = row_data[col]
            x_tmp.append( v )
            if math.isnan(v):
                fg = False
        y_tmp = row_data[label]
        #print( x_tmp, y_tmp )
        if fg:
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

# predict this week with +1 feature: this week open
def predict_this_week(df_data, features, label="nw_close"):
    X_data, y_data = get_datasets( df_data[:-2].copy(), features, label ) # data until 2 weeks ago
    test_X, test_y = get_row_data(df_data, -2, features, label) # last week data + this week open -> this week close/high/low

    model = train_model(X_data, y_data)
    y_pred = model.predict(X_data) # error analysis
    error_df = error_analyze(y_data, y_pred, False)

    test_X = np.array([test_X])
    test_y_pred = model.predict(test_X)

    test_y_real = test_y
    return test_y_pred, test_y_real, error_df

# predict next week 
def predict_next_week(df_data, features, label="nw_close"):    
    X_data, y_data = get_datasets( df_data[:-1].copy(), features, label ) # data until 2 weeks ago
    test_X, test_y = get_row_data(df_data, -1, features, label) # last week data + this week open -> this week close/high/low

    model = train_model(X_data, y_data)
    y_pred = model.predict(X_data) # error analysis
    error_df = error_analyze(y_data, y_pred, False)

    test_X = np.array([test_X])
    test_y_pred = model.predict(test_X)
    
    test_y_real = test_y
    return test_y_pred, test_y_real, error_df

# %%
stocks_info = [
    ('BABA', '9988.HK', 1),
    ('BIDU', '9888.HK', 1),
    ('JD', '9618.HK', 0.5 * 7.8),
    ('MPNGY', '3690.HK', 0.5 * 7.8),
    ('NTES', '9999.HK', 0.2 * 7.8),
	('LI', '2015.HK', 0.5 * 7.8),
	('XPEV', '9868.HK', 0.5*7.8),	
	('BILI', '9626.HK', 1 * 7.8),
	('TCOM', '9961.HK', 1 * 7.8),
	('YUMC', '9987.HK', 1*7.8),
	('EDU', '9901.HK', 0.1*7.8), 
    ('NIO', '9866.HK', 1 * 7.8),
	('ZTO', '2057.HK', 1*7.8),
    ('BEKE', '2423.HK', 0.5*7.8),
    ('ZH', '2390.HK', 3 * 7.8), 
    ('WB', '9898.HK', 1*7.8),
    ('MNSO', '9896.HK', 0.5*7.8),
    ('ZLAB', '9688.HK', 0.5*7.8),
    ('TENCENT', '0700.HK', 1*7.8),
    ('TME', '1698.HK', 1*7.8),
    ('SMIC', '0981.HK', 1*7.8),
    ('SenseTime', '0020.HK', 1*7.8),
    ('Kuaishou', '1024.HK', 1*7.8),
    ('Xiaomi', '1810.HK', 1*7.8)
]

st, et = "2020-01-01", "2023-07-31"
stock_data = []

for it in stocks_info[:]:# 
    reference_stock, target_stock, ratio = it    
    print("time range:\t", st, "-", et)
    df_data1 = get_df_data(ticker_name=target_stock, start_time=st, end_time=et)
    df_data2 = generate_weekly_df(df_data1)

    stock_name = reference_stock + "_" + target_stock
    # stock prediction
    pred_info = [stock_name]
    features1 = ['open_price', 'close_price', 'high_price', 'low_price', 'rise_N', 'fall_N', 
                'open_avg', 'open_std', 'close_avg', 'close_std', 'volume_avg', 'volume_std', 
                'MA1', 'MA2',
                'nw_open']    
    feature2  = ['open_price', 'close_price', 'high_price', 'low_price', 'rise_N', 'fall_N', 
                'open_avg', 'open_std', 'close_avg', 'close_std', 'volume_avg', 'volume_std',
                'MA1', 'MA2',] 
    for lbl in ["nw_high", "nw_low", "nw_close"]:
        test_y_pred, test_y_real, error_df = predict_this_week(df_data2, features1, lbl)
        this_week_y_pred = round(test_y_pred[0], 2)
        this_week_y_real = round(test_y_real, 2)
        error_mean, error_median = error_df['dif'].mean(), error_df['dif'].median()
        pred_info.append( this_week_y_pred )
        pred_info.append( this_week_y_real )
        pred_info.append( error_mean )
        #pred_info.append( error_median )

    for lbl in ["nw_high", "nw_low", "nw_close"]:
        test_y_pred, test_y_real, error_df = predict_next_week(df_data2, feature2, lbl)
        next_week_y_pred = round(test_y_pred[0], 2)
        next_week_y_real = round(test_y_real, 2)
        error_mean, error_median = error_df['dif'].mean(), error_df['dif'].median()
        pred_info.append( next_week_y_pred )
        #pred_info.append( next_week_y_real )
        pred_info.append( error_mean )
        #pred_info.append( error_median )
    
    stock_data.append( pred_info )

# %%
col_names = [
            "stock-name",
            "cw_high", "high_real", "cw_high_er1", #"cw_high_er2",
            "cw_low", "low_real", "cw_low_er1", #"cw_low_er2",
            "cw_close", "close_real", "cw_close_er1", #"cw_close_er2",
            "nw_high", "nw_high_er1", #"nw_high_er2",
            "nw_low", "nw_low_er1", #"nw_low_er2",
            "nw_close", "nw_close_er1", #"nw_close_er2"
            ]
stock_df = pd.DataFrame( stock_data, columns=col_names )
stock_df

# %%



