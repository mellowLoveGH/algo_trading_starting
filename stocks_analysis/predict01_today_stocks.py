import numpy as np 
import matplotlib.pyplot as plt 
#!pip install yfinance
import yfinance as yf # https://pypi.org/project/yfinance/
import math
import random
import seaborn as sns
import datetime
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings
warnings.filterwarnings(action='ignore')

from bs4 import BeautifulSoup
import requests
import json

import re

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


# get data by ticker-name, start-time & end-time
def get_df_data(ticker_name="AAPL", start_time="2022-01-01", end_time="2022-10-09"):
    df_data = yf.download(tickers=ticker_name, start=start_time, end=end_time) 
    df_data = df_data[ ["Open", "High", "Low", "Close", "Volume"] ]

    df_data['previous_Close'] = df_data['Close'].shift(1)
    df_data['daily_return'] = (df_data['Close']-df_data['previous_Close'])/df_data['previous_Close']
    df_data['overnight_return'] = (df_data['Open']-df_data['previous_Close'])/df_data['previous_Close']
    
    df_data['Volume_log'] = np.log2(df_data['Volume'])
    """
    df_data['previous_Volume'] = df_data['Volume_log'].shift(1)
    df_data['Volume_return'] = (df_data['Volume_log'] - df_data['previous_Volume'])/df_data['previous_Volume']

    df_data['Price_log'] = np.log2(df_data['Close'])
    df_data['previous_Price'] = df_data['Price_log'].shift(1)
    df_data['Price_return'] = (df_data['Price_log'] - df_data['previous_Price'])/df_data['previous_Price'] 
    """   

    MA1, MA2 = 5, 20
    df_data['MA1'] = df_data['Close'].rolling(MA1).mean()
    df_data['MA2'] = df_data['Close'].rolling(MA2).mean()
    df_data['weekday'] = df_data.index.weekday
    df_data = calculate_week_number(df_data)
    return df_data

def merge_stocks(df_data1, df_data2):
    data = []
    i = 0
    while i<len(df_data1):
        target_time = df_data1.index[i]

        j = 0
        while j<len(df_data2):
            ref_time = df_data2.index[j]
            if ref_time>=target_time:
                break
            j += 1
        ref_time = df_data2.index[j-1]
        if ref_time<target_time:
            it1 = df_data1.iloc[i]   
            open1, high1, low1, close1 = it1['Open'], it1['High'], it1['Low'], it1['Close']
            daily_return1 = it1['daily_return']
            volume1 = it1['Volume_log'] # Volume, Volume_log
            #target_MA1, target_MA2 = it1['MA1'], it1['MA2']
            it2 = df_data2.iloc[j-1]
            open2, high2, low2, close2 = it2['Open'], it2['High'], it2['Low'], it2['Close']
            daily_return2 = it2['daily_return']
            volume2 = it2['Volume_log'] # Volume, Volume_log
            #ref_MA1, ref_MA2 = it2['MA1'], it2['MA2']

            tmp_list = [ target_time, open1, high1, low1, close1, daily_return1, volume1, ref_time, open2, high2, low2, close2, daily_return2, volume2 ]
            data.append( tmp_list )
            #print( target_time, ref_time )
        i += 1
    col_names = ['target_time', 'target_open', 'target_high', 'target_low', 'target_close', 'target_return', 'target_volume', 
                'ref_time', 'ref_open', 'ref_high', 'ref_low', 'ref_close', 'ref_return', 'ref_volume']
    df = pd.DataFrame(data, columns = col_names)
    return df

def get_datasets(merged_data, feature_list=['open', 'close', 'high', 'low'], label="close", movement=1):
    data_source = merged_data.copy()
    if movement == 1:
        data_source = data_source[ data_source['ref_return']>0 ]
    elif movement == 0:
        data_source = data_source[ data_source['ref_return']<0 ]

    features = []
    for f in feature_list:
        tmp_list = list( data_source['ref_' + f] )
        features.append( tmp_list )
    # pack features
    packed_features = []
    i = 0
    while i<len(features[0]):
        it = []
        j = 0
        while j<len(features):
            v = features[j][i]
            it.append(v)            
            j += 1
        packed_features.append(it)
        i += 1
    labels = data_source['target_'+label]
    return packed_features, labels

def train_model(train_X, train_y):
    model = LinearRegression().fit(train_X, train_y)

    r_sq = model.score(train_X, train_y)
    print(f"coefficient of determination: {r_sq}")
    print(f"intercept: {model.intercept_}\tslope: {model.coef_}")
    return model

def basic_info(df):
    print("mean:\t", df['dif'].mean())
    print("median:\t", df['dif'].median())
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


# as per recommendation from @freylis, compile once only
CLEANR = re.compile('<.*?>') 
def cleanhtml(raw_html):
    cleantext = re.sub(CLEANR, '', raw_html)
    return cleantext

def get_realtime_info1(stock_code):
    url = 'https://www.hstong.com/quotes/10000-0' + stock_code + '-HK'
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:80.0) Gecko/20100101 Firefox/80.0'}
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.content, "html.parser")
    st = cleanhtml(str(soup))
    lines = []
    for ln in st.split('\n'):
        if len(ln.strip())>0:
            lines.append( ln )
    stock_name, current_time, current_price, info = real_time_info(lines)
    current_time = current_time.strip()
        #
    tmp_list = info_process(info)
    current_volume_info = ""
    for sub_info in tmp_list:
        print( "\t\t", sub_info )
        current_volume_info = sub_info
    return stock_name, current_time, current_price, info, current_volume_info

def get_realtime_info2(stock_code, printing=True):
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


def get_realtime_info(stock_code, printing=True):
    #num = random.randint(0, 20)
    #if num<=6:
        #return get_realtime_info1(stock_code)
    return get_realtime_info2(stock_code, printing)


# http://www.aastocks.com/tc/usq/market/china-concept-stock.aspx
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
    ('NIO', '9866.HK', 1 * 7.8),
	('EDU', '9901.HK', 0.1*7.8), 
	('ZTO', '2057.HK', 1*7.8),
    ('BEKE', '2423.HK', 0.5*7.8),
    ('ZH', '2390.HK', 3 * 7.8), 
    ('WB', '9898.HK', 1*7.8),
    ('MNSO', '9896.HK', 0.5*7.8)
]

OHLC_list = ['open', 'high', 'low', 'close']
daily_prediction = []
daily_real = []

for it in stocks_info[:]: # 
    reference_stock, target_stock, ratio = it
    st, et = "2020-01-01", "2023-04-30"
    print("time range:\t", st, "-", et)
    df_data1 = get_df_data(ticker_name=target_stock, start_time=st, end_time=et)
    print("target stock:\t", target_stock, "\t", len(df_data1))
    df_data2 = get_df_data(ticker_name=reference_stock, start_time=st, end_time=et)
    print("reference stock:\t", reference_stock, "\t", len(df_data2))
        # merge reference-stock & target-stock: target stock (datetime, open, high, low, close), 1-day previous reference stock (datetime, open, high, low, close) 
    merged_data = merge_stocks(df_data1, df_data2)
    print("merged_data:\ttarget_time:\t", merged_data.iloc[0]['target_time'], "\t", merged_data.iloc[-1]['target_time'])
    print("merged_data:\tref_time:\t", merged_data.iloc[0]['ref_time'], "\t", merged_data.iloc[-1]['ref_time'])

    stock_name = reference_stock + "_" + target_stock
    
    points = df_data1.iloc[-1]
    daily_real.append( [stock_name, points['Open'], points['High'], points['Low'], points['Close'], points['previous_Close']] )
    
    info_list = [stock_name]

    for label_name in OHLC_list:
        print(label_name)
        #label_name = "high"
        feature_names = ['open', 'high', 'low', 'close'] # 'open', 'high', 'low', 'close'

        n_days = 60
        movement = 2
        train_X, train_y = get_datasets(merged_data[-n_days:-1], feature_names, label_name, movement)
        train_X = np.array(train_X)
        train_y = np.array(train_y)

        model = train_model(train_X, train_y)

        y_pred = model.predict(train_X)
        df = error_analyze(train_y, y_pred, False)
        error_mean = df['dif'].mean()
        error_median = df['dif'].median()
        error_75 = df['dif'].quantile(0.75)

        #test_X, test_y = get_datasets(merged_data[-1:], feature_names, label_name, 2)
        test_X = [ list(df_data2.iloc[-1])[:4] ]
        test_X = np.array(test_X)
        #test_y = np.array(test_y)

        test_y_pred = model.predict(test_X)
        print(f"reference:\t{test_X} \t predicted:\t{test_y_pred}")
        print()
        info_list.append( test_y_pred[0] )
        info_list.append( error_mean )
        info_list.append( error_median )
    daily_prediction.append( info_list )


###
col_names = [
            'stock-name', 
            'open_predicted', 'open_error1%', 'open_error2%',
            'high_predicted', 'high_error1%', 'high_error2%',
            'low_predicted', 'low_error1%', 'low_error2%',
            'close_predicted', 'close_error1%', 'close_error2%'
            ]
info_df = pd.DataFrame(daily_prediction, columns=col_names)
info_df

col_names = ['stock-name', 'open', 'high', 'low', 'close', 'previous_Close']
real_df = pd.DataFrame(daily_real, columns=col_names)
real_df['daily_return'] = (real_df['close'] - real_df['previous_Close'])/real_df['previous_Close']*100
real_df

# http://www.aastocks.com/tc/usq/market/china-concept-stock.aspx
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
    ('NIO', '9866.HK', 1 * 7.8),
	('EDU', '9901.HK', 0.1*7.8), 
	('ZTO', '2057.HK', 1*7.8),
    ('BEKE', '2423.HK', 0.5*7.8),
    ('ZH', '2390.HK', 3 * 7.8), 
    ('WB', '9898.HK', 1*7.8),
    ('MNSO', '9896.HK', 0.5*7.8)
]

close_list = []
realtime_priceChange = []
realtime_dailyReturn = []
for it in stocks_info[:]:
    ref_code, target_code, ratio = it
    stock_code = target_code[:4] # 9988, 9998, 3690
    stock_name, current_time, current_price, info, current_volume_info = get_realtime_info(stock_code, False)
    price_change, price_dif = info[0].split()
    price_change, price_dif = float(price_change), float(price_dif.replace("%", ""))
    price_change, price_dif
    #print( ref_code, target_code, current_time, current_price, info[0], sep="\t" )
    close_list.append( float(current_price) )
    realtime_priceChange.append( price_change )
    realtime_dailyReturn.append( price_dif )

print()
print()
print_df = info_df[ ['stock-name', 'open_predicted', 'open_error1%', 'open_error2%'] ]
print(print_df)
print_df = info_df[ ['stock-name', 'high_predicted', 'high_error1%', 'high_error2%'] ]
print(print_df)
print_df = info_df[ ['stock-name', 'low_predicted', 'low_error1%', 'low_error2%'] ]
print(print_df)
print_df = info_df[ ['stock-name', 'close_predicted', 'close_error1%', 'close_error2%'] ]
print(print_df)
print()
print()

tmp_df = info_df[ ['stock-name', 'close_predicted', 'close_error1%', 'close_error2%'] ].copy()
tmp_df['close_error3%'] = (tmp_df['close_error1%'] + tmp_df['close_error2%'])/2
tmp_df['close_error%'] = tmp_df['close_error1%']
tmp_df['close_real'] = close_list
tmp_df['pred_real_dif%'] = (tmp_df['close_predicted'] - tmp_df['close_real'])/tmp_df['close_predicted']*100
tmp_df['pred_real_dif%'] = tmp_df['pred_real_dif%'].abs()
tmp_df['Correct'] = tmp_df['close_error1%'] > tmp_df['pred_real_dif%']

tmp_df['price_change'] = realtime_priceChange
tmp_df['previous_Close'] = tmp_df['close_real'] - tmp_df['price_change']

tmp_df['daily_return%'] = realtime_dailyReturn
col_names = [
    'stock-name'
    , 'close_predicted', 'close_error%'
    , 'close_real', 'pred_real_dif%'
    , 'Correct'
    , 'price_change', 'previous_Close', 'daily_return%'
    #,'close_error1%', 'close_error2%', 'close_error3%'
    ]
# , 'close_error1%': 2, 'close_error2%': 2, 'close_error3%': 2
tmp_df = tmp_df[ col_names ]
round_dic = {'close_predicted': 2, 'close_error%': 2, 'close_real': 2, 'pred_real_dif%': 2, 'previous_Close':2, 'daily_return%':2 }
print_df = tmp_df.round(round_dic)
print(print_df)
