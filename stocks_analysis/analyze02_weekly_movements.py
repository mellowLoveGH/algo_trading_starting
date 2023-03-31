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

from bs4 import BeautifulSoup
import requests
import json

import re

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
    if ".HK" in ticker_name:
        stock_name, current_time, current_price, info, current_volume_info = get_realtime_info(ticker_name, False)
        open_price, high_price, low_price = info[1].split()[1], info[2].split()[1], info[3].split()[1]
        df_data.at[df_data.index[-1], "Open"] = float(open_price)
        df_data.at[df_data.index[-1], "High"] = float(high_price)
        df_data.at[df_data.index[-1], "Low"] = float(low_price)
        df_data.at[df_data.index[-1], "Close"] = float(current_price)

    df_data = df_data[ ["Open", "High", "Low", "Close", "Volume"] ]
    df_data['Volume_log'] = np.log2(df_data['Volume'])
    df_data['previous_Close'] = df_data['Close'].shift(1)
    df_data['daily_return'] = (df_data['Close']-df_data['previous_Close'])/df_data['previous_Close']
    df_data['overnight_return'] = (df_data['Open']-df_data['previous_Close'])/df_data['previous_Close']
    MA1, MA2 = 5, 20
    df_data['MA1'] = df_data['Close'].rolling(MA1).mean()
    df_data['MA2'] = df_data['Close'].rolling(MA2).mean()
    df_data['weekday'] = df_data.index.weekday
    df_data = calculate_week_number(df_data)    
    return df_data

def weekly_return1(df_data, wn):
    week_df = df_data[ df_data['week_num']==wn ]
    week_open = week_df['Open'][0]
    week_close = week_df['Close'][-1]
    r = (week_close-week_open)/week_open * 100
    #print( week_open, week_close, r )
    return round(r, 2)

def weekly_return2(df_data, wn):
    prev_week_df = df_data[ df_data['week_num']==wn-1 ]
    cur_week_df = df_data[ df_data['week_num']==wn ]
    prev_week_close = prev_week_df['Close'][-1]
    cur_week_close = cur_week_df['Close'][-1]
    r = (cur_week_close-prev_week_close)/prev_week_close * 100
    #print( prev_week_close, cur_week_close, r )
    return round(r, 2)

def addlabels(x_list, y_list, x_offset=0.5, y_offset=3):
    for i in range(len(x_list)):
        v = y_list[i]
        if v>0:
            plt.text(x_list[i]+x_offset,y_list[i]+y_offset,v)
        else:
            plt.text(x_list[i]+x_offset,y_list[i]-y_offset,v)
    return 
    
def get_xticks(x_label):
    tmp_label = []
    for it in x_label:
        st = str(it)[:10]
        tmp_label.append( st )
    return tmp_label


### daily returns
stocks_info = [
    ('BABA', '9988.HK', 1), # ('BABA', 'BABA', 1),
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
    ('HTHT', '1179.HK', 0.1*7.8),
    #('TENCENT', '0700.HK', 1*7.8)
    ('TME', '1698.HK', 0.5*7.8),	
    ('ZH', '2390.HK', 3 * 7.8),    
    ('QFIN', '3660.HK', 0.5*7.8),
    ('TUYA', '2391.HK', 1*7.8),
    ('ZLAB', '9688.HK', 0.5*7.8),
    ('ATHM', '2518.HK', 0.5*7.8),
    ('KC', '3896.HK', 0.5*7.8),
    ('WB', '9898.HK', 1*7.8),
    ('MNSO', '9896.HK', 0.5*7.8),
    ('GDS', '9698.HK', 1)
]

for it in stocks_info[:1]:
    st, et = "2021-02-01", "2023-03-31"
    stock_code = "9988.HK" # 0700.HK, 9988.HK， 9888.HK, 9618.HK, 3690.HK, 9999.HK, 9961.HK, 2015.HK, 9866.HK, 
    ref_code, stock_code, _ = it
    df_data1 = get_df_data(stock_code, st, et)
    x_list, y_list = [], []
    for wn in range(1, max(df_data1['week_num'])+1):
        r = weekly_return2(df_data1, wn)
        x_list.append(wn)
        y_list.append(r)

    plt.figure(figsize=(20, 6))
    plt.bar(x_list, y_list , label="weekly return")  # Plot the chart
    addlabels(x_list, y_list, -0.8, 1)
    plt.title(ref_code+"-"+stock_code + " weekly return")
    plt.legend()
    plt.show()  # display