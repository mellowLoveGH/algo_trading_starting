import datetime
import math
import random
from bs4 import BeautifulSoup
import requests
import json
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
#!pip install yfinance
import yfinance as yf # https://pypi.org/project/yfinance/
import warnings
warnings.filterwarnings(action='ignore')





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


# get data by ticker-name, start-time & end-time
def get_df_data(ticker_name="AAPL", start_time="2022-01-01", end_time="2022-10-09"):
    df_data = yf.download(tickers=ticker_name, start=start_time, end=end_time) 
    if ".HK" in ticker_name:
        stock_name, current_time, current_price, info, current_volume_info = get_realtime_info(ticker_name, False)
        today_date = current_time.strip()[:10]
        today_date = datetime.datetime.strptime(today_date, '%Y-%m-%d')
        if today_date == df_data.index[-1]:  
          print("update real time for today")      
          open_price, high_price, low_price = info[1].split()[1], info[2].split()[1], info[3].split()[1]
          df_data.at[df_data.index[-1], "Open"] = float(open_price)
          df_data.at[df_data.index[-1], "High"] = float(high_price)
          df_data.at[df_data.index[-1], "Low"] = float(low_price)
          df_data.at[df_data.index[-1], "Close"] = float(current_price)
        elif today_date > df_data.index[-1]:
          print("add real time for today")
          open_price, high_price, low_price = info[1].split()[1], info[2].split()[1], info[3].split()[1]
          new_row = [float(open_price), float(high_price), float(low_price), float(current_price), float(current_price), 0]
          col_names = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
          df_data = df_data.append(pd.DataFrame([ new_row ],index=[ today_date ],columns=col_names))

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


df_data = get_df_data('9988.HK', "2021-02-01", "2023-03-31")
df_data
