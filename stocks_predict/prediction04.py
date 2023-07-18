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

CLEANR = re.compile('<.*?>') 
def cleanhtml(raw_html):
    cleantext = re.sub(CLEANR, '', raw_html)
    return cleantext
def real_time_info(lines):
    stock_name, current_time, current_price, info = lines[9:13]
    stock_name = stock_name.strip()
    current_time = current_time.strip()
    current_price = current_price.strip()
    info = info.strip()
    info = info.replace("华盛通", '')
    info = info.replace("立即下载", '')
    return stock_name, current_time, current_price, info
def info_process(info):
    i1 = info.find("最   高")
    i2 = info.find("今   开")
    i3 = info.find("成交量")
    i4 = info.find("最   低")
    i5 = info.find("昨   收")
    i6 = info.find("总市值")
    i7 = info.find("52周最高")
    tmp_list = []
    tmp_list.append( info[:i1].strip() )
    tmp_list.append( info[i1:i2].strip() + "\t" + info[i4:i5].strip() )
    tmp_list.append( info[i2:i3].strip() + "\t" + info[i5:i6].strip())
    tmp_list.append( info[i3:i4].strip() )
    #tmp_list.append( info[i6:i7].strip() ) # 总市值
    #tmp_list.append( info[i7:].strip() ) # 52周最高
    return tmp_list
def get_realtime_info1(stock_code, printing=True):
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

def get_realtime_info2(stock_code, printing=True):
    URL_link = "https://finance.now.com/stock/?s=0"+str(stock_code[:4])
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:80.0) Gecko/20100101 Firefox/80.0'}
    soup = BeautifulSoup(requests.get(URL_link, headers=headers).content, 'html.parser')

    st = str(soup)
    if len(st)>0:
        current_price = soup.find("span", class_="price upValue").get_text() # current price
        current_price = float(current_price)

        price_change = soup.find("span", class_="change upValue upChange").get_text() # current price
        price_change = float(price_change)

        open_price = soup.find("td", class_="open").get_text() # current price
        open_price = float(open_price)

        prev_price = soup.find("td", class_="prevClose").get_text() # current price
        prev_price = float(prev_price)

        high_low = soup.select("#miniChart") # 
        high_low = str(high_low[0])
        high_low = cleanhtml(high_low)
        high_low = high_low.replace('最低', '')
        it = high_low.split('最高')
        high_price, low_price = float(it[0]), float(it[1])

        current_volume_info = soup.find("td", class_="volume").get_text() # 
        current_volume_info = current_volume_info.replace('萬', '')
        current_volume_info = float(current_volume_info)*10000


        stock_name = soup.find("span", class_="name").get_text() # 
        current_time = soup.find("span", class_="lastUpdate").get_text() # 

        price_dif = round(price_change/prev_price*100, 2)
        info = [str(price_change)+" "+str(price_dif)+"%", "open_price "+str(open_price), "high_price "+str(high_price), "low_price "+str(low_price), "prev_price "+str(prev_price)]
    
        return stock_name, current_time, current_price, info, current_volume_info
    return None


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
    #df_data['weekday'] = df_data.index.weekday
    #df_data = calculate_week_number(df_data)   
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
            target_MA1, target_MA2 = it1['MA1'], it1['MA2']
            it2 = df_data2.iloc[j-1]
            open2, high2, low2, close2 = it2['Open'], it2['High'], it2['Low'], it2['Close']
            daily_return2 = it2['daily_return']
            volume2 = it2['Volume_log'] # Volume, Volume_log
            ref_MA1, ref_MA2 = it2['MA1'], it2['MA2']

            tmp_list = [ target_time, open1, high1, low1, close1, daily_return1, volume1, target_MA1, target_MA2, ref_time, open2, high2, low2, close2, daily_return2, volume2, ref_MA1, ref_MA2 ]
            data.append( tmp_list )
            #print( target_time, ref_time )
        i += 1
    col_names = ['target_time', 'target_open', 'target_high', 'target_low', 'target_close', 'target_return', 'target_volume', 'target_MA1', 'target_MA2',
                'ref_time', 'ref_open', 'ref_high', 'ref_low', 'ref_close', 'ref_return', 'ref_volume', 'ref_MA1', 'ref_MA2']
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

def train_model(train_X, train_y, printing=True):
    model = LinearRegression().fit(train_X, train_y)

    r_sq = model.score(train_X, train_y)
    if printing:
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

def estimate1(st, et):
    stocks_info = [
        ('TENCENT', '0700.HK', 1*7.8),
        ('TME', '1698.HK', 1*7.8),
        ('SMIC', '0981.HK', 1*7.8),
        ('SenseTime', '0020.HK', 1*7.8),
        ('Kuaishou', '1024.HK', 1*7.8),
        ('Xiaomi', '1810.HK', 1*7.8)
    ]

    prevClose_list = []
    open_list = []
    rise10, rise25, rise50, rise75, rise90 = [], [], [], [], []
    fall10, fall25, fall50, fall75, fall90 = [], [], [], [], []

    for it in stocks_info:
        ref_code, target_code, ratio = it
        df_data1 = get_df_data(ticker_name=target_code, start_time=st, end_time=et)    
        prev_close = df_data1.iloc[-1]['previous_Close']
        open_price = df_data1.iloc[-1]['Open']
        day_num = 100
        history_df = df_data1[-day_num:-1].copy()
        rise_df = history_df[history_df['daily_return']>0]
        fall_df = history_df[history_df['daily_return']<0]
        rise10.append( rise_df['daily_return'].quantile(0.1) * 100 )
        rise25.append( rise_df['daily_return'].quantile(0.25) * 100 )
        rise50.append( rise_df['daily_return'].quantile(0.50) * 100 )
        rise75.append( rise_df['daily_return'].quantile(0.75) * 100 )
        rise90.append( rise_df['daily_return'].quantile(0.9) * 100 )

        fall10.append( fall_df['daily_return'].quantile(0.1) * 100 )
        fall25.append( fall_df['daily_return'].quantile(0.25) * 100 )
        fall50.append( fall_df['daily_return'].quantile(0.50) * 100 )
        fall75.append( fall_df['daily_return'].quantile(0.75) * 100 )
        fall90.append( fall_df['daily_return'].quantile(0.9) * 100 )

        prevClose_list.append( prev_close )
        open_list.append( open_price )
    
    stockname_list = []
    close_list = []
    realtime_priceChange = []
    realtime_dailyReturn = []
    realtime_high, realtime_low = [], []

    for it in stocks_info:
        ref_code, target_code, ratio = it
        #print(ref_code, target_code)
        stock_code = target_code[:4] # 9988, 9998, 3690
        stock_name, current_time, current_price, info, current_volume_info = get_realtime_info(stock_code, False)
        price_change, price_dif = info[0].split() 
        price_change, price_dif = float(price_change), float(price_dif.replace("%", ""))
        _, high_price = info[2].split()
        _, low_price = info[3].split()
        #print( ref_code, target_code, current_time, current_price, info[0], sep="\t" )
        stockname_list.append( ref_code+"_"+target_code )
        close_list.append( float(current_price) )
        realtime_priceChange.append( price_change )
        realtime_dailyReturn.append( price_dif )
        realtime_high.append( high_price )
        realtime_low.append( low_price )

    stock_df = pd.DataFrame()
    stock_df['stock_name'] = stockname_list
    stock_df['Close_real'] = close_list
    stock_df['price_change'] = realtime_priceChange
    stock_df['daily_return%'] = realtime_dailyReturn
    stock_df['high_real'] = realtime_high
    stock_df['low_real'] = realtime_low

    stock_df['rise25%'] = rise25
    stock_df['rise50%'] = rise50
    stock_df['rise75%'] = rise75
    stock_df['fall25%'] = fall25
    stock_df['fall50%'] = fall50
    stock_df['fall75%'] = fall75

    round_dic = {'rise25%': 2, 'rise50%': 2, 'rise75%': 2, 'fall25%': 2, 'fall50%':2, 'fall75%':2 }    
    return stock_df.round(round_dic)


def futures_HK():
    URL_link = 'http://www.aastocks.com/sc/stocks/market/bmpfutures.aspx'
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:80.0) Gecko/20100101 Firefox/80.0'}
    soup = BeautifulSoup(requests.get(URL_link, headers=headers).content, 'html.parser')

    soup
    its = soup.findAll("td", class_="bold txt_r cls")
    for it in its:
        num = it.get_text()
        print(num)
    return 



# %%
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
	('EDU', '9901.HK', 0.1*7.8), 
    ('NIO', '9866.HK', 1 * 7.8),
	('ZTO', '2057.HK', 1*7.8),
    ('BEKE', '2423.HK', 0.5*7.8),
    ('ZH', '2390.HK', 3 * 7.8), 
    ('WB', '9898.HK', 1*7.8),
    ('MNSO', '9896.HK', 0.5*7.8),
    ('ZLAB', '9688.HK', 0.5*7.8)
]

st, et = "2020-01-01", "2023-07-31"

OHLC_list = ['open', 'high', 'low', 'close']
daily_prediction = []
daily_real = []

for it in stocks_info[:]:# 
    reference_stock, target_stock, ratio = it    
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
        printing = False
        if printing:
            print(label_name)
        #label_name = "high"
        feature_names = ['open', 'high', 'low', 'close', 'MA1', 'MA2'] # 'open', 'high', 'low', 'close'

        n_days = 60
        movement = 2
        train_X, train_y = get_datasets(merged_data[-n_days:-1], feature_names, label_name, movement)
        train_X = np.array(train_X)
        train_y = np.array(train_y)

        model = train_model(train_X, train_y, printing)

        y_pred = model.predict(train_X)
        df = error_analyze(train_y, y_pred, printing)
        error_mean = df['dif'].mean()
        error_median = df['dif'].median()
        error_75 = df['dif'].quantile(0.75)
        #print( round(error_mean, 2), round(error_median, 2), round(error_75, 2) )

        #test_X, test_y = get_datasets(merged_data[-1:], feature_names, label_name, 2)
        test_X = [ list(df_data2.iloc[-1])[:4]+[ df_data2.iloc[-1]['MA1'], df_data2.iloc[-1]['MA2'] ]  ] #  
        test_X = np.array(test_X)
        #test_y = np.array(test_y)

        test_y_pred = model.predict(test_X)
        if printing:
            print(f"reference:\t{test_X} \t predicted:\t{test_y_pred}")
            print()
        info_list.append( test_y_pred[0] )
        info_list.append( error_mean )
        info_list.append( error_median )
        info_list.append( error_75 )
    daily_prediction.append( info_list )

# %%
###
col_names = [
            'stock-name', 
            'open_predicted', 'open_error1%', 'open_error2%', 'open_error3%',
            'high_predicted', 'high_error1%', 'high_error2%', 'high_error3%',
            'low_predicted', 'low_error1%', 'low_error2%', 'low_error3%',
            'close_predicted', 'close_error1%', 'close_error2%', 'close_error3%'
            ]
info_df = pd.DataFrame(daily_prediction, columns=col_names)
info_df

col_names = ['stock-name', 'open', 'high', 'low', 'close', 'previous_Close']
real_df = pd.DataFrame(daily_real, columns=col_names)
real_df['daily_return'] = (real_df['close'] - real_df['previous_Close'])/real_df['previous_Close']*100
real_df

# http://www.aastocks.com/tc/usq/market/china-concept-stock.aspx
close_list = []
realtime_priceChange = []
realtime_dailyReturn = []
realtime_high, realtime_low = [], []
for it in stocks_info[:]:
    ref_code, target_code, ratio = it
    #print(ref_code, target_code)
    stock_code = target_code[:4] # 9988, 9998, 3690
    stock_name, current_time, current_price, info, current_volume_info = get_realtime_info(stock_code, False)
    #print(stock_name, current_time, current_price, info, current_volume_info)
    price_change, price_dif = info[0].split() 
    price_change, price_dif = float(price_change), float(price_dif.replace("%", ""))
    _, high_price = info[2].split()
    _, low_price = info[3].split()
    #print( ref_code, target_code, current_time, current_price, info[0], sep="\t" )
    close_list.append( float(current_price) )
    realtime_priceChange.append( price_change )
    realtime_dailyReturn.append( price_dif )
    realtime_high.append( high_price )
    realtime_low.append( low_price )

tmp_df = info_df[ ['stock-name', 'close_predicted', 'close_error1%', 'close_error2%', 'close_error3%', 'low_predicted', 'high_predicted'] ].copy()

#tmp_df['close_error3%'] = (tmp_df['close_error1%'] + tmp_df['close_error2%'])/2
tmp_df['close_error%'] = tmp_df['close_error3%']

tmp_df['close_real'] = close_list
tmp_df['pred_real_dif%'] = (tmp_df['close_predicted'] - tmp_df['close_real'])/tmp_df['close_predicted']*100
tmp_df['pred_real_dif%'] = tmp_df['pred_real_dif%'].abs()
tmp_df['Correct'] = tmp_df['close_error%'] >= tmp_df['pred_real_dif%']

tmp_df['price_change'] = realtime_priceChange
tmp_df['previous_Close'] = tmp_df['close_real'] - tmp_df['price_change']

tmp_df['daily_return%'] = realtime_dailyReturn

tmp_df['low_real'] = realtime_low
tmp_df['high_real'] = realtime_high

col_names = [
    'stock-name'
    , 'previous_Close', 'price_change'
    , 'close_real', 'daily_return%'
    , 'close_predicted', 'close_error%', 'pred_real_dif%'
    , 'Correct'
    #,'close_error1%', 'close_error2%', 'close_error3%'
    , 'low_predicted', 'low_real'
    , 'high_predicted', 'high_real'
    ]
# , 'close_error1%': 2, 'close_error2%': 2, 'close_error3%': 2
tmp_df = tmp_df[ col_names ]
round_dic = {'close_predicted': 2, 'close_error%': 2, 'close_real': 2, 'pred_real_dif%': 2, 'previous_Close':2, 'daily_return%':2, 'low_predicted':2, 'high_predicted':2 }

filename = "prediction02_" + str(datetime.datetime.now())[:10] + ".csv"
tmp_df.round(round_dic).to_csv( 'C:/Users/Admin/Desktop/stocks_analyze_predict/stocks_predict/'+filename)
tmp_df.round(round_dic) 

# %%
estimate1(st, et)

# %%
futures_HK()

# %%
#(24.3*300+21.65*300+20.0*400)/1000=21.785 # (24.5-21.785)*1000

(11.30-10.72)/10.72*100
(1.96-1.89)/1.96*100


