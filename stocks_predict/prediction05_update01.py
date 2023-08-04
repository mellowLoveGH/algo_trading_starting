# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


import math
import random
import datetime
import time
import warnings
warnings.filterwarnings(action='ignore')



# https://www.citifirst.com.hk/tc/stock/code/9988
from bs4 import BeautifulSoup
import requests
import json
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
import yfinance as yf # https://pypi.org/project/yfinance/
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
    #print(ticker_name, ":\t", real_time_str)
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
        st = str(it)[2:10]
        tmp_label.append( st )
    return tmp_label

def draw_trend(trend_df, ref_code, stock_code, show_list=[1, 4, 5], ma1=6, ma2=27, n_days1=20, n_days2=2):
    x_list, y_list = trend_df.index, trend_df['Close'].round(2)
    y_list1, y_list2 = trend_df['Low'], trend_df['High']
    x_list = list( range(len(y_list)) )
    x_label = list(trend_df.index)
    z_list1, z_list2 = trend_df['MA1'], trend_df['MA2']

    plt.figure(figsize=(20, 6))
    if 1 in show_list:
        plt.plot(x_list, y_list, label="Close")  # Plot the chart
        #addlabels(x_list, y_list, -0.8, 1)
    if 2 in show_list:
        plt.plot(x_list, y_list1, label="Low")  # Plot the chart
    if 3 in show_list:
        plt.plot(x_list, y_list2, label="High")  # Plot the chart
    if 4 in show_list:
        plt.plot(x_list, z_list1 , label="MA1: "+str(ma1))  # Plot the chart
    if 5 in show_list:    
        plt.plot(x_list, z_list2 , label="MA2: "+str(ma2))  # Plot the chart

    # bollinger band
    trend_df['TP'] = (trend_df['Close'] + trend_df['High'] + trend_df['Low']) / 3 # typical price
    trend_df['TP_MA'] = trend_df['TP'].rolling(n_days1).mean()
    trend_df['TP_SD'] = trend_df['TP'].rolling(n_days1).std()
    trend_df['UP'] = trend_df['TP_MA'] + n_days2 * trend_df['TP_SD']
    trend_df['DN'] = trend_df['TP_MA'] - n_days2 * trend_df['TP_SD']
    bollinger_up = trend_df['UP']
    bollinger_dn = trend_df['DN']
    plt.plot(x_list, bollinger_up , label="B-UP: "+str(n_days1) + ", " + str(n_days2))
    plt.plot(x_list, bollinger_dn , label="B-DN: "+str(n_days1) + ", " + str(n_days2))


    plt.title(ref_code+"-"+stock_code)
    plt.xticks(x_list, get_xticks(x_label), rotation=90)
    plt.legend()
    plt.show()  # display
    return 

def draw_dailyReturn(daily_df, ref_code, stock_code):
    daily_df['daily_return'] = daily_df['daily_return']*100
    daily_df['daily_return'] = daily_df['daily_return'].round(2)
    x_list, y_list = list(daily_df.index), list(daily_df['daily_return'])
    x_list = list( range(len(y_list)) )
    x_label = list(daily_df.index)

    plt.figure(figsize=(20, 6))
    plt.bar(x_list, y_list , label="daily return")  # Plot the chart
    addlabels(x_list, y_list, 0.05, 0.05)

    plt.title(ref_code+"-"+stock_code)
    plt.xticks(x_list, get_xticks(x_label), rotation=90)
    plt.legend()
    plt.show()  # display
    return

def draw_weeklyReturn(weekly_df, ref_code, stock_code):
    x_list, y_list = [], []
    for wn in range(1, max(weekly_df['week_num'])+1):
        r = weekly_return2(weekly_df, wn)
        x_list.append(wn)
        y_list.append(r)

    plt.figure(figsize=(20, 6))
    plt.bar(x_list, y_list , label="weekly return")  # Plot the chart
    addlabels(x_list, y_list, -0.8, 1)
    plt.title(ref_code+"-"+stock_code + " weekly return")
    plt.legend()
    plt.show()  # display
    return 

def current_info(df_data):
    current_open = df_data.iloc[-1]['Open']
    current_high = df_data.iloc[-1]['High']
    current_low = df_data.iloc[-1]['Low']
    current_close = df_data.iloc[-1]['Close']
    current_return = df_data.iloc[-1]['daily_return']
    return 

# %%
def buy_sell_info(trend_df):
    x_list, y_list = trend_df.index, trend_df['Close'].round(2)
    y_list1, y_list2 = trend_df['Low'], trend_df['High']
    x_list = list( range(len(y_list)) )
    x_label = list(trend_df.index)
    z_list1, z_list2 = trend_df['MA1'].round(2), trend_df['MA2'].round(2)
    assert len(x_list)==len(y_list)==len(x_label)==len(z_list1)==len(z_list2)

    buy_signal, sell_signal = [], []
    for idx in range( 0, len(x_list) ):
        xv = x_list[idx]
        yv = y_list[idx]
        lv = x_label[idx]
        z1, z2 = z_list1[idx], z_list2[idx]
        current_dif = z1 - z2
        cross_flag = False
        if idx>0:
            previous_dif = z_list1[idx-1] - z_list2[idx-1]
            cross_flag = (current_dif * previous_dif)<=0
        msg = ""
        if cross_flag:
            if previous_dif < current_dif:
                # buy_signal
                msg = str(lv) + "\tbuy_signal:\tclose value: " + str(yv) + "\tMA1: " + str(z1) + "\tMA2: " + str(z2)
                #print(msg)
                it = [lv, yv, z1, z2, msg]
                if len(buy_signal)==len(sell_signal):
                    buy_signal.append( it )
            elif current_dif < previous_dif:
                # sell_signal
                msg = str(lv) + "\tsell_signal:\tclose value: " + str(yv) + "\tMA1: " + str(z1) + "\tMA2: " + str(z2)
                #print(msg)
                it = [lv, yv, z1, z2, msg]
                if len(buy_signal)==len(sell_signal)+1:
                    sell_signal.append( it )
        ###
        if len(buy_signal) > len(sell_signal) and idx==len(x_list)-1:
            msg = str(lv) + "\tsell_signal:\tclose value: " + str(yv) + "\tMA1: " + str(z1) + "\tMA2: " + str(z2)
            it = [lv, yv, z1, z2, msg]
            sell_signal.append( it )
        previous_dif = current_dif
    #print(len(buy_signal), len(sell_signal), len(buy_signal)==len(sell_signal))
    assert len(buy_signal)==len(sell_signal)


    trade_data = []
    for i in range( 0, len(buy_signal) ):
        it1, it2 = buy_signal[i], sell_signal[i]
        td = it1[:-1] + it2[:-1] # buy-date, buy-close, buy-MA1, buy-MA2, sell-date, sell-close, sell-MA1, sell-MA2, 
        trade_data.append( td )
        """
        msg1, msg2 = it1[-1], it2[-1]
        print(msg1)
        print(msg2)
        v1, v2 = it1[1], it2[1]
        rtn = round((v2-v1)/v1*100, 2)
        print("return: ", rtn)
        """

    col_names = [
        "buy-date", "buy-close", "buy-MA1", "buy-MA2", "sell-date", "sell-close", "sell-MA1", "sell-MA2"
    ]
    trade_df = pd.DataFrame(trade_data, columns=col_names)
    trade_df['rtn'] = (trade_df['sell-close']-trade_df['buy-close'])/trade_df['buy-close']*100
    trade_df['rtn'] = trade_df['rtn'].round(2)
    trade_df

    c1, c2 = len(trade_df[ trade_df['rtn']>0 ]), len(trade_df[ trade_df['rtn']<0 ])
    c1, c2
    win_loss_ratio = c1/(c1+c2)
    win_loss_ratio
    return trade_df, win_loss_ratio

###
df_collection = {}
def get_df_data02(stock_code, st, et, ma1=6, ma2=27):
    if stock_code not in df_collection.keys():
        df_data = get_df_data(stock_code, st, et, "Close", "Close", ma1, ma2)
        df_collection[stock_code] = df_data[ ["Open", "High", "Low", "Close", "Volume"] ].copy()
        return df_data
    else:
        df_data = df_collection[stock_code].copy()
        #df_data = df_data[ ["Open", "High", "Low", "Close", "Volume"] ]
        df_data['Volume_log'] = np.log2(df_data['Volume'])
        df_data['previous_Close'] = df_data['Close'].shift(1)
        df_data['daily_return'] = (df_data['Close']-df_data['previous_Close'])/df_data['previous_Close']
        #df_data['overnight_return'] = (df_data['Open']-df_data['previous_Close'])/df_data['previous_Close']
        #MA1, MA2 = 5, 20 # for one week and one month
        df_data['MA1'] = df_data['Close'].rolling(ma1).mean()
        df_data['MA2'] = df_data['Close'].rolling(ma2).mean()
        return df_data

def correlation_score(it_df1, it_df2, component='Close'):
    dates1 = it_df1.index
    dates2 = it_df2.index
    dates3 = []
    for d in dates1:
        if d in dates2:
            dates3.append( d )
    filter_df = pd.DataFrame()
    filter_df['stock 1'] = it_df1[component].loc[ dates3 ]
    filter_df['stock 2'] = it_df2[component].loc[ dates3 ]
    
    score = filter_df['stock 1'].corr(filter_df['stock 2'])
    return round(score*100, 2)



def relative_values(y_list, offset_v=0):
    tmp_list = []
    base_value = 1
    for r in y_list:
        base_value = base_value * (1+r)
        tmp_list.append( base_value*100+offset_v )
    return tmp_list

def basic_correlations(stocks_info, st, et, cpn='Close', mode='HK'):
    stocks_US_df = pd.DataFrame() # daily_real, columns=col_names
    stocks_HK_df = pd.DataFrame() # daily_real, columns=col_names

    i = 0
    while i<len(stocks_info):
        name1, stock_code1, _ = stocks_info[i]        
        it_df = get_df_data(stock_code1, st, et)
        if ".HK" in stock_code1:
            stocks_HK_df[name1] = it_df[cpn]
        else:
            stocks_US_df[name1] = it_df[cpn]
        i += 1

    stocks_HK_df.corr().to_csv('C:/Users/Admin/Desktop/stocks_analyze_predict/stocks_analysis/stocks_corr01.csv')

    if mode=='HK':
        return stocks_HK_df.corr()


    stocks_US_df.corr().to_csv('C:/Users/Admin/Desktop/stocks_analyze_predict/stocks_analysis/stocks_corr02.csv')

    dates_US = stocks_US_df.index
    dates_HK = stocks_HK_df.index

    dates = []
    for d in dates_US:
        if d in dates_HK:
            dates.append( d )

    df1 = stocks_US_df.loc[ dates ]
    df2 = stocks_HK_df.loc[ dates ]

    stocks_df = pd.merge(df1, df2, left_index=True, right_index=True)
    stocks_df.corr().to_csv('C:/Users/Admin/Desktop/stocks_analyze_predict/stocks_analysis/stocks_corr03.csv')
    return stocks_df.corr()

def draw_lines(x_list, x_label, y1, y2, name1, name2):
    plt.figure(figsize=(20, 6))
    plt.plot(x_list, y1 , label=name1)  # Plot the chart
    plt.plot(x_list, y2 , label=name2)  # Plot the chart

    plt.title("relative values")
    plt.xticks(x_list, get_xticks(x_label), rotation=90)
    plt.grid(True)
    plt.legend()
    plt.show()  # display
    return 

def draw_bars(x_list, x_label, z_gap):
    plt.figure(figsize=(20, 6))
    plt.bar(x_list, z_gap , label="difference")  # Plot the chart
    #addlabels(x_list, z_gap, 0.05, 1.5)
    plt.plot(x_list, [z_gap[-1]]*len(x_list), label="latest level "+str(z_gap[-1]), color="red")

    plt.title("relative differences")
    plt.xticks(x_list, get_xticks(x_label), rotation=90)
    plt.grid(True)
    plt.legend()
    plt.show()  # display
    return 

# %%
### trend by moving average
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
    ('ZLAB', '9688.HK', 0.5*7.8),
    ('TENCENT', '0700.HK', 1*7.8),
    ('SMIC', '0981.HK', 1*7.8),
    ('SenseTime', '0020.HK', 1*7.8),
    ('Kuaishou', '1024.HK', 1*7.8),
    ('Xiaomi', '1810.HK', 1*7.8)
]

st, et = "2021-01-01", "2023-08-31"

trend_parameters = {}
for n_stocks in range(1, len(stocks_info)+1):
    it = stocks_info[n_stocks-1]
    
    ref_code, stock_code, _ = it
    print(ref_code, stock_code)
    #
    search_list = []
    for ma1 in range(2, 20): #
        for ma2 in range(ma1+5, 40):
            avg_ratio = 1
            # 
            df_data1 = get_df_data02(stock_code, st, et, ma1, ma2)
            day_num = 500 # latest two years
            trend_df = df_data1[-day_num:].copy()
            trade_df, win_loss_ratio = buy_sell_info(trend_df)
            trade_df['duration'] = trade_df['sell-date'] - trade_df['buy-date']
            #print( ref_code, stock_code, round(win_loss_ratio*100, 2), sep="\t" )
            avg_ratio = avg_ratio * (win_loss_ratio*100)    
            #search_list.append( [math.pow(avg_ratio, 1/n_stocks), ma1, ma2] )
            search_list.append( [round(win_loss_ratio*100, 1), ma1, ma2, trade_df] )
    
    top10 = sorted(search_list, reverse=True)[:10]
    trend_parameters[ref_code+"_"+stock_code] = top10

# %%
#print(len(df_collection), df_collection.keys())
for k in trend_parameters.keys():
    r, ma1, ma2, trade_df = trend_parameters[k][0]
    print(k, r, ma1, ma2, len(trade_df), sep="\t")

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
    ('Xiaomi', '1810.HK', 1*7.8),
    ('CMB', '3968.HK', 1*7.8),
]

for it in stocks_info[:]: #[('ZLAB', '9688.HK', 0.5*7.8)]:#
    st, et = "2021-01-01", "2023-08-31"
    ref_code, stock_code, _ = it
    # 
    ma1, ma2 = 5, 20
    """ 
    k = ref_code+"_"+stock_code
    r, ma1, ma2, trade_df = trend_parameters[k][0]
    print(k, r, ma1, ma2, len(trade_df), sep="\t")
    """
    #
    df_data1 = get_df_data(stock_code, st, et, 'Close', 'Close', ma1, ma2)

    day_num = 120
    trend_df = df_data1[-day_num:].copy()
    draw_trend(trend_df, ref_code, stock_code, [1, 4, 5], ma1, ma2, 20, 2) # long-short terms MA, Bollinger band

    day_num = 120
    daily_df = df_data1[-day_num:].copy()
    draw_dailyReturn(daily_df, ref_code, stock_code)

    weekly_df = df_data1[:].copy()
    draw_weeklyReturn(weekly_df, ref_code, stock_code)

# %%
stocks_info = [
    # http://www.aastocks.com/tc/usq/market/china-concept-stock.aspx
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
    ('TME', '1698.HK', 1*7.8),
    ('TENCENT', '0700.HK', 1*7.8),
    ('SMIC', '0981.HK', 1*7.8),
    ('SenseTime', '0020.HK', 1*7.8),
    ('Kuaishou', '1024.HK', 1*7.8),
    ('Xiaomi', '1810.HK', 1*7.8),
    ('CMB', '3968.HK', 1*7.8),
    ('PDD', 'PDD', 1*7.8),
    # https://tw.tradingview.com/markets/stocks-usa/market-movers-large-cap/
    ('Apple', 'AAPL', 1*7.8),
    ('Microsoft', 'MSFT', 1*7.8),
    ('Google', 'GOOG', 1*7.8),
    ('Amazon', 'AMZN', 1*7.8),
    ('NVIDIA', 'NVDA', 1*7.8),
    ('Meta', 'META', 1*7.8),
    ('Tesla', 'TSLA', 1*7.8),
    ('Exxon', 'XOM', 1*7.8),
    ('Visa', 'V', 1*7.8),
    ('UnitedHealth', 'UNH', 1*7.8),
    ('Johnson & Johnson', 'JNJ', 1*7.8),
    ('Walmart', 'WMT', 1*7.8),
    ('JP Morgan', 'JPM', 1*7.8),
    ('Procter & Gamble', 'PG', 1*7.8),
    ('Eli Lilly', 'LLY', 1*7.8),
    ('Mastercard', 'MA', 1*7.8),
]


st, et = "2021-08-01", "2023-08-31"

stock_corr_df = basic_correlations(stocks_info[:], st, et, cpn='Close', mode='HK') # intersected time for US & HK stocks

# %%
print( st, et )
for stock_it in stocks_info:
    stock_name1, stock_code1, _ = stock_it
    if len(stock_corr_df[ stock_corr_df.index==stock_name1 ])<1:
        continue
    cur_row = stock_corr_df[ stock_corr_df.index==stock_name1 ].iloc[0]
    related_stocks = []
    for stock_name2 in cur_row.index:
        if stock_name1!=stock_name2:
            #print( stock_name1, stock_name2, round(cur_row[stock_name2]*100, 2) )
            related_stocks.append( (round(cur_row[stock_name2]*100, 2), stock_name1, stock_name2) )
    print( sorted(related_stocks, reverse=True)[:5] )

# %%
corr_df = stock_corr_df.copy()

stocks_info = [ ('TME', '1698.HK', 1*7.8), ('TENCENT', '0700.HK', 1*7.8) ] # ('TCOM', '9961.HK', 1 * 7.8), ('PDD', 'PDD', 1*7.8)
st, et = "2022-12-01", "2023-08-31"

stock_name1, stock_code1, _ = stocks_info[0]
stock_name2, stock_code2, _ = stocks_info[1]

it_df1 = get_df_data(stock_code1, st, et)
it_df2 = get_df_data(stock_code2, st, et)

corr_score = corr_df[ corr_df.index==stock_name1 ][stock_name2]
corr_score = list(corr_score)[0]
print( stock_name1, "and", stock_name2, ":\t", round(corr_score, 3) )


y1 = relative_values(list(it_df1['daily_return'])[1:], 0)
y2 = relative_values(list(it_df2['daily_return'])[1:], 0)
x_label = list(it_df1.index)[1:]
x_list = list( range(len(y1)) )

draw_lines(x_list, x_label, y1, y2, stock_name1+"_"+stock_code1, stock_name2+"_"+stock_code2)

z_gap = []
i = 0
while i<len(y1):
    v = round((y1[i]-y2[i])*-1, 1)
    z_gap.append( v )
    i += 1

draw_bars(x_list, x_label, z_gap)

# %%
st, et = "2021-06-01", "2023-07-31"
tmp_df = get_df_data('3968.HK', st, et)
tmp_df[tmp_df['Low']<25.3]
tmp_df[-30:]

# %%


# %%



