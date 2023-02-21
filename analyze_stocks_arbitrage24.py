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

def calculate_weeknum(df_data):
    df_data['weekday'] = df_data.index.weekday
    start_weekday = df_data['weekday'][0]
    weeknum_list = []
    for i in range(len(df_data)):
        cur_date = df_data.index[i]
        days = (cur_date-df_data.index[0]).days+start_weekday
        weeknum_list.append( days//7 )
    df_data['weeknum'] = weeknum_list
    return df_data

# get data by ticker-name, start-time & end-time
def get_df_data(ticker_name="AAPL", start_time="2022-01-01", end_time="2022-10-09"):
    df_data = yf.download(tickers=ticker_name, start=start_time, end=end_time) 
    df_data = df_data[ ["Open", "High", "Low", "Close", "Volume"] ]
    df_data['previous_Close'] = df_data['Close'].shift(1)
    df_data['daily_return'] = (df_data['Close']-df_data['previous_Close'])/df_data['previous_Close']
    df_data['overnight_return'] = (df_data['Open']-df_data['previous_Close'])/df_data['previous_Close']
    MA1, MA2 = 5, 20
    df_data['MA1'] = df_data['Close'].rolling(MA1).mean()
    df_data['MA2'] = df_data['Close'].rolling(MA2).mean()
    #
    df_data = calculate_weeknum(df_data)
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
            volume1 = it1['Volume']
            target_MA1, target_MA2 = it1['MA1'], it1['MA2']
            it2 = df_data2.iloc[j-1]
            open2, high2, low2, close2 = it2['Open'], it2['High'], it2['Low'], it2['Close']
            daily_return2 = it2['daily_return']
            volume2 = it2['Volume']
            ref_MA1, ref_MA2 = it2['MA1'], it2['MA2']

            tmp_list = [ target_time, open1, high1, low1, close1, ref_time, open2, high2, low2, close2 ]
            data.append( tmp_list )
            #print( target_time, ref_time )
        i += 1
    col_names = ['target_time', 'target_open', 'target_high', 'target_low', 'target_close',
                'ref_time', 'ref_open', 'ref_high', 'ref_low', 'ref_close']
    df = pd.DataFrame(data, columns = col_names)
    return df

def LR(train_data, label_name, min_num=20):
    x = train_data["ref_"+label_name]
    y = train_data["target_"+label_name]
    if len(x)<min_num:
        return 1, 0, 0
    k, b, R, p, std_err = stats.linregress(list(x), list(y)) # R*R -> R2
    #print( '\tlinear model: y = ', round(k, 4), '* x + ', round(b, 4), "\t R2:", round(R*R, 4), "\t std error:", round(std_err, 3) )
    return k, b, R*R

def daily_models(merged_data, label_name, min_num):
    data = []
    k_list, b_list, R2_list = [], [], []
    i = 0
    while i<len(merged_data):
        train_data = merged_data.iloc[ max(0, i-100):i ].copy()
        k, b, R2 = LR(train_data, label_name, min_num)
        k_list.append( k )
        b_list.append( b )
        R2_list.append( R2 )
        if len(train_data)>0:            
            it = merged_data.iloc[i]
            target_time = it['target_time']
            target_point = it['target_'+label_name]
            ref_time = it['ref_time']
            ref_point = it['ref_'+label_name]
            train_data_it1, train_data_it2 = train_data.iloc[0], train_data.iloc[-1]
            target_from_time,target_to_time = train_data_it1['target_time'], train_data_it2['target_time']
            ref_from_time,ref_to_time = train_data_it1['ref_time'], train_data_it2['ref_time']
            #print( target_time, "\t", target_time1, ref_time1, "\t", target_time2, ref_time2 )
            data.append( [target_time, target_point, ref_time, ref_point, target_from_time, target_to_time, ref_from_time, ref_to_time, k, b, R2] )
        i += 1
    #
    col_names = ['target_time', 'target_point', 'ref_time', 'ref_point', 'target_from_time', 'target_to_time', 'ref_from_time', 'ref_to_time',
                'k_'+label_name, 'b_'+label_name, 'R2'+label_name]
    df = pd.DataFrame(data, columns = col_names)
    return df[min_num:]

def calculate_buy_price(model_df, label_name, MA_days1, MA_days2, adj_ratio=0.985):
    model_df['k_'+label_name+"_MA"] = model_df['k_'+label_name].rolling(MA_days1).mean()
    model_df['b_'+label_name+"_MA"] = model_df['b_'+label_name].rolling(MA_days1).mean()
    model_df['k_'+label_name+"_MA_MA"] = model_df['k_'+label_name+"_MA"].rolling(MA_days2).mean()
    model_df['b_'+label_name+"_MA_MA"] = model_df['b_'+label_name+"_MA"].rolling(MA_days2).mean()
      # recommended buy-price
    model_df['buy_price'] = (model_df['ref_point']*model_df['k_'+label_name+"_MA_MA"]+model_df['b_'+label_name+"_MA_MA"])*adj_ratio        
    buy_df = model_df[ model_df['buy_price']>=model_df['target_point'] ].copy()
    return buy_df[MA_days1+MA_days2:]


def search_by_index(df_data1, target_time):
  i = 0
  while i<len(df_data1):
    current_time = df_data1.index[i]
    if target_time==current_time:
      return i
    i += 1
  return -1

def possible_returns(buy_df, df_data1, hold_days):
  hold_data1 = []
  hold_data2 = []
  Len = 5 + hold_days*3 + 2 # number of columns
  i = 0
  while i<len(buy_df):
    it1 = buy_df.iloc[i]
    target_time = it1['target_time']
    buy_price = it1['buy_price']
    wn = get_weeknum(df_data1, target_time) # current week
    pw1, pw2 = weekly_return2(df_data1, wn-1), weekly_return2(df_data1, wn-2)

    j = search_by_index(df_data1, target_time)
    it2 = df_data1.iloc[j]
    hold_time = df_data1.index[j]
    assert target_time == hold_time
    hold_info = [ target_time, round(buy_price,1) ]
    hold_info.append( hold_time )
    hold_info.append( round(it2['Low'],1) )
    hold_info.append( round(it2['Close'],1) )
    j = j + 1
    holding_period = min(len(df_data1), j+hold_days)
    while j<holding_period:
      it2 = df_data1.iloc[j]
      hold_time = df_data1.index[j]
      hold_info.append( hold_time )
      hold_info.append( round(it2['Low'],1) )
      hold_info.append( round(it2['High'],1) )
      j += 1
        
    hold_info.append( pw1 )
    hold_info.append( pw2 )

    if len(hold_info)==Len:
      hold_data1.append( hold_info )
    else:
      hold_data2.append( hold_info )
    i += 1
  col_names = ["buy_time", "buy_price"]
  for day in range(hold_days+1):
    st = "sell_"+str(day+1)
    col_names.append( st )
    col_names.append( st+"_low" )
    col_names.append( st+"_high" )
  col_names.append( "prev_1week" )
  col_names.append( "prev_2week" )
  df = pd.DataFrame(hold_data1, columns = col_names)  
  return df, hold_data2

def row_to_list(hold_df, i):
  it = hold_df.iloc[i]
  buy_time, buy_price = it['buy_time'], it['buy_price']
  data_list = list(it)
  floor_list, ceiling_list = [], []
  date_list = []
  num = 2
  while num<len(data_list)-2:
    sell_time = data_list[num]
    num += 1
    sell_low = data_list[num]
    num += 1
    sell_high = data_list[num]
    num += 1
    date_list.append(sell_time)
    floor_list.append( sell_low ) # round(sell_low/buy_price*100, 2)
    ceiling_list.append( sell_high ) # round(sell_high/buy_price*100, 2)
  return date_list, floor_list, ceiling_list, buy_time, buy_price

def find_max(date_list, ceiling_list):
  max_date, max_price = date_list[0], ceiling_list[0]
  i = 1
  while i<len(date_list):
    if ceiling_list[i]>max_price:
      max_price = ceiling_list[i]
      max_date = date_list[i]
    i += 1
  return max_date, max_price

def find_min(date_list, floor_list):
  min_date, min_price = date_list[0], floor_list[0]
  i = 1
  while i<len(date_list):
    if floor_list[i]<min_price:
      min_price = floor_list[i]
      min_date = date_list[i]
    i += 1
  return min_date, min_price

def get_weeknum(df_data, tmp_date):
    weeknum = df_data.loc[tmp_date]['weeknum']
    return int(weeknum)

def weekly_return1(df_data, wn):
    week_df = df_data[ df_data['weeknum']==wn ]
    week_open = week_df['Open'][0]
    week_close = week_df['Close'][-1]
    r = (week_close-week_open)/week_open * 100
    #print( week_open, week_close, r )
    return round(r, 2)

def weekly_return2(df_data, wn):
    prev_week_df = df_data[ df_data['weeknum']==wn-1 ]
    cur_week_df = df_data[ df_data['weeknum']==wn ]
    prev_week_close = prev_week_df['Close'][-1]
    cur_week_close = cur_week_df['Close'][-1]
    r = (cur_week_close-prev_week_close)/prev_week_close * 100
    #print( prev_week_close, cur_week_close, r )
    return round(r, 2)

def weekly_volume(df_data, wn):
    prev_week_df = df_data[ df_data['weeknum']==wn-1 ]
    cur_week_df = df_data[ df_data['weeknum']==wn ]
    prev_week_vol = sum(prev_week_df['Volume'])
    cur_week_vol = sum(cur_week_df['Volume'])
    r = (cur_week_vol-prev_week_vol)/prev_week_vol * 100
    #print( prev_week_close, cur_week_close, r )
    return round(r, 2)

def get_MA(df_data, tmp_date):
    tmp_df = df_data.copy()
    tmp_df['MA1_prev'] = tmp_df['MA1'].shift(1)
    tmp_df['MA2_prev'] = tmp_df['MA2'].shift(1)
    MA1_prev, MA2_prev = tmp_df.loc[tmp_date]['MA1_prev'], tmp_df.loc[tmp_date]['MA2_prev']
    return round(MA1_prev, 2), round(MA2_prev, 2)

def evaluate1(df_data, hold_df, print_flag=True):
  data_info = []
  for i in range(len(hold_df)):
    date_list, floor_list, ceiling_list, buy_time, buy_price = row_to_list(hold_df, i)
    max_date, max_price = find_max(date_list, ceiling_list)
    min_date, min_price = find_min(date_list, floor_list)
    r1 = round((max_price-buy_price)/buy_price*100, 2)
    r2 = round((min_price-buy_price)/buy_price*100, 2)
    wn = get_weeknum(df_data, buy_time)
    prev_week1, prev_week2 = weekly_return2(df_data, wn-1), weekly_return2(df_data, wn-2)
    MA1_prev, MA2_prev = get_MA(df_data, buy_time)
    vol_change = weekly_volume(df_data, wn)
    data_info.append( [buy_time, buy_price, max_date, max_price, r1, min_date, min_price, r2, prev_week1, prev_week2, MA1_prev, MA2_prev, vol_change] )  
    
    if print_flag:      
      print(buy_time, buy_time.weekday(), "\t", round(buy_price, 2), "\t", max_date, "\t", max_price, "\t", r1, "\t\tweeks before: ", prev_week1, prev_week2, vol_change, "\t", min_date, "\t", min_price, "\t", r2)
  df = pd.DataFrame(data_info, columns = ['buy_time', 'buy_price', 'max_date', 'max_price', 'gain', 'min_date', 'min_price', 'loss', 'prev_week1', 'prev_week2', 'MA1_prev', 'MA2_prev', 'vol_change'])
  df['prev_weeks'] = df['prev_week1'] + df['prev_week2']
  df['MA_rate'] = (df['MA1_prev']-df['MA2_prev'])/df['MA2_prev']*100
  return df

def search_parameters(evaluate_df):
    parameters = []
    for p1 in range(1, 15):    
        tmp_df = evaluate_df.copy()
        tmp_df = tmp_df[ tmp_df['prev_week1']<=p1 ]
        #tmp_df = tmp_df[ tmp_df['prev_week2']<=p2 ]
        #tmp_df = tmp_df[ tmp_df['vol_change']<=p3 ]
        c1, c2 = len(tmp_df[ tmp_df['gain']<1.5 ]), len(tmp_df[ tmp_df['gain']>=1.5 ])
        if c2==0:
          continue
        r1, r2 = round(c1/(c1+c2)*100, 2), round(c2/(c1+c2)*100, 2)
        parameters.append( [r1, r2, p1, c1, c2, c2*r2] )
    #r1, r2, p1, p2, p3, c1, c2 = sorted(parameters, key=lambda it: (it[6], it[1]), reverse=True)[-1]
    L1 = sorted(parameters, key=lambda it: (it[1]), reverse=True)[-3:]
    L2 = sorted(parameters, key=lambda it: (it[5]), reverse=True)[-3:]
    L3 = sorted(parameters, key=lambda it: (it[4]), reverse=True)[-3:]
    return L1+L2+L3

stocks_info = [
    ('BABA', '9988.HK', '2022-11-01', 1),
    ('BIDU', '9888.HK', '2022-10-25', 1),
    ('JD', '9618.HK', '2022-10-24', 0.5 * 7.8),
    ('MPNGY', '3690.HK', '2022-10-24', 0.5 * 7.8),
    ('NTES', '9999.HK', '2022-10-26', 0.2 * 7.8),
    ('LI', '2015.HK', '2022-10-26', 0.5 * 7.8),
    ('NIO', '9866.HK', '2022-10-26', 1 * 7.8)
]

total_business_days = 0
gain_rate, loss_rate = 1.3, 9
trade_info_col = {}
for it in stocks_info[:6]: # 
    reference_stock, target_stock, st, ratio = it
    st, et = "2021-01-01", "2023-02-28"
    print("time range:\t", st, "-", et)
    df_data1 = get_df_data(ticker_name=target_stock, start_time=st, end_time=et)
    print("target stock:\t", target_stock, "\t", len(df_data1))
    df_data2 = get_df_data(ticker_name=reference_stock, start_time=st, end_time=et)
    print("reference stock:\t", reference_stock, "\t", len(df_data2))
        # merge reference-stock & target-stock: target stock (datetime, open, high, low, close), 1-day previous reference stock (datetime, open, high, low, close) 
    merged_data = merge_stocks(df_data1, df_data2)
    for label_name in ['low']: # "open", "high", "low", "close"
            # daily LR modelling
        model_df = daily_models(merged_data, label_name, 100)
            # moving average to smoothen model errors for daily models
        MA_days1, MA_days2 = 20, 5
        print("MA_days1: ", MA_days1)
        print("MA_days2: ", MA_days2)
        buy_df = calculate_buy_price(model_df, label_name, MA_days1, MA_days2, adj_ratio=0.985)
            # calculate the returns when holding some days
        holding_days = 30
        print("holding_days: ", holding_days)
        hold_df, hold_data2 = possible_returns(buy_df, df_data1, holding_days)
        evaluate_df = evaluate1(df_data1, hold_df, False)
        parameters = search_parameters(evaluate_df)
        for ps in parameters:
            [r1, r2, p1, c1, c2, _] = ps
            print("previous 1 week return <=", p1)
            print(c1 + c2, "\t", c1, "\t", r1)
            print(c1 + c2, "\t", c2, "\t", r2)
            # 
        recent_data = model_df.iloc[-1]        
        print( "target_time:\t", recent_data['target_time'] )
        print( "ref_time:\t", recent_data['ref_time'] )
        print( "ref_point:\t", round(recent_data['ref_point'], 1) )
        print( "buy_price:\t", round(recent_data['buy_price'], 1) )
    print()
