#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt 
#get_ipython().system('pip install yfinance')
import yfinance as yf # https://pypi.org/project/yfinance/
import math
import random
import seaborn as sns
import datetime
import pandas as pd
from scipy import stats # python -m pip install scipy
import warnings
warnings.filterwarnings(action='ignore')


# In[14]:


# get data by ticker-name, start-time & end-time
def get_df_data(ticker_name="AAPL", start_time="2022-01-01", end_time="2022-10-09"):
    df_data = yf.download(tickers=ticker_name, start=start_time, end=end_time) 
    df_data = df_data[ ["Open", "High", "Low", "Close", "Volume"] ]
    df_data['previous_Close'] = df_data['Close'].shift(1)
    df_data['daily_return'] = (df_data['Close']-df_data['previous_Close'])/df_data['previous_Close']
    return df_data

def search_by_index(df_data, index_time):
    tmp_list = list(df_data.index)
    i = 0
    for t in tmp_list:
        if t>=index_time:
            break
        i += 1
    return i-1

def merge_stocks(df_data1, df_data2):
    data = []
    for i in range(len(df_data1)):
        # get the info of target stock on current business day
        index1 = df_data1.index[i] 
        it1 = df_data1.iloc[i]    
        open1, high1, low1, close1 = it1['Open'], it1['High'], it1['Low'], it1['Close']
        daily_return1 = it1['daily_return']
        volume1 = it1['Volume']
            # get the info of reference stock on previous business day
        j = search_by_index(df_data2, index1)
        index2 = df_data2.index[j] 
        it2 = df_data2.iloc[j]    
        open2, high2, low2, close2 = it2['Open'], it2['High'], it2['Low'], it2['Close']
        daily_return2 = it2['daily_return']
        volume2 = it2['Volume']

        if index1>index2:
            tmp_list = [ index1, open1, high1, low1, close1, index2, open2, high2, low2, close2 ]
            data.append( tmp_list )
        i += 1
    col_names = ['target_time', 'target_open', 'target_high', 'target_low', 'target_close',
                'ref_time', 'ref_open', 'ref_high', 'ref_low', 'ref_close']
    df = pd.DataFrame(data, columns = col_names)
    return df

def LR(x, y):
    k, b, R, p, std_err = stats.linregress(x, y) # R*R -> R2
    print( '\tlinear model: y = ', round(k, 4), '* x + ', round(b, 4), "\t R2:", round(R*R, 4), "\t std error:", round(std_err, 3) )
    mymodel = []
    for v in x:
        mymodel.append( k*v + b )
    assert len(y)==len(mymodel)
    print( "\tdata numbers (x & y): ", len(y), len(mymodel) )
    return k, b, R, std_err, mymodel

def move_line(old_list, offset_y=2):
    new_list = []
    for v in old_list:
        new_list.append(v-offset_y*v/100)
    return new_list

def lists_dif(y1, y2):
    dif_list = []
    i = 0
    while i<len(y1):
        v = (y1[i] - y2[i])/y2[i] * 100
        if v<0:
            v = -v
        dif_list.append( v )
        i += 1
    df = pd.DataFrame(dif_list, columns = ['model_error'])
    return df

def printing01(label, num):
    print( label, "\t", round(num, 2), "%" )
    return

def printing02(label, num):
    print( label, "\t", round(num, 2) )
    return

def printing03(it):
    st = ""
    for v in it:
        st = st + str(v) + "\t"
    print(st)
    return

def remove_timezone(dt):
    # HERE `dt` is a python datetime
    # object that used .replace() method
    return dt.replace(tzinfo=None)

def create_bins(from_num, to_num, N):
    delta = round((to_num-from_num)/N, 1)
    tmp = []
    r1 = -100
    for i in range(N+1):
        r2 = from_num + i*delta
        tmp.append( (r1, r2) )
        r1 = r2
    r2 = 100
    tmp.append( (to_num, r2) )
    return tmp

def which_bin(num, bin_list):
    for it in bin_list:
        n1, n2 = it
        if num>n1 and num<=n2:
            return it
    return ()


def range_frequency(num_list, bin_list):
    dic = {}
    for v in sorted(num_list):
        k = which_bin(v, bin_list)
        if k not in dic:
            dic[k] = 1
        else:
            dic[k] += 1
    cumulative = 0
    for k in list(dic.keys()):
        r = round(dic[k]*100/len(num_list), 2)
        cumulative += r
        print( k, "\t", dic[k], "\t", r, "\t", round(cumulative, 1) )
    return list(dic.keys()), list(dic.values())


def get_items(tmp_df, j, buy_price, buy_time):
    tmp_it = tmp_df.iloc[j]
    sell_price, sell_time = tmp_it['target_high'], tmp_it['target_time01']
    return (round(sell_price-buy_price, 2), (sell_time-buy_time).days)

def filter_by_time(df_data1, df_data2):
    merged_data = merge_stocks(df_data1, df_data2)
    oneday = datetime.timedelta(days=1)
    twodays = datetime.timedelta(days=2)
    merged_data['target_time01'] = merged_data['target_time'].apply(remove_timezone)
    merged_data['ref_time01'] = merged_data['ref_time'].apply(remove_timezone)
    merged_data['time_dif'] = merged_data['target_time01'] - merged_data['ref_time01']
    copied_data = merged_data.copy()
    merged_data = merged_data[ merged_data['time_dif'] == oneday ]
    print( "merged_data:\t", len(merged_data) )
    print('model info:')
    print( "\tref_time \t", list(merged_data['ref_time'])[0], "\t", list(merged_data['ref_time'])[-1] )
    print( "\ttarget_time \t", list(merged_data['target_time'])[0], "\t", list(merged_data['target_time'])[-1] )
    return merged_data, copied_data

def model_error_info01(y, mymodel):
    print( "\tmodel_error - absolute values: " )
    error_df = lists_dif(y, mymodel)
    printing01('\tmodel_error '+'max', error_df['model_error'].max())
    printing01('\tmodel_error '+'min', error_df['model_error'].min())
    printing01('\tmodel_error '+'mean', error_df['model_error'].mean())
    printing01('\tmodel_error '+'median', error_df['model_error'].median())
    printing01('\tmodel_error '+'std', error_df['model_error'].std())
    printing01('\tmodel_error '+'10%', error_df['model_error'].quantile(0.1))
    printing01('\tmodel_error '+'25%', error_df['model_error'].quantile(0.25))
    printing01('\tmodel_error '+'50%', error_df['model_error'].quantile(0.5))
    printing01('\tmodel_error '+'75%', error_df['model_error'].quantile(0.75))
    printing01('\tmodel_error '+'90%', error_df['model_error'].quantile(0.9))
    return error_df

def model_error_info02(merged_data, label_name):
    print( "\tmodel_error - relative values: " )
    printing01('\tpred_error '+'10%', merged_data[label_name+'_dif'].quantile(0.1))
    printing01('\tpred_error '+'20%', merged_data[label_name+'_dif'].quantile(0.2))
    printing01('\tpred_error '+'25%', merged_data[label_name+'_dif'].quantile(0.25))
    printing01('\tpred_error '+'50%', merged_data[label_name+'_dif'].quantile(0.5))
    printing01('\tpred_error '+'75%', merged_data[label_name+'_dif'].quantile(0.75))
    printing01('\tpred_error '+'80%', merged_data[label_name+'_dif'].quantile(0.8))
    printing01('\tpred_error '+'90%', merged_data[label_name+'_dif'].quantile(0.9))
    return 

def model_visualization(x, y, mymodel, label_name, error_df, pred_x, pred_y, show_or_not):
    if show_or_not:
        plt.subplots(figsize=(20, 10))
        plt.scatter(x, y, label=label_name) # points
        plt.plot(x, mymodel, label=label_name+" LR") # model
    # error range
    for percentile in [75]: # 10, 25, 50, 75, 
        error_gap = error_df['model_error'].quantile( round(percentile/100.0, 2) )
        error_gap = round(error_gap, 2)
        print( "\terror_gap: ", error_gap, "%" )
        if show_or_not:
            plt.plot(x, move_line(mymodel,-error_gap), label=label_name+" LR - safety -"+str(percentile)+"%")
            plt.plot(x, move_line(mymodel,+error_gap), label=label_name+" LR - safety +"+str(percentile)+"%")
  # predict today point
    print( "\t", label_name+" ref: ", pred_x, "\t", label_name+" target predicted: ", round(pred_y, 2) )
    if show_or_not:
        plt.scatter([pred_x], [pred_y], label="prediction", color ="red")
        plt.legend()
        plt.show()
    return 

def return_analysis(copied_data, models_parameters):
    log_str = ""
    col_names = [
      'target_time01', 'target_open', 'target_high', 'target_low', 'target_close', 
      'ref_time01', 'ref_open', 'ref_high', 'ref_low', 'ref_close'
    ]
    tmp_df = copied_data[ col_names ]
    counter00, counter01, counter02, counter03 = 0, 0, 0, 0
    return_time_info = []
    possible_hold_days = 15
    for i in range(len(tmp_df)-possible_hold_days): # 
        it = tmp_df.iloc[i]
        target_time01 = it['target_time01']
        ref_time01 = it['ref_time01']
        time_dif = target_time01 - ref_time01
        # oneday = datetime.timedelta(days=1)
        label_name = "low"
        k, b, r2 = models_parameters[label_name]
        #
        target_close = it['target_close']
        target_low, ref_low = it['target_low'], it['ref_low']
        pred_low = ref_low * k + b
        pred_low = pred_low * 0.985
        pred_low = round( pred_low, 1 )
        if pred_low-target_low>=0.1:
            possible_price = [ (round(target_close-pred_low, 2), 0) ]
            j = i + 1
            while j<=min(i+possible_hold_days, len(tmp_df)-1):
                possible_price.append( get_items(tmp_df, j, pred_low, target_time01) )
                j += 1
              #
            max_return = sorted(possible_price)[-1][0]
            st = str(target_time01) + "\t" + str(ref_time01) + "\t" + str(time_dif.days) + "\t" + str(pred_low) + "\t" + str(max_return) + "\t" + str(possible_price)
            #print( st )
            log_str = log_str + st + "\n"
            max_return = round(max_return*100/pred_low, 2)
            return_time_info.append( max_return )
            if max_return>=2: # return>=2%
                counter02 += 1 
            else:
                counter03 += 1
            counter01 += 1
        counter00 += 1
    s1 = str(counter00) + "\ttradable day:\t" + str(counter01) + "\t" + str(round(counter01/counter00*100, 2))
    s2 = "\treturn>=2%:\t" + str(counter01) + "\t" + str(counter02) + "\t" + str(round(counter02/counter01*100, 2))
    s3 = "\treturn<2%:\t" + str(counter01) + "\t" + str(counter03) + "\t" + str(round(counter03/counter01*100, 2))
    #print( s1 )
    #print( s2 )
    #print( s3 )
    log_str = log_str + s1 + "\n" + s2 + "\n" + s3 + "\n"
    return return_time_info, log_str

def return_info(return_time_info):
    return_df = pd.DataFrame(return_time_info, columns = ['model_return'])
    printing01('\tmodel_return '+'10%', return_df['model_return'].quantile(0.1))
    printing01('\tmodel_return '+'20%', return_df['model_return'].quantile(0.2))
    printing01('\tmodel_return '+'25%', return_df['model_return'].quantile(0.25))
    printing01('\tmodel_return '+'30%', return_df['model_return'].quantile(0.3))
    printing01('\tmodel_return '+'50%', return_df['model_return'].quantile(0.5))
    printing01('\tmodel_return '+'70%', return_df['model_return'].quantile(0.7))
    printing01('\tmodel_return '+'75%', return_df['model_return'].quantile(0.75))
    printing01('\tmodel_return '+'80%', return_df['model_return'].quantile(0.8))
    printing01('\tmodel_return '+'90%', return_df['model_return'].quantile(0.9))
    return 


# In[18]:


stocks_info = [
    ('BABA', '9988.HK', '2022-11-01', 1),
    ('BIDU', '9888.HK', '2022-10-25', 1),
    ('JD', '9618.HK', '2022-10-24', 0.5 * 7.8),
    ('MPNGY', '3690.HK', '2022-10-24', 0.5 * 7.8),
    ('NTES', '9999.HK', '2022-10-26', 0.2 * 7.8),
    ('LI', '2015.HK', '2022-10-26', 0.5 * 7.8),
    ('NIO', '9866.HK', '2022-10-26', 1 * 7.8)
]
#('BIDU', '9888.HK', '2022-10-25', 1), # ('JD', '9618.HK', '2022-10-24', 0.5 * 7.8), ('NTES', '9999.HK', '2022-10-26', 0.2 * 7.8),
stocks_info = [ 
    ('BABA', '9988.HK', '2022-11-01', 1),
    ('MPNGY', '3690.HK', '2022-10-24', 0.5 * 7.8),
    ('LI', '2015.HK', '2022-10-26', 0.5 * 7.8) # ('NIO', '9866.HK', '2022-10-26', 1 * 7.8)
]
whole_log_str = ""
pred_info = []
for it in stocks_info[:]:
    # get market data of target-stock & reference-stock
    reference_stock, target_stock, st, ratio = it
    st, et = "2022-08-01", "2023-02-28"
    print("time range:\t", st, "-", et)
    df_data1 = get_df_data(ticker_name=target_stock, start_time=st, end_time=et)
    print("target stock:\t", target_stock, "\t", len(df_data1))
    df_data2 = get_df_data(ticker_name=reference_stock, start_time=st, end_time=et)
    print("reference stock:\t", reference_stock, "\t", len(df_data2))

    # target stock (datetime, open, high, low, close), 1-day previous reference stock (datetime, open, high, low, close) 
    merged_data, copied_data = filter_by_time(df_data1, df_data2)
   
    # Linear regression to fit data of target-stock & reference-stock
    models_parameters = {} 
    for label_name in ["open", "high", "low", "close"]:
        print( label_name )
          # model info
        x, y = list(merged_data['ref_' + label_name]), list(merged_data['target_'+label_name])
        k, b, R, std_err, mymodel = LR(x, y)
        models_parameters[label_name] = [k, b, R*R] # record model parameters
        error_df = model_error_info01(y, mymodel)
          # data visualization
        ref_it = df_data2.iloc[-1]
        ref_dic = {"high":ref_it['High'], "low":ref_it['Low'], "open":ref_it['Open'], "close":ref_it['Close']}
        pred_x =  round(ref_dic[label_name], 2)
        pred_y = k*pred_x+b
        show_or_not = False # True # 
        model_visualization(x, y, mymodel, label_name, error_df, pred_x, pred_y, show_or_not)
        pred_info.append( (reference_stock, target_stock, label_name, pred_x, round(pred_y, 2), round(pred_y*0.985, 1)) )
  
    # more details about the model errors
    for label_name in ["open", "high", "low", "close"]:
        k, b, r2 = models_parameters[label_name]
        merged_data[label_name+"_pred"] = merged_data["ref_"+label_name]*k + b
        merged_data[label_name+"_dif"] = (merged_data["target_"+label_name] - merged_data[label_name+"_pred"])/merged_data[label_name+"_pred"]*100
        merged_data[label_name+"_dif"] = merged_data[label_name+"_dif"].round(1)
        print(label_name, len(merged_data), "records")
        model_error_info02(merged_data, label_name)  
        # distribution analysis of model errors
        """
        print(label_name, " difference distribution: ")
        num_list = list(merged_data[label_name+'_dif'])
        bin_list = create_bins(-4, 4, 16)
        x_label, y = range_frequency(num_list, bin_list)
        x = list( range(len(y)) )
        plt.bar(x, y, color ='maroon', width = 0.4)
        plt.xticks(x, x_label, rotation ='vertical')
        plt.show()
        """

    # algorithmic trading, return analysis if buying according the predicted values
    print( "algorithmic trading - return analysis if buying according the predicted low-price: " )
    print( "\tbuy at 99% * predicted low-price, and see the possible returns in 10 business days" )
    print(reference_stock, "-", target_stock)
    return_time_info, sub_log_str = return_analysis(copied_data, models_parameters)
    return_info(return_time_info)
    whole_log_str = whole_log_str + reference_stock + "-" + target_stock + "\n" + sub_log_str

info_dic = {}
stock_head = ""
for ln in whole_log_str.strip().split("\n"):
    if ".HK" in ln: # new stock info
        stock_head = ln
        #print( stock_head )
    else:
        ln = stock_head + "\t" + ln
  #
    if "2022-" in ln or "2023-" in ln:
        it = ln.split("\t")
        stock_info, target_time, ref_time = it[0], it[1], it[2]
        buy_price, return_per_share = float(it[4]), float(it[5])
        return_per_share = round(return_per_share/buy_price*100, 2)
        #print( it )
        if target_time not in info_dic:
            info_dic[target_time] = [(return_per_share, buy_price, stock_info)]
        else:
            info_dic[target_time].append((return_per_share, buy_price, stock_info))
    #else:
    #print( ln )
print("trading days: ", len(info_dic))
profit_list = []
for k in sorted( list(info_dic.keys()) ):
    v = sorted(info_dic[k], reverse=True)
    print(k, "\t", v)
    profit_list.append( v[-1] )

profit_counter01, profit_counter02, profit_counter03 = 0, 0, 0
for it in sorted(profit_list):
    print(it)
    if it[0]<1.5:
        profit_counter01 += 1
    elif it[0]<=2.0:
        profit_counter02 += 1
    else:
        profit_counter03 += 1
print(stocks_info)
sum1 = profit_counter01 + profit_counter02 + profit_counter03
print( sum1, "\t", profit_counter01, "\t", round(profit_counter01*100/sum1, 2) )
print( sum1, "\t", profit_counter02, "\t", round(profit_counter02*100/sum1, 2) )
print( sum1, "\t", profit_counter03, "\t", round(profit_counter03*100/sum1, 2) )


# In[6]:
print("ref_stock, target_stock, label_name, ref_price, pred_price, adjusted_price")
for it in pred_info:
    #ref_stock, target_stock, label_name, ref_price, pred_price, adjusted_price = it
    #print(ref_stock, target_stock, label_name, ref_price, pred_price, adjusted_price)
    printing03(it)

# In[4]:


"""
stocks_info = [
    ('BABA', '9988.HK', '2022-11-01', 1),
    ('BIDU', '9888.HK', '2022-10-25', 1),
    ('JD', '9618.HK', '2022-10-24', 0.5 * 7.8),
    ('MPNGY', '3690.HK', '2022-10-24', 0.5 * 7.8),
    ('NTES', '9999.HK', '2022-10-26', 0.2 * 7.8),
    ('LI', '2015.HK', '2022-10-26', 0.5 * 7.8),
    ('NIO', '9866.HK', '2022-10-26', 1 * 7.8)
]

for info in stocks_info[:]:
    ref_code, target_code, _, _ = info

    df_data_tmp = get_df_data(ticker_name=target_code, start_time='2022-09-01', end_time='2023-02-28') # 000001.SS
    df_data_tmp['MA1'] = df_data_tmp['Close'].rolling(5).mean()
    df_data_tmp['MA2'] = df_data_tmp['Close'].rolling(10).mean()

    ###
    plt.subplots(figsize=(20, 10))
    plt.title(target_code+" - "+ref_code)
    
    x_axis = []
    x_label = []
    for i in range(len(df_data_tmp)):
        it = df_data_tmp.iloc[i]
        hp, lp, op, cp = it['High'], it['Low'], it['Open'], it['Close']
        daily_return = it['daily_return']    
        #x_label.append(  )
        if daily_return>0:
            clr = "green"
        else:
            clr = "red"
        plt.plot([i, i], [hp, lp], color=clr) # , label="lines-"+ticker_code
        #plt.scatter([i, i], [hp, lp], color=clr)
        plt.plot([i, i], [op, cp], color=clr, linewidth=7.0)
        #
        x_axis.append(i)
        date_str = str(df_data_tmp.index[i].strftime('%y-%m-%d'))
        x_label.append(date_str)    
    
    plt.plot(x_axis, list(df_data_tmp['MA1']), color="blue", label="MA1: 5")
    plt.plot(x_axis, list(df_data_tmp['MA2']), color="orange", label="MA2: 10")
    #plt.plot(x_axis, list(df_data_tmp['High']), color="fuchsia", label="high")
    plt.xticks( x_axis, x_label, rotation='vertical' )
    plt.legend()
    plt.show()  # display
"""


# In[ ]:




