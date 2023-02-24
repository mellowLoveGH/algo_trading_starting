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

def LR(x, y, print_flag=True):
    k, b, R, p, std_err = stats.linregress(list(x), list(y)) # R*R -> R2
    if print_flag:
        print( '\tlinear model: y = ', round(k, 4), '* x + ', round(b, 4), "\t R2:", round(R*R, 4), "\t std error:", round(std_err, 3) )
    model_y = []
    for v in x:
        model_y.append( v*k+b )
    return round(k, 3), round(b, 3), round(R*R, 3), p, std_err, model_y

def get_dataset(df_data, i, j, label_name="Close"):
    assert min(i, j)>=0
    assert max(i, j)<len(df_data)
    date_list, x_list, y_list = [], [], []
    k = i
    while k<=j:
        xv = k
        x_list.append( xv )
        yv = round(df_data[label_name][k], 2)
        y_list.append( yv )
        date_list.append( df_data.index[k] )
        k += 1
    return date_list, x_list, y_list

def get_Matrix(M, N):
    model_info = []
    i = 0
    while i<M:
        it = []
        j = 0
        while j<N:
            it.append(0)
            j += 1
        model_info.append(it)
        i += 1
    return model_info

def print_Matrix(model_info):
    M = len(model_info)
    N = len(model_info[0])
    for i in range(M):
        for j in range(N):
            print(model_info[i][j], end=" ")
        print()
    return 
#print_Matrix(model_info)

def segementation(start_num, end_num):
    mn, mx = 5, 60
    seg_list = []
    n = start_num
    while n<end_num:
        seg_list.append(n)
        n = n + random.randint(mn, mx)
    if seg_list[-1]<=end_num-mn:
        seg_list.append(end_num-1)
    return seg_list

def model_performance(seg_list, model_info):
    root_value = 1
    i = seg_list[0]
    num = 0
    for j in seg_list[1:]:
        it = model_info[i][j]
        score, R2 = it
        #print( R2, end="\t" )
        root_value = root_value * R2
        num += 1
        i = j
    v = round(pow(root_value, 1/num), 3)
    # print("\t", v)
    return v
   

st, et = "2021-01-01", "2021-12-31"
st, et = "2022-01-01", "2022-10-24"
st, et = "2022-10-24", "2023-02-28"
stock_code = "9988.HK" # 0700.HK, 9988.HK, 9888.HK
df_data1 = get_df_data(stock_code, st, et)
df_data1
label_name="Close"

M, N = len(df_data1), len(df_data1)
model_info = get_Matrix(M, N)

i = 0
while i<len(df_data1):
    sub_info = []
    
    max_days = 90
    j = i+1
    while j<min(len(df_data1), i+max_days):
        date_list, x_list, y_list = get_dataset(df_data1, i, j, label_name)
        k, b, R2, p, std_err, model_y = LR(x_list, y_list, False)
        score = round(R2*(j-i), 2)
        model_info[i][j] = (score, R2)
        j += 1
    i += 1

col = {}
for times in range(1000*1000*100):
    start_num, end_num = 0, len(df_data1)
    seg_list = segementation(start_num, end_num)
    v = model_performance(seg_list, model_info)
    #
    k = str(seg_list)
    if k not in col:
        col[k] = v

print("len:\t", len(col))
sorted(col.items(), key=lambda x:x[1], reverse=True)[:3]


draw_graph = True

for it in sorted(col.items(), key=lambda x:x[1], reverse=True)[:3]:
    k, v = it
    k_list = []
    for n in k[1:-1].split(","):
        k_list.append( int(n) )
    print(k_list, v, end="\t")
    pv = k_list[0]
    for n in k_list[1:]:
        print(n-pv, end="\t")
        pv = n
    print()

    if draw_graph:
        plt.figure(figsize=(20, 6))

    x_values, x_labels = [], []

    i = k_list[0]
    for j in k_list[1:]:
        date_list, x_list, y_list = get_dataset(df_data1, i, j, label_name)
        k, b, R2, p, std_err, model_y = LR(x_list, y_list, True)
        date1, date2 = date_list[0], date_list[-1]
        print("\t", i, "-", j, "\t", date1, date2)
        i = j

        # next point
        px1 = x_list[-1] + 1
        py1 = px1*k+b


        if draw_graph:
            #plt.plot(x_list, y_list, label="real points")  # Plot the chart
            plt.plot(x_list, model_y, label="model points", linewidth=3)  # Plot the chart
            plt.scatter([px1], [py1], label="next points")  # Plot the chart

    if draw_graph:
        date_list, x_list, y_list = get_dataset(df_data1, 0, len(df_data1)-1, label_name)
        plt.plot(x_list, y_list, label="real points")  # Plot the chart        
        plt.xticks(x_list, date_list, rotation=90)
        plt.show()  # display