from bs4 import BeautifulSoup
import requests

from pyecharts.charts import Bar, Line # python3 -m pip install pyecharts
from pyecharts import options as opts

import re
# as per recommendation from @freylis, compile once only
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

def recording(every_line, file_path):
    f = open(file_path, "a", encoding="utf-8")
    f.write(every_line + "\n")
    f.close()
    return 

def get_controller():
    file_path = "C:/Users/Admin/Desktop/stocks_analyze_predict/controller01.txt"
    f = open(file_path, "r", encoding="utf-8")
    signal = f.readlines()
    if len(signal)>0:
        return signal[0].strip() == "True"
    return False

import datetime
import time

import keyboard # python3 -m pip install keyboard
def swtich_right_desktop():
    keyboard.press("ctrl")
    keyboard.press("windows")
    keyboard.press("right")
    keyboard.release("ctrl")
    keyboard.release("windows")
    keyboard.release("right")
    return
def swtich_left_desktop():
    keyboard.press("ctrl")
    keyboard.press("windows")
    keyboard.press("left")
    keyboard.release("ctrl")
    keyboard.release("windows")
    keyboard.release("left")
    return 

import random

time_list = []
price_list = []

stocks_info = [
    ('BABA', '9988.HK', 109.74, 1),
    ('BIDU', '9888.HK', 151.29, 1),
    ('JD', '9618.HK', 59.31, 0.5 * 7.8),
    ('MPNGY', '3690.HK', 44.52, 0.5 * 7.8),
    ('NTES', '9999.HK', 90.73, 0.2 * 7.8),
    ('LI', '2015.HK', 26.67, 0.5 * 7.8)
]

for i in range(100*6):
    if not get_controller():
        break
    try:
        #
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        stock_pair = stocks_info[ random.randint(0, len(stocks_info)-1) ]
        reference_stock_code, stock_code, reference_stock_price, ratio = stock_pair
        reference_stock_price = round(reference_stock_price * ratio, 2)
        
        stock_code = stock_code[:4] # 9988, 9998, 3690
        url = 'https://www.hstong.com/quotes/10000-0' + stock_code + '-HK'
        r = requests.get(url)
        soup = BeautifulSoup(r.content, "html.parser")
        st = cleanhtml(str(soup))
        lines = []
        for ln in st.split('\n'):
            if len(ln.strip())>0:
                lines.append( ln )
        stock_name, current_time, current_price, info = real_time_info(lines)
        #
        print((i+1), "\t", now, "\t", stock_name, "\t", current_time, "\t", current_price) #info
        tmp_list = info_process(info)
        for sub_info in tmp_list:
            print( "\t\t", sub_info )

        #
        price_list.append( float(current_price) )
        time_list.append( "2023-" + current_time[3:].strip() )

        # write to file        
        file_prefix = "C:/Users/Admin/Desktop/stocks_analyze_predict/"
        file_path = file_prefix+"common_stocks_log05.txt"#"DB_"+stock_code+"_quote02.txt"
        every_line = reference_stock_code+"\t"+str(reference_stock_price)+"\t"+stock_name+"\t"+current_time+"\t"+current_price
        recording(every_line, file_path)
        print(every_line)

        # data visualization
        time_list = []
        reference_price, target_price = [], []
        f = open(file_path, "r", encoding="utf-8")
        for ln in f.readlines()[:]:
            it = ln.strip().split("\t")
            [code1, price1, code2, time2, price2] = it
            code2 = code2[1:8]
            time2 = "2023-"+time2[4:]
            if code1==reference_stock_code and code2==stock_code+".HK":
                #print(code1, price1, code2, time2, price2)
                time_list.append( time2 )
                reference_price.append( float(price1) )
                target_price.append( float(price2) )
        graph_path = "C:/Users/Admin/Desktop/stocks_analyze_predict/lines02.html"
        dvx, dvy = [], []
        if len(target_price)>30:
            dvx = time_list[-30:]
            dvy = target_price[-30:]
        else:
            dvx = time_list
            dvy = target_price
        min_level = min( min(target_price), min(reference_price) )
        min_level = int(min_level)
        line_update = (
            Line(init_opts=opts.InitOpts(width="1800px", height="960px"))
            .add_xaxis(dvx)
            .add_yaxis(""+reference_stock_code, [reference_price[0]]*len(dvy))
            .add_yaxis(""+stock_code+".HK", dvy)
            .set_global_opts(title_opts=opts.TitleOpts(title="stocks quote"), 
                             tooltip_opts=opts.TooltipOpts(trigger="axis"),
                            yaxis_opts=opts.AxisOpts(name='dollar',splitline_opts=opts.SplitLineOpts(is_show=True),min_=min_level))# , subtitle="商店A中六樣商品數"

        )
        line_update.render(graph_path)
        
        #
        time.sleep( 10+random.randint(0, 20) )
        if i % 30 == 0:
            swtich_right_desktop()
            swtich_right_desktop()
            swtich_left_desktop()
            swtich_left_desktop()
            print('move screen to avoid computer sleeping')
    except:
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print((i+1), "\t", now, 'could not get the right info')
