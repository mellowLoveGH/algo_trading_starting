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

import datetime
import time


time_list = []
price_list = []

for i in range(100):
    #
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    reference_stock_code = 'MPNGY'
    reference_stock_price = round(44.08 * 3.9, 2)
    
    stock_code = '3690' # 9988, 9998
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

    # write to file
    price_list.append( float(current_price) )
    time_list.append( "2023-" + current_time[3:].strip() )
    file_prefix = "C:/Users/Admin/Desktop/stocks_analyze_predict/"
    file_path = file_prefix+"DB_"+stock_code+"_quote01.txt"
    every_line = str(reference_stock_price)+"\t"+stock_name+"\t"+current_time+"\t"+current_price
    recording(every_line, file_path)

    # show in graph
    graph_path = "C:/Users/Admin/Desktop/stocks_analyze_predict/lines02.html"
    dvx, dvy = [], []
    if len(price_list)>30:
        dvx = time_list[-30:]
        dvy = price_list[-30:]
    else:
        dvx = time_list
        dvy = price_list
    line_update = (
        Line(init_opts=opts.InitOpts(width="1800px", height="960px"))
        .add_xaxis(dvx)
        .add_yaxis(""+reference_stock_code, [reference_stock_price]*len(dvy))
        .add_yaxis(""+stock_code, dvy)
        .set_global_opts(title_opts=opts.TitleOpts(title="stocks quote"), 
                         tooltip_opts=opts.TooltipOpts(trigger="axis"),
                        yaxis_opts=opts.AxisOpts(name='dollar',splitline_opts=opts.SplitLineOpts(is_show=True),min_=160))# , subtitle="商店A中六樣商品數"
        
    )
    line_update.render(graph_path)

    #
    time.sleep( 10 )
