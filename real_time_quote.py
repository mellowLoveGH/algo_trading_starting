import numpy as np 
import matplotlib.pyplot as plt # python3 -m pip install matplotlib
import yfinance as yf # python3 -m pip install yfinance
import math
import random
import seaborn as sns # python3 -m pip install seaborn
import datetime
import pandas as pd
from pyecharts.charts import Bar, Line # python3 -m pip install pyecharts
from pyecharts import options as opts
import pyautogui # python3 -m pip install pyautogui
import keyboard # python3 -m pip install keyboard
from tkinter import Tk
import time


### functions
def draw_lines_html(x, y1, y2, title_name="stocks comparison"):
    line = (
        Line(init_opts=opts.InitOpts(width="1500px", height="800px"))
        .add_xaxis(x)
        .add_yaxis("BABA-Close", y1)
        .add_yaxis("9988_real-quote", y2)
        .set_global_opts(title_opts=opts.TitleOpts(title=title_name), 
                         tooltip_opts=opts.TooltipOpts(trigger="axis"),
                        yaxis_opts=opts.AxisOpts(name='dollar',splitline_opts=opts.SplitLineOpts(is_show=True),min_=80)) # , subtitle="商店A中六樣商品數"
    )
    line.render("C:/Users/Admin/Desktop/stocks_analyze_predict/lines01.html")
    return 

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

def keyboard_copy():
    keyboard.press("ctrl")
    keyboard.press("c")
    keyboard.release("ctrl")
    keyboard.release("c")
    return 

def switch_window(posx=600, posy=300, time_duration=0.2):
    keyboard.press("alt")
    keyboard.press("tab")
    keyboard.release("alt")
    keyboard.release("tab")
    pyautogui.moveTo(posx, posy, duration=time_duration)
    pyautogui.click(posx, posy)
    return 

def recording(every_line, file_path):
    f = open(file_path, "a")
    f.write(every_line + "\n")
    f.close()
    return 



###### running from here
file_path = "C:/Users/Admin/Desktop/stocks_analyze_predict/database_quote03.txt"
x, y1, y2 = [], [], []
refer_close = 89.86

for i in range(10):
        #
    swtich_right_desktop()
        # move mouse
    posx, posy = 1120, 310
    pyautogui.moveTo(posx, posy, duration=0.1)
    pyautogui.click(posx, posy)
    pyautogui.click(posx, posy)
    pyautogui.typewrite(["9", "9", "8", "8"]) # , "enter"
    posx, posy = 1360, 310
    pyautogui.moveTo(posx, posy, duration=0.1)
    pyautogui.click(posx, posy)
        # copy the quote
    posx, posy = 1270, 380
    pyautogui.moveTo(posx, posy, duration = 0.2)
    pyautogui.click(posx, posy)
    pyautogui.click(posx, posy)
    keyboard_copy() # https://www.bochk.com/tc/investment/econanalysis/infostock.html
        # current time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_quote = float(Tk().clipboard_get()) # get content from the clip board
    difference_ratio = (refer_close-current_quote)/current_quote*100    
        # print
    print( i, "\t", current_time, "\t", refer_close, current_quote, round(difference_ratio, 2) )
        #
    #switch_window()
    swtich_left_desktop()
        #
        # data visualization: file:///C:/Users/Admin/lines.html
    x.append( current_time )
    y1.append( refer_close )
    y2.append( current_quote )
    limit_len = 30
    if len(x)>limit_len:
        draw_lines_html(x[-limit_len:], y1[-limit_len:], y2[-limit_len:], str(i+1))
    else:
        draw_lines_html(x, y1, y2, str(i+1))
        # refresh webpage    
    posx, posy = 80, 50
    pyautogui.moveTo(posx, posy, duration=0.1)
    pyautogui.click(posx, posy)
        #write to file
    every_line = str(refer_close) + "\t" + str(current_time) + "\t" + str(current_quote) + "\t" + str(round(difference_ratio, 2))
    recording(every_line, file_path)
        # 
    time.sleep(8)
#
print("monitor finished")
