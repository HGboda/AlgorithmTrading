# -*- coding: UTF-8 -*-


import webbrowser
import csv
import os
from datetime import datetime,timedelta

import sqlite3
import pandas.io.sql as pd_sql
import pandas as pd

from IPython.html.widgets import interact, interactive, fixed
from IPython.html import widgets
from IPython.display import clear_output, display, HTML
pd.set_option('display.width',500)

import urllib
import time

import stockcore as stcore
import tradingalgo as talgo 
import data_mani as dmani


import time
import datetime as dt
import csv
import xlrd

import xlwt
from xlutils.copy import copy # http://pypi.python.org/pypi/xlutils
from xlrd import open_workbook # http://pypi.python.org/pypi/xlrd
from xlwt import easyxf # http://pypi.python.org/pypi/xlwt

import stockcore as stcore
import tradingalgo as talgo 
import data_mani as dmani
import ver2_0_i as v20
import sqlite3
import pandas.io.sql as pd_sql
import pandas as pd

def simulAuto(startNum,endNum):
    # book = xlrd.open_workbook("../../Kospi_symbols.xls")
    # sheet = book.sheet_by_name('kospi')
    # book = xlrd.open_workbook("../../Kosdaq_symbols.xls")
    # sheet = book.sheet_by_name('kosdaq')


    # data_df = pd.read_csv('../../SP500.csv')
    data_df = pd.read_csv('../../dowjonesIA.csv')
    # data_df = pd.read_csv('../../NASDAQComposite.csv')

    srcsite = 1#google
    # srcsite = 2#yahoo
    runcount = 1
    dbtradinghist = 'none'
    # histmode = 'histdb'
    histmode = 'none'
    plotly = 'plotly'
    stdmode = 'stddb'
    tangentmode = 'tangentdb'
    updbpattern = 'none'
    appenddb = 'none'
    writedblog = 'writedblog'
    startdatemode = 2
    daych = 0
    startdate = '2015-01-01'

    start = time.clock()
    name = u''
    code = 0        

    algo1_totalaccumgain = []
    algo1_totaltradinggain = []
    algo1_totaltradingnum = []        

    algo_title = []
    algo_stance = []    
    errortitles = []
    algo_benchmark = []
    totalsum = []
    algo_lastDate= []
    algo_macdsig = []
    algo_slowsig = []
    total_sum = 0
    total_cnt = 0
    print data_df.head()
    for cnt in range(startNum,endNum):
        # print data_df['Ticker'][cnt],data_df['Code'][cnt]
        try:
            symbol = data_df['Code'][cnt]
            title = data_df['Ticker'][cnt]
            print title,symbol
        except:
            errortitles.append(title)
            stcore.PrintException()
            pass       


        try:
            accumGain\
            ,tradingGain\
            ,tradingNum\
            ,tradingStance\
            ,lastbench\
            ,totalmoney\
            ,lastbuydate\
            ,macdsig,slowsig = v20.RunSimul_world(symbol,startdate)
            algo_title.append(title)
            algo1_totalaccumgain.append(accumGain)
            algo1_totaltradinggain.append(tradingGain)
            algo1_totaltradingnum.append(tradingNum)        
            algo_stance.append(tradingStance)
            algo_benchmark.append(lastbench)
            totalsum.append(totalmoney)
            algo_lastDate.append(lastbuydate)
            algo_macdsig.append(macdsig)
            algo_slowsig.append(slowsig)
            clear_output()
            if tradingNum != 0:
                total_cnt += 1
                total_sum = total_sum + totalmoney
        except:
            errortitles.append(title)
            stcore.PrintException()
            pass   
        # runcount += 1
        # if runcount >= 1:
        #     break

    elapsed = (time.clock() - start)
    print 'total run elapsed time:',elapsed,'total runcount:',runcount

    # print algo1_totalaccumgain
    # print 'stance1',stance_algo1,'stance2',stance_algo2
    tradingresultdf = pd.DataFrame({'title':algo_title,'algo1_accumgain':algo1_totalaccumgain,'algo1_tradinggain':algo1_totaltradinggain\
        ,'algo1_tradingnum':algo1_totaltradingnum\
        ,'totalsum':totalsum\
        ,'lastDate':algo_lastDate\
        ,'benchmark':algo_benchmark\
        ,'MACDSig':algo_macdsig\
        ,'SLOWSig':algo_slowsig\
        ,'stance':algo_stance}\
        )    
    # display(HTML('<font size=1>'+tradingresultdf._repr_html_()+'</font>'))
    display(HTML(tradingresultdf.to_html()))
    # display(HTML('<font size=2>'+tradingresultdf.to_html()+'</font>'))


    initial_capital = 10000000
    tnum = 99
    # print totalsum
    # print 'totalsum',sum(totalsum),(float(sum(totalsum))- float(initial_capital*tnum))/float(initial_capital*tnum)
    print 'totalsum',sum(totalsum),'total_cnt',total_cnt
    print 'totalsum',total_sum,(float(total_sum)- float(initial_capital*total_cnt))/float(initial_capital*total_cnt)
    print '-----error title-------'
    for title in errortitles:
        print title



