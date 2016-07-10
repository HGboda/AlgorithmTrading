%matplotlib inline
# from matplotlib import interactive
# interactive(True)

import numpy as np
import pylab as pl
import matplotlib
import csv
import time
import datetime as dt
from datetime import datetime,timedelta
from matplotlib.dates import date2num
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.finance import candlestick
import pylab 
import urllib2
import matplotlib.animation as animation 
import csv
import scipy.spatial as spatial
from time import mktime
from scipy.optimize import fsolve
from numpy import polyfit,poly1d
import scipy as sp
import pandas as pd
import Quandl
from pandas.io.data import DataReader
#from backtest import Strategy, Portfolio
from abc import ABCMeta, abstractmethod
import plotly
import talib
import statsmodels.api as sm
from copy import deepcopy
import urllib2
import re
from BeautifulSoup import BeautifulSoup
from lxml import etree
import lxml.html as LH
import sqlite3
import pandas.io.sql as pd_sql

pd.set_option('display.width',500)

def ReadHistFromDB(codearg,typearg,namearg,mode):
    print 'read hist data from DB'

    
    dbname = 'hist_db_'+codearg+'_'+namearg+'.sqlite'
    con = sqlite3.connect("../data/hist/"+dbname)

    query = "SELECT * FROM sqlite_master WHERE type='table'"
    df = pd.io.sql.read_frame(query,con)

    tablelen = len(df)
    print 'hist tablelen:',tablelen    
    tablename = 'result_'+codearg+'_'+namearg

    histdf = pd_sql.read_frame("SELECT * from "+tablename, con)
    
    from pandas.lib import Timestamp
    histdf.Date = histdf.Date.apply(Timestamp)
    histdf2 = histdf.set_index('Date')

    # histdf.index = histdf['Date']
    # histdf = histdf.drop('Date',1)
    print 'histdf from db:'
    print histdf2.head()
    print 'hist index type:',type(histdf2.index)
    con.close()
    return histdf2


def fetchHistData(codearg,namearg,symbol,startdate):
    print 'fetchHistData'
    dbname = 'hist_db_'+codearg+'_'+namearg+'.sqlite'
    con = sqlite3.connect("../data/hist/"+dbname)

    query = "SELECT * FROM sqlite_master WHERE type='table'"
    df = pd.io.sql.read_frame(query,con)

    tablelen = len(df)
    print 'hist tablelen:',tablelen    
    tablename = 'result_'+codearg+'_'+namearg

    histdf = pd_sql.read_frame("SELECT * from "+tablename, con)
    
    from pandas.lib import Timestamp
    histdf.Date = histdf.Date.apply(Timestamp)
    histdf2 = histdf.set_index('Date')

    histdf2 = histdf2[histdf2.index >= startdate]
    # histdf.index = histdf['Date']
    # histdf = histdf.drop('Date',1)
    print 'histdf from db:'
    print histdf2.head()
    print histdf2.tail()
    print 'hist index type:',type(histdf2.index)
    con.close()
    return histdf2


def fetchRealData(code,symbol,typearg,startdate):
    bars_org =  Quandl.get(symbol, collapse='Daily', trim_start=startdate, trim_end=datetime.today(),authtoken="")
    # bars = Quandl.get(symbol, collapse='Daily', trim_start=startdate, trim_end=enddate,authtoken="")
    # print bars[-10:]
    print bars_org.tail()
    print '---------'
    #print len(bars)

    
    today = datetime.today()
    startday = today- timedelta(days=7 )
    # print today.year,today.month,today.day
    # print startday.year,startday.month,startday.day
    if typearg == 3:
        histurl = 'http://ichart.yahoo.com/table.csv?s=^KS11'+'&a='+str(startday.month-1)+\
        '&b='+str(startday.day)+'&c='+str(startday.year)+'&d='+str(today.month-1)+'&e='+str(today.day)+'&f='+str(today.year)+'&ignore=.csv'
        print histurl
    else:
        histurl = 'http://ichart.yahoo.com/table.csv?s='+code+'.KS'+'&a='+str(startday.month-1)+\
        '&b='+str(startday.day)+'&c='+str(startday.year)+'&d='+str(today.month-1)+'&e='+str(today.day)+'&f='+str(today.year)+'&ignore=.csv'
        print histurl
    '''
    yahoo scrape api 
    '''
    response = urllib2.urlopen(histurl)
    histdf = pd.read_csv(response)

    datelen = len(histdf.Date)

    for cnt in range(datelen):
        str1 = histdf.Date[cnt].split('-')
        dtdate = datetime(int(str1[0]),int(str1[1]),int(str1[2]),0)
        histdf.Date[cnt]= dtdate

    histdf = histdf[histdf.Volume != 0]
    histdf = histdf.drop('Adj Close',1)
    histdf.index= histdf.Date
    histdf.index.name = 'Date'
    histdf = histdf.drop('Date',1)
    print '----date adjust start---'
    bars_new_unique = histdf[~histdf.index.isin(bars_org.index)]
    bars_org = pd.concat([bars_org, bars_new_unique])
    print bars_org.tail()
    print '----date adjust end-----'

    ''' 
    naver scrape for yahoo ichart alternative
    '''
    # histdf = fetchData(code) 
    # histdf = histdf[histdf.Volume != 0]
    # print histdf
    # print '----date adjust start---'
    # bars_new_unique = histdf[~histdf.index.isin(bars_org.index)]
    # bars_org = pd.concat([bars_org, bars_new_unique])
    # print bars_org.tail()
    # print '----date adjust end-----'
    return bars_org

def RunSimul(codearg,typearg,namearg,mode,dbmode,histmode,runcount,srcsite):

    code = codearg #'097950'#'005930' #'005380'#009540 #036570
    if srcsite == 1:
        if typearg == 1:
            symbol = 'GOOG/KRX_'+code
        elif typearg == 2:
            symbol = 'GOOG/KOSDAQ_'+code
        elif typearg == 3:
            symbol = 'GOOG/INDEXKRX_KOSPI200'  
    elif srcsite == 2:
        if typearg == 1:
            symbol = 'YAHOO/KS_'+code
        elif typearg == 2:
            symbol = 'YAHOO/KQ_'+code
        elif typearg == 3:
            symbol = 'YAHOO/INDEX_KS11'  
    # symbol = 'GOOG/INDEXKRX_KOSPI200'
    startdate = '2014-01-01'
    # enddate = '2008-12-30'
    print symbol,namearg
    if mode == 'realtime':
        if histmode == 'none':
            bars_org = fetchRealData(code,symbol,typearg,startdate)
        elif histmode == 'histdb':
            bars_org = fetchHistData(codearg,namearg,symbol,startdate)
    elif mode =='dbpattern':
        bars_org = ReadHistFromDB(codearg,typearg,namearg,mode)    

    if typearg == 1:
        rtsymbol = code+'.KS'
    elif typearg == 2:        
        rtsymbol = code+'.KQ'
    elif typearg == 3:
        rtsymbol = '^KS11'#'^KS200'
    # rtsymbol = '^KS200'
    realtimeURL = 'http://finance.yahoo.com/d/quotes.csv?s='+rtsymbol+'&f=sl1d1t1c1ohgv&e=.csv'
    print realtimeURL
    response = urllib2.urlopen(realtimeURL)
    cr = csv.reader(response)
    for row in cr:
        rtsymbol = row[0]
        rtclose = float(row[1])
        rtdate = row[2]
        rttime = row[3]
        rtchange = float(row[4])
        rtopen = float(row[5])
        rthigh = float(row[6])
        rtlow = float(row[7])
        rtvolume = float(row[8])
        print 'rtsymbol:',rtsymbol,'rtclose:',rtclose,rtdate,rttime,rtchange,'rtopen:',rtopen,'rthigh:',rthigh,'rtlow:',rtlow,'rtvolume:',rtvolume

    # print date2num(datetime.strptime(rtdate.replace('/',' '),'%m %d %Y'))

    # print bars.index[-1]
    # print date_object > bars.index[-1]
    # date_object  = date_object- dt.timedelta(days=1) 
    # print date_object > bars.index[-1]
    date_object = datetime.strptime(rtdate.replace('/',' '), '%m %d %Y')
    rtdf = pd.date_range(date_object, date_object)

    date_append = False
    # print len(bars_org),len(bars_org['Close']),len(bars_org['Volume'])
    if date_object > bars_org.index[-1]:
        d={'Open':rtopen, 'High':rthigh,'Low':rtlow,'Close':rtclose,'Volume':rtvolume }
        appenddf = pd.DataFrame(d,index=rtdf)
        appenddf.index.name = 'Date'
        date_append = True
        print appenddf,date_append
        bars = pd.concat([bars_org,appenddf])
        print '----------'
        print bars.tail()
    else:
        bars = bars_org


    # varout = talib.VAR(bars['Close'].values,5,1)
    # vardf = pd.DataFrame(varout,index = bars.index,columns=['VAR'])    

    fig = plt.figure(figsize=(20, 10))

    fig.patch.set_facecolor('white')     # Set the outer colour to white
    ax1 = fig.add_subplot(211,  ylabel='Price in $')

    # # Plot the AAPL closing price overlaid with the moving averages
    bars['Close'].plot(ax=ax1, color='r', lw=2.)
    # ax2 = fig.add_subplot(212,ylabel='VAR')
    # vardf.plot(ax=ax2, color='r', lw=2.)

    print '50 std:',bars['Close'][-50:].std()
    print '40 std:',bars['Close'][-40:].std()
    print '30 std:',bars['Close'][-30:].std()
    print '20 std:',bars['Close'][-20:].std()
    print '10 std:',bars['Close'][-10:].std()

