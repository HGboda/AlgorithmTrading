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
import os
import sys



def fetchData(code):

    financeurl = 'http://finance.naver.com/item/sise_day.nhn?code='+code+'&page=1'

    response = urllib2.urlopen(financeurl)
    content = response.read()
    # response.close()

    soup = BeautifulSoup(content)
#     print soup.tr.td.span
    # dates = soup.findAll("tr")[1]#[0].findAll('span')
    datelist = []
    dates = soup.findAll('td',attrs={'align':'center'})#[0].findAll('span')
    for date in dates:
        # print date.text
        datelist.append(date.text.replace('.','-'))

    closep = []    
    openp = []
    highp = []
    lowp = []
    volume = []        
    colcnt = 0
    contents = soup.findAll('td', attrs={'class':'num'})
    for content in contents:
        content.findAll('span')
        if colcnt == 0:
            closep.append(int(content.text.replace(',','')))
        elif colcnt == 2:
            openp.append(int(content.text.replace(',','')))
        elif colcnt == 3:
            highp.append(int(content.text.replace(',','')))
        elif colcnt == 4:
            lowp.append(int(content.text.replace(',','')))
        elif colcnt == 5:
            volume.append(int(content.text.replace(',','')))
        colcnt += 1
        if colcnt == 6:
            colcnt = 0
        # print content.text,colcnt
        # print content.text
    # print 'date',datelist        
    # print 'closep',closep
    # print 'openp',openp
    # print 'highp',highp
    # print 'lowp',lowp
    # print 'volume',volume
    
    datelen = len(datelist)
    datedflist = []
    for cnt in range(datelen):
        str1 = datelist[cnt].split('-')
        dtdate = datetime(int(str1[0]),int(str1[1]),int(str1[2]),0)
        datedflist.append(dtdate)

    # rtdf = pd.date_range(datedflist[0], datedflist[-1])    
    # print rtdf
    d={'Open':openp, 'High':highp,'Low':lowp,'Close':closep,'Volume':volume }
    # print d
    adjustdf = pd.DataFrame(d,index=datedflist)
    # print adjustdf
    adjustdf.index.name = 'Date'
    print 'fetchData adjustdf:',adjustdf.head()
    return adjustdf



def HistSaving(codearg,typearg,namearg,mode):

    code = codearg #'097950'#'005930' #'005380'#009540 #036570
    if typearg == 1:
        symbol = 'GOOG/KRX_'+code
    elif typearg == 2:
        symbol = 'GOOG/KOSDAQ_'+code
    elif typearg == 3:
        symbol = 'GOOG/INDEXKRX_KOSPI200'  
    # symbol = 'GOOG/INDEXKRX_KOSPI200'
    startdate = '2007-01-01'
    # enddate = '2008-12-30'
    print symbol,namearg
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
    histurl = 'http://ichart.yahoo.com/table.csv?s='+code+'.KS'+'&a='+str(startday.month-1)+\
    '&b='+str(startday.day)+'&c='+str(startday.year)+'&d='+str(today.month-1)+'&e='+str(today.day)+'&f='+str(today.year)+'&ignore=.csv'
    print histurl

    # response = urllib2.urlopen(histurl)
    # histdf = pd.read_csv(response)

    # datelen = len(histdf.Date)

    # for cnt in range(datelen):
    #     str1 = histdf.Date[cnt].split('-')
    #     dtdate = datetime(int(str1[0]),int(str1[1]),int(str1[2]),0)
    #     histdf.Date[cnt]= dtdate

    # histdf = histdf[histdf.Volume != 0]
    # histdf = histdf.drop('Adj Close',1)
    # histdf.index= histdf.Date
    # histdf.index.name = 'Date'
    # histdf = histdf.drop('Date',1)
    # print '----date adjust start---'
    # bars_new_unique = histdf[~histdf.index.isin(bars_org.index)]
    # bars_org = pd.concat([bars_org, bars_new_unique])
    # print bars_org.tail()
    # print '----date adjust end-----'

    ''' 
    naver scrape for yahoo ichart alternative
    '''
    histdf = fetchData(code) 
    histdf = histdf[histdf.Volume != 0]
    print histdf
    print '----date adjust start---'
    bars_new_unique = histdf[~histdf.index.isin(bars_org.index)]
    bars_org = pd.concat([bars_org, bars_new_unique])
    print bars_org.tail()
    print '----date adjust end-----'


    if typearg == 1:
        rtsymbol = code+'.KS'
    elif typearg == 2:        
        rtsymbol = code+'.KQ'
    elif typearg == 3:
        rtsymbol = '^KS200'
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

    return bars    



'''
pattern save
'''

#file delete


# if mode == 'one':
#     deletename = basepos+u'hist_db_'+codearg+u'_'+namearg+u'.sqlite'
#     if os.path.isfile(deletename):
#         os.remove(deletename)

    
print 'hist saving start'
basepos = u"./data/hist/"
import sqlite3
import pandas.io.sql as pd_sql

for codearg,namearg,bars in zip(codes,names,amr):
    dbname = 'hist_db_'+codearg+'_'+namearg+'.sqlite'
    con = sqlite3.connect("./data/hist/"+dbname)
    tablename_base = 'result_'+codearg+'_'+namearg

    # for cnt in range(dblen):
    tablename = tablename_base
    # print 'writetable:',tablename
    con.execute("DROP TABLE IF EXISTS "+tablename)

    bars2 = bars.reset_index()
    print 'bars2'
    print bars2.tail()
    pd_sql.write_frame(bars2, tablename, con)


# readlist = []    
# for cnt in range(dblen):
#     tablename = tablename_base+'_'+str(cnt)
#     # print 'readtable:',tablename
#     patterndf = pd_sql.read_frame("SELECT * from "+tablename, con)
#     readlist.append(PatternData(patterndf))
#     readlist[cnt].patterndf.index = readlist[cnt].patterndf['Date']
#     readlist[cnt].patterndf = readlist[cnt].patterndf.drop('Date',1)


# print 'read pattern:',readlist[0].patterndf
# print 'org patternAr:',patternAr_org[0].patterndf

# con.close()    
    
    
    # con.close()        
    
    # print 'histdb save done'

'''
hist auto saving
'''    
# import xlrd

# book = xlrd.open_workbook("symbols.xls")
# sheet = book.sheet_by_name('kospi')
# # sheet = book.sheet_by_name('kosdaq')
# length = sheet.nrows

# basepos = u"./data/hist/"
# filelist = [ f for f in os.listdir(basepos) if f.endswith(u".sqlite") ]
# for f in filelist:
#     deletename = basepos+f
#     if os.path.isfile(deletename):
#         os.remove(deletename)


# for rowcnt in range(3,5):
#     code = '{0:06d}'.format(int(sheet.row_values(rowcnt)[1]))
#     name = sheet.row_values(rowcnt)[2]
#     try:
#         RunSimul(str(code),1,name,'one')
#     except:
#         pass        