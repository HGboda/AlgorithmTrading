# %matplotlib inline

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
# import plotly
from abc import ABCMeta, abstractmethod
# import talib
import statsmodels.api as sm
from copy import deepcopy
import urllib2
import re
from BeautifulSoup import BeautifulSoup
from lxml import etree
import lxml.html as LH
import sqlite3
import pandas.io.sql as pd_sql
import os,sys
from pykalman import UnscentedKalmanFilter
import linecache
from IPython.html.widgets import interact, interactive, fixed
from IPython.html import widgets
from IPython.display import clear_output, display, HTML

# from guiqwt.plot import CurveDialog
# from guiqwt.builder import make


global office
office = False


pd.set_option('display.width',500)

class PatternData:
    def __init__(self,df):
        self.patterndf = df

class PatternExtractData:
    def __init__(self,df,idnum,parentid,targetpatNum):
        self.patterndf = df
        self.patternid = idnum
        self.foundnum = 0
        self.parentid = parentid
        self.targetpatNum = targetpatNum
    def setFoundCount(self,num):
        self.foundnum = num

    def getFoundCount(self):
        return self.foundnum


class Deque:
    def __init__(self):
        self.items = []
        self.front = 0
        self.rear = 0

    def getValue(self,index):
        if self.size() > 0:
            return self.items[index]       
        else:
            return 'NA'
    
    def isEmpty(self):
        return self.items == []

    def addFront(self, item):
        self.items.append(item)

    def addRear(self, item):
        self.items.insert(0,item)

    def removeFront(self):
        return self.items.pop()

    def removeRear(self):
        return self.items.pop(0)

    def size(self):
        return len(self.items)






def movingaverage(values,window):
    weights = np.repeat(1.0, window)/window
    smas = np.convolve(values, weights, 'valid')
    return smas


class Strategy(object):
    """Strategy is an abstract base class providing an interface for
    all subsequent (inherited) trading strategies.

    The goal of a (derived) Strategy object is to output a list of signals,
    which has the form of a time series indexed pandas DataFrame.

    In this instance only a single symbol/instrument is supported."""

    __metaclass__ = ABCMeta

    @abstractmethod
    def generate_signals(self):
        """An implementation is required to return the DataFrame of symbols 
        containing the signals to go long, short or hold (1, -1 or 0)."""
        raise NotImplementedError("Should implement generate_signals()!")
        
# backtest.py

class Portfolio(object):
    """An abstract base class representing a portfolio of 
    positions (including both instruments and cash), determined
    
    on the basis of a set of signals provided by a Strategy."""

    __metaclass__ = ABCMeta

    @abstractmethod
    def generate_positions(self):
        """Provides the logic to determine how the portfolio 
        positions are allocated on the basis of forecasting
        signals and available cash."""
        raise NotImplementedError("Should implement generate_positions()!")

    @abstractmethod
    def backtest_portfolio(self):
        """Provides the logic to generate the trading orders
        and subsequent equity curve (i.e. growth of total equity),
        as a sum of holdings and cash, and the bar-period returns
        associated with this curve based on the 'positions' DataFrame.

        Produces a portfolio object that can be examined by 
        other classes/functions."""
        raise NotImplementedError("Should implement backtest_portfolio()!")

class MovingAverageCrossStrategy(Strategy):
    """    
    Requires:
    symbol - A stock symbol on which to form a strategy on.
    bars - A DataFrame of bars for the above symbol.
    short_window - Lookback period for short moving average.
    long_window - Lookback period for long moving average."""

    def __init__(self, symbol, bars, short_window=100, long_window=400):
        self.symbol = symbol
        self.bars = bars

        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self):
        """Returns the DataFrame of symbols containing the signals
        to go long, short or hold (1, -1 or 0)."""
        signals = pd.DataFrame(index=self.bars.index)
        signals['signal'] = 0.0

        # Create the set of short and long simple moving averages over the 
        # respective periods
        signals['short_mavg'] = pd.rolling_mean(self.bars['Close'], self.short_window, min_periods=1)
        signals['long_mavg'] = pd.rolling_mean(self.bars['Close'], self.long_window, min_periods=1)

        # Create a 'signal' (invested or not invested) when the short moving average crosses the long
        # moving average, but only for the period greater than the shortest moving average window
        signals['signal'][self.short_window:] = np.where(signals['short_mavg'][self.short_window:] 
            > signals['long_mavg'][self.short_window:], 1.0, 0.0)   

        # signal :1 -> short moving average is bigger than long moveing average
        # signal :0

        # Take the difference of the signals in order to generate actual trading orders
        signals['positions'] = signals['signal'].diff()   
        # positions 1: buy signal
        # positions -1: sell signal

        #print signals.index
        #print signals.head()
        #print signals['signal']
        #print signals.ix[signals.positions == -1.0].index
        #print signals.ix[signals.positions == 1.0].index
        #print signals.ix[signals.signal == 0.0].index
        #print signals.ix[signals.signal == 1.0].index
        return signals

class MarketOnClosePortfolio(Portfolio):
    """Encapsulates the notion of a portfolio of positions based
    on a set of signals as provided by a Strategy.

    Requires:
    symbol - A stock symbol which forms the basis of the portfolio.
    bars - A DataFrame of bars for a symbol set.
    signals - A pandas DataFrame of signals (1, 0, -1) for each symbol.
    initial_capital - The amount in cash at the start of the portfolio."""

    def __init__(self, symbol, bars, signals, initial_capital=1000000.0):
        self.symbol = symbol        
        self.bars = bars
        self.signals = signals
        self.initial_capital = float(initial_capital)
        self.positions = self.generate_positions()
        
    def generate_positions(self):
        #print signals.head()
        #print signals.index
        positions = pd.DataFrame(index=self.signals.index).fillna(0.0)
        positions[self.symbol] = 10*self.signals['signal']   # This strategy buys 10 shares
        #positions['test'] has 10 shares * (1 or 0)
        return positions
                    
    def backtest_portfolio(self):
        #print len(self.positions),len(self.bars['Close'])
        portfolio = self.positions*self.bars['Close']
        #portfolio has (10 shares or 0) * price 

        pos_diff = self.positions.diff()
        # if sell signal, pos_diff has sell shares for example -10 shares
        # if buy signal, pos_diff has buy shares for example 10 shares

        portfolio['holdings'] = (self.positions*self.bars['Close']).sum(axis=1)
        # holdings means the current share price if in buy position
        portfolio['cash'] = self.initial_capital - (pos_diff*self.bars['Close']).sum(axis=1).cumsum()
        # if buy signal, cash has initial capital plus the current remaining money 
        # if sell signal, cash has return money plus initial capital

        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        portfolio['returns'] = portfolio['total'].pct_change()
        #print portfolio.tail()
        #print portfolio.head()
        return portfolio

class MarketOnMixedPortfolio(Portfolio):
    
    def __init__(self, symbol, ms, initial_capital=1000000.0):
        self.symbol = symbol        
        self.ms = ms
        self.signals = ms['newsignals']
        self.initial_capital = float(initial_capital)
        self.positions = self.generate_positions()
        
    def generate_positions(self):
        
        positions = pd.DataFrame(index=self.signals.index).fillna(0.0)
        
        # positions[self.symbol] = 100*self.signals   # This strategy buys 100 shares
        positions[self.symbol] = self.signals   # This strategy buys 100 shares
        #positions['test'] has 10 shares * (1 or 0)
        # print positions['2012-05-10':'2012-06-10']
        
        return positions
                    
    def backtest_portfolio(self):
        #print len(self.positions),len(self.ms['Price'])
        try:
            portfolio = self.positions*self.ms['Price']
            # display(HTML(portfolio.to_html()))
            #portfolio has (10 shares or 0) * price 
            portfolio['Price'] = 0
            portfolio['Price'] = self.ms['Price']
            portfolio['tsig'] = 0
            portfolio['tsig'] = self.positions
            portfolio['tsig'] = portfolio['tsig'].diff()

            stance = 'none'
            portfolio['cash'] = 0
            portfolio['holdings'] = 0
            portfolio['tnum'] = 0
            portfolio['buysignal'] = 0
            portfolio['sellsignal'] = 0
            tradecnt = 0
            buyday = 0
            lastcash = 0
            hold_shareNum = 0

            # display(HTML(portfolio.to_html()))
            for inday in range(0,len(portfolio),1):

                if tradecnt == 0 and stance == 'none':
                    portfolio['cash'][inday] = self.initial_capital                
                    portfolio['holdings'][inday] = 0
                    portfolio['tnum'][inday] = 0
                    


                if tradecnt > 0:
                    if portfolio['tsig'][inday] == 0 and stance == 'none':
                        portfolio['holdings'][inday] = 0
                        portfolio['cash'][inday] = portfolio['cash'][inday-1]
                        portfolio['tnum'][inday] = 0

                    

                if portfolio['tsig'][inday] == 0 and stance == 'hold':
                    portfolio['holdings'][inday] = hold_shareNum *  portfolio['Price'][inday]
                    portfolio['cash'][inday] = portfolio['cash'][inday-1]
                    portfolio['tnum'][inday] = hold_shareNum

                if portfolio['tsig'][inday] == 1:
                    stance = 'hold'
                    portfolio['buysignal'][inday] = 100
                    # if tradecnt == 0:
                    #     hold_shareNum = int(self.initial_capital / portfolio['Price'][inday])
                    #     portfolio['holdings'][inday] = hold_shareNum *  portfolio['Price'][inday]
                    #     portfolio['cash'][inday] = self.initial_capital - hold_shareNum*portfolio['Price'][inday]
                    #     portfolio['tnum'][inday] = hold_shareNum
                    #     print portfolio.index[inday],portfolio['cash'][inday-1],hold_shareNum
                    # elif tradecnt > 0:

                    hold_shareNum = int(portfolio['cash'][inday-1] / portfolio['Price'][inday])
                    portfolio['holdings'][inday] = hold_shareNum *  portfolio['Price'][inday]
                    portfolio['cash'][inday] = portfolio['cash'][inday-1] - hold_shareNum*portfolio['Price'][inday]
                    portfolio['tnum'][inday] = hold_shareNum
                    # print portfolio.index[inday],portfolio['cash'][inday-1],hold_shareNum

                    tradecnt += 1    
                    buyday = inday
                if portfolio['tsig'][inday] == -1:
                    stance = 'none'
                    portfolio['sellsignal'][inday] = -100
                    portfolio['cash'][inday] = (hold_shareNum *  portfolio['Price'][inday]) + portfolio['cash'][inday-1]
                    portfolio['holdings'][inday] = 0
                    portfolio['tnum'][inday] = 0
                    lastcash = portfolio['cash'][inday]

            
            """
            pos_diff = self.positions.diff()
            # if sell signal, pos_diff has sell shares for example -10 shares
            # if buy signal, pos_diff has buy shares for example 10 shares
            # print pos_diff['2012-05-10':'2012-06-10']
            portfolio['buysignal'] = pos_diff[pos_diff == 100]
            portfolio['buysignal'].fillna(0.0)
            portfolio['sellsignal'] = pos_diff[pos_diff == -100]
            portfolio['sellsignal'].fillna(0.0)
            # print portfolio[portfolio.buysignal == 100] ,portfolio[portfolio.sellsignal == -100]
            portfolio['holdings'] = (self.positions*self.ms['Price']).sum(axis=1)
            # holdings means the current share price if in buy position
            portfolio['cash'] = self.initial_capital - (pos_diff*self.ms['Price']).sum(axis=1).cumsum()
            # if buy signal, cash has initial capital plus the current remaining money 
            # if sell signal, cash has return money plus initial capital
            # print portfolio.ix['2012-08-05':'2012-08-20']
            """
            portfolio['total'] = portfolio['cash'] + portfolio['holdings']
            portfolio['returns'] = portfolio['total'].pct_change()
            # display(HTML(portfolio.to_html()))
        except Exception,e:
            PrintException()
        #print portfolio.tail()
        #print portfolio.head()
        return portfolio




def readlastTradingDay(codearg,namearg,dbpath):
    print 'read last trading day from DB'

    import glob
    if dbpath == 1:
        lists = glob.glob("../../data/result/*.sqlite")
    elif dbpath ==2:
        lists = glob.glob("../../data/result2/*.sqlite")
    elif dbpath ==3:
        lists = glob.glob("../../data/result2/kosdaq/*.sqlite")    

    dbname = lists[-1] 

    dbdate = dbname.split('.sqlite')[0]
    dbdate = dbdate.split('_')[2]

    today = str(datetime.today()).split(' ')[0]
    today = today.replace('-','')
    if today == dbdate:
        dbname = lists[-2] 
        print 'dbname change'

    print 'readlastTradingDay dbname',dbname
    # dbname = '../../data/result/result_db_20140731.sqlite'
    # print 'dbname',dbname
    con = sqlite3.connect(dbname)
    query = "SELECT * FROM sqlite_master WHERE type='table'"

    dbdate = dbname.split('.sqlite')[0]
    dbdate = dbdate.split('_')[2]
    df = pd_sql.read_frame(query,con)
    
    tablelen = len(df['name'])
    searchname = 'result_'+str(codearg)+'_'+namearg+'_signal_'
    print 'search table name',searchname
    for cnt in range(tablelen):
        if  searchname in df['name'][cnt]:
            tablename = df['name'][cnt]
            tabledf = pd.io.sql.read_frame("SELECT * from "+tablename, con)    
            break

    con.close()        
    if len(tabledf) > 0:
        # ltradingdate = tabledf[-1:]['Date'].values
        # ltdate = ltradingdate[0].split(' ')[0]
        # ltstance = tabledf[-1:]['Stance'].values[0]
        # return ltdate,ltstance
        return tabledf
    else:
        return -1



def dateParser(s):
    #return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")
    return datetime(int(s[0:4]), int(s[5:7]), int(s[8:10]))*1000
    #return np.datetime64(s)
    #return pandas.Timestamp(s, "%Y-%m-%d %H:%M:%S.%f", tz='utc' )



def ReadPatternsAppendFromDB(codearg,namearg):
    print 'read append pattern DB'

    basepos = u"../../data/pattern/append/"

    patname = basepos+u'pattern_db_'+codearg+u'_'+namearg+u'.sqlite'
    readlist = []    
    if os.path.isfile(patname):
        
        con = sqlite3.connect(patname)

        query = "SELECT * FROM sqlite_master WHERE type='table'"
        df = pd.io.sql.read_frame(query,con)

        tablelen = len(df)
        print 'read tablelen:',tablelen    
        tablename_base = 'result_'+codearg+'_'+namearg
        
        for cnt in range(tablelen):
            tablename = tablename_base+'_'+str(cnt)
            # print 'readtable:',tablename
            patterndf = pd_sql.read_frame("SELECT * from "+tablename, con)
            readlist.append(PatternData(patterndf))
            readlist[cnt].patterndf.index = readlist[cnt].patterndf['Date']
            readlist[cnt].patterndf = readlist[cnt].patterndf.drop('Date',1)

        con.close()    
    elif not os.path.isfile(patname):
        print 'no down trend db pattern'
        return -1
    print 'append db ok!!'        
    return readlist

def patternCompareAppend(gpatternAr,gbarsdf):

    # global gpatternAr
    lpatternAr = gpatternAr
    totallen = len(gpatternAr)
    
    curpat = gbarsdf.reset_index()

    corrclosex1 = curpat['Close'][-10:].pct_change().cumsum()
    closex1 = pd.DataFrame(corrclosex1,columns=['ClosePc']).fillna(0.0)
    closex1 = closex1.reset_index()
    
    
    corropenx1 = curpat['Open'][-10:].pct_change().cumsum()
    openx1 = pd.DataFrame(corropenx1,columns=['OpenPc']).fillna(0.0)
    openx1 = openx1.reset_index()
    
    
    corrhighx1 = curpat['High'][-10:].pct_change().cumsum()
    highx1 = pd.DataFrame(corrhighx1,columns=['HighPc']).fillna(0.0)
    highx1 = highx1.reset_index()
    

    corrlowx1 = curpat['Low'][-10:].pct_change().cumsum()
    lowx1 = pd.DataFrame(corrlowx1,columns=['LowPc']).fillna(0.0)
    lowx1 = lowx1.reset_index()
    

    corrvolx1 = curpat['Volume'][-10:].pct_change().cumsum()
    volx1 = pd.DataFrame(corrvolx1,columns=['VolPc']).fillna(0.0)
    volx1 = volx1.reset_index()
    
    
    for num in range(len(lpatternAr)):

        pattern0 = lpatternAr[num].patterndf.reset_index()
        
        pattern0_closepc = pattern0['Close'].pct_change().cumsum()   
        closepc0 = pd.DataFrame(pattern0_closepc,index = pattern0.index,columns=['ClosePc']).fillna(0.0)
        closecorr = closex1['ClosePc'].corr(closepc0['ClosePc'])
        if closecorr <= 0.8:
            continue
        
        pattern0_openpc = pattern0['Open'].pct_change().cumsum()   
        openpc0 = pd.DataFrame(pattern0_openpc,index = pattern0.index,columns=['OpenPc']).fillna(0.0)

        
        pattern0_highpc = pattern0['High'].pct_change().cumsum()   
        highpc0 = pd.DataFrame(pattern0_highpc,index = pattern0.index,columns=['HighPc']).fillna(0.0)

        pattern0_lowpc = pattern0['Low'].pct_change().cumsum()   
        lowpc0 = pd.DataFrame(pattern0_lowpc,index = pattern0.index,columns=['LowPc']).fillna(0.0)
        
        pattern0_volpc = pattern0['Volume'].pct_change().cumsum()   
        volpc0 = pd.DataFrame(pattern0_volpc,index = pattern0.index,columns=['VolPc']).fillna(0.0)
        
        # patternall = pd.concat([pattern0,openpc0,closepc0,highpc0,lowpc0,volpc0],axis=1)
    
        
        opencorr = openx1['OpenPc'].corr(openpc0['OpenPc'])    
        highcorr = highx1['HighPc'].corr(highpc0['HighPc'])    
        lowcorr = lowx1['LowPc'].corr(lowpc0['LowPc'])
        volcorr = volx1['VolPc'].corr(volpc0['VolPc'])

        # print 'pattern append close corr:',closecorr,' open corr:',opencorr,' high corr:',highcorr,' low corr:',lowcorr,' vol corr:',volcorr

        if np.isnan(volcorr):
            if closecorr > 0.9 \
                and opencorr > 0.9\
                and highcorr > 0.9\
                and lowcorr > 0.9:
                return num
                break    
        else:                
            if closecorr > 0.9 \
                and volcorr > 0.9\
                and opencorr > 0.9\
                and highcorr > 0.9\
                and lowcorr > 0.9:
            # corrsum = (closecorr+opencorr+highcorr+lowcorr+volcorr)/5
            # if corrsum > 0.5:
                # print 'found pattern :',num
                return num
                break
    # print 'loop end!!'    
    return -1


        

def patternCompare(gpatternAr,gbarsdf,gextractid):
    # print 'pattern matching'

    # global gpatternAr
    lpatternAr = gpatternAr
    totallen = len(gpatternAr)
    # global gbarsdf
    # print lpatternAr[num].patterndf
    # print(num,showaccum)
    
    curpat = gbarsdf.reset_index()
    # print curpat[-10:]
    # print curpat['Close'][-10:]
    # print curpat['Close'][-20:]
    corrclosex1 = curpat['Close'][-10:].pct_change().cumsum()
    closex1 = pd.DataFrame(corrclosex1,columns=['ClosePc']).fillna(0.0)
    closex1 = closex1.reset_index()
    
    
    corropenx1 = curpat['Open'][-10:].pct_change().cumsum()
    openx1 = pd.DataFrame(corropenx1,columns=['OpenPc']).fillna(0.0)
    openx1 = openx1.reset_index()
    
    
    corrhighx1 = curpat['High'][-10:].pct_change().cumsum()
    highx1 = pd.DataFrame(corrhighx1,columns=['HighPc']).fillna(0.0)
    highx1 = highx1.reset_index()
    

    corrlowx1 = curpat['Low'][-10:].pct_change().cumsum()
    lowx1 = pd.DataFrame(corrlowx1,columns=['LowPc']).fillna(0.0)
    lowx1 = lowx1.reset_index()
    

    corrvolx1 = curpat['Volume'][-10:].pct_change().cumsum()
    volx1 = pd.DataFrame(corrvolx1,columns=['VolPc']).fillna(0.0)
    volx1 = volx1.reset_index()
    
    # global gextractid
    for num in gextractid:
    # for num in range(totallen):

        pattern0 = lpatternAr[num].patterndf.reset_index()
        
        pattern0_openpc = pattern0['Open'].pct_change().cumsum()   
        openpc0 = pd.DataFrame(pattern0_openpc,index = pattern0.index,columns=['OpenPc']).fillna(0.0)
        opencorr = openx1['OpenPc'].corr(openpc0['OpenPc'])    
        if opencorr <= 0.9:
            continue

        pattern0_closepc = pattern0['Close'].pct_change().cumsum()   
        closepc0 = pd.DataFrame(pattern0_closepc,index = pattern0.index,columns=['ClosePc']).fillna(0.0)
        closecorr = closex1['ClosePc'].corr(closepc0['ClosePc'])
        # if closecorr <= 0.6:
        #     continue
        
        pattern0_highpc = pattern0['High'].pct_change().cumsum()   
        highpc0 = pd.DataFrame(pattern0_highpc,index = pattern0.index,columns=['HighPc']).fillna(0.0)

        pattern0_lowpc = pattern0['Low'].pct_change().cumsum()   
        lowpc0 = pd.DataFrame(pattern0_lowpc,index = pattern0.index,columns=['LowPc']).fillna(0.0)
        
        pattern0_volpc = pattern0['Volume'].pct_change().cumsum()   
        volpc0 = pd.DataFrame(pattern0_volpc,index = pattern0.index,columns=['VolPc']).fillna(0.0)
        
        # patternall = pd.concat([pattern0,openpc0,closepc0,highpc0,lowpc0,volpc0],axis=1)
    
        

        highcorr = highx1['HighPc'].corr(highpc0['HighPc'])    
        lowcorr = lowx1['LowPc'].corr(lowpc0['LowPc'])
        volcorr = volx1['VolPc'].corr(volpc0['VolPc'])

        # print 'close corr:',closecorr,' open corr:',opencorr,' high corr:',highcorr,' low corr:',lowcorr,' vol corr:',volcorr
        if np.isnan(volcorr):
            if closecorr > 0.9 \
                and opencorr > 0.9\
                and highcorr > 0.9\
                and lowcorr > 0.9:
                return num
                break    
        else:                
            if closecorr > 0.9 \
                and volcorr > 0.9\
                and opencorr > 0.9\
                and highcorr > 0.9\
                and lowcorr > 0.9:
            # corrsum = (closecorr+opencorr+highcorr+lowcorr+volcorr)/5
            # if corrsum > 0.5:
                # print 'found pattern :',num
                return num
                break
    # print 'loop end!!'    
    return -1

def PatternSave(barsdf,mmsigdf,mmsignp):

    totallen = len(barsdf['Close'])
    npclose = barsdf['Close'].values
    npopen = barsdf['Open'].values
    nphigh = barsdf['High'].values
    nplow = barsdf['Low'].values
    npvol = barsdf['Volume'].values
    benchgain = barsdf['Close'].pct_change().cumsum()   
    benchgainnp =benchgain.values

    print 'barsdf len',len(barsdf['Close']),'mmsigdf len',len(mmsigdf['MMSignals'])

    stance = 'none'
    negsig = np.zeros(totallen)
    possig = np.zeros(totallen)
    negnum = 0
    posnum = 0
    patternAr = []
    debugneg = 0

    for daycnt in range(totallen):
        # print barsdf.index[daycnt],npclose[daycnt],benchgainnp[daycnt]
        if mmsignp[daycnt] == 1 :
            if stance == 'none':
                startgain = benchgainnp[daycnt]
                stance = 'holding'
                # print 'holding:',barsdf.index[daycnt],npclose[daycnt],benchgainnp[daycnt]
        elif mmsignp[daycnt] == 0 and stance == 'holding':
            periodgain = benchgainnp[daycnt] - startgain
            stance = 'none'
            if periodgain <0 :
                negnum +=1
                negsig[daycnt] = 1
                if daycnt > 10:
                    # print npclose[daycnt-9:daycnt+1],npclose[daycnt]
                    # debugneg += 1
                    # patternAr.append(PatternData(npopen[daycnt-9:daycnt+1], npclose[daycnt-9:daycnt+1],nphigh[daycnt-9:daycnt+1],nplow[daycnt-9:daycnt+1],npvol[daycnt-9:daycnt+1]))                      
                    # if debugneg == 1:                    
                    #     print barsdf[daycnt-9:daycnt+1]
                    patternAr.append(PatternData(barsdf[daycnt-10:daycnt]))
            else:
                posnum +=1
                possig[daycnt] = 1
                # if daycnt > 10:
                    # patternAr.append(PatternData(barsdf[daycnt-9:daycnt+1]))
            # print 'gain:',periodgain,barsdf.index[daycnt],npclose[daycnt],benchgainnp[daycnt]\
            #     ,'negnum:',negnum,' posnum:',posnum

    negsigdf = pd.DataFrame(negsig,index = barsdf.index,columns=['Negsig']).fillna(0.0)
    negsigdf.index.name = 'Date'    

    possigdf = pd.DataFrame(possig,index = barsdf.index,columns=['Possig']).fillna(0.0)
    possigdf.index.name = 'Date'    
    # print patternAr[0].open,patternAr[0].close,patternAr[1].close
    # print patternAr[0].patterndf,patternAr[1].patterndf

    # patterndf = pd.DataFrame({'Open':patternAr[0].open,'Close':patternAr[0].close}).fillna(0.0)
    # patterndf.index.name = 'Date'    
    # print len(patternAr)
    # print patterndf


    return negsigdf,possigdf,patternAr


def PatternSaveUp(barsdf,mmsigdf,mmsignp):

    totallen = len(barsdf['Close'])
    npclose = barsdf['Close'].values
    npopen = barsdf['Open'].values
    nphigh = barsdf['High'].values
    nplow = barsdf['Low'].values
    npvol = barsdf['Volume'].values
    benchgain = barsdf['Close'].pct_change().cumsum()   
    benchgainnp =benchgain.values

    print 'barsdf len',len(barsdf['Close']),'mmsigdf len',len(mmsigdf['MMSignals'])

    stance = 'none'
    negsig = np.zeros(totallen)
    possig = np.zeros(totallen)
    negnum = 0
    posnum = 0
    patternAr = []
    debugneg = 0

    startday = 0

    for daycnt in range(totallen):
        # print barsdf.index[daycnt],npclose[daycnt],benchgainnp[daycnt]
        if daycnt > 10:

            if mmsignp[daycnt] == 1 :
                if stance == 'none':
                    startgain = benchgainnp[daycnt]
                    stance = 'holding'
                    startday = daycnt
                    # print 'holding:',barsdf.index[daycnt],npclose[daycnt],benchgainnp[daycnt]
            elif mmsignp[daycnt] == 0 and stance == 'holding':
                periodgain = benchgainnp[daycnt] - startgain
                stance = 'none'
                if periodgain <= 0:
                    negnum +=1
                    negsig[daycnt] = 1
                    # if daycnt > 10:
                        # print npclose[daycnt-9:daycnt+1],npclose[daycnt]
                        # debugneg += 1
                        # patternAr.append(PatternData(npopen[daycnt-9:daycnt+1], npclose[daycnt-9:daycnt+1],nphigh[daycnt-9:daycnt+1],nplow[daycnt-9:daycnt+1],npvol[daycnt-9:daycnt+1]))                      
                        # if debugneg == 1:                    
                        #     print barsdf[daycnt-9:daycnt+1]
                        # patternAr.append(PatternData(barsdf[daycnt-10:daycnt]))
                elif periodgain > 0:
                    posnum +=1
                    possig[daycnt] = 1
                    # patternAr.append(PatternData(barsdf[daycnt-10:daycnt]))
                    patternAr.append(PatternData(barsdf[startday-9:startday+1]))
                    startday = 0
                    # if daycnt > 10:
                        # patternAr.append(PatternData(barsdf[daycnt-9:daycnt+1]))
                # print 'gain:',periodgain,barsdf.index[daycnt],npclose[daycnt],benchgainnp[daycnt]\
                #     ,'negnum:',negnum,' posnum:',posnum

    negsigdf = pd.DataFrame(negsig,index = barsdf.index,columns=['Negsig']).fillna(0.0)
    negsigdf.index.name = 'Date'    

    possigdf = pd.DataFrame(possig,index = barsdf.index,columns=['Possig']).fillna(0.0)
    possigdf.index.name = 'Date'    
    # print patternAr[0].open,patternAr[0].close,patternAr[1].close
    # print patternAr[0].patterndf,patternAr[1].patterndf

    # patterndf = pd.DataFrame({'Open':patternAr[0].open,'Close':patternAr[0].close}).fillna(0.0)
    # patterndf.index.name = 'Date'    
    # print len(patternAr)
    # print patterndf


    return negsigdf,possigdf,patternAr

def patternAllRunUp(targetdf,targetnp,barsdf,gpatternAr):
    # global targetdf 
    # global targetlen
    # global gbarsdf

    # global gpatternAr
    lpatternAr = gpatternAr    
    # global patterntotallen

    targetpatNum = 0
    searchpatNum = 0
    missingpat = []
    allselectpattern = []
    start = time.clock()
    
    totallen = len(barsdf['Close'])
    stance = 'none'

    for daycnt in range(totallen):
        if targetnp[daycnt] == 1 :
            if stance == 'none':
                stance = 'holding'

            gcurday = daycnt

            curpat = barsdf.reset_index()

            corrclosex1 = curpat['Close'][gcurday-9:gcurday+1].pct_change().cumsum()
            closex1 = pd.DataFrame(corrclosex1,columns=['ClosePc']).fillna(0.0)
            closex1 = closex1.reset_index()
            
            corropenx1 = curpat['Open'][gcurday-9:gcurday+1].pct_change().cumsum()
            openx1 = pd.DataFrame(corropenx1,columns=['OpenPc']).fillna(0.0)
            openx1 = openx1.reset_index()
            
            corrhighx1 = curpat['High'][gcurday-9:gcurday+1].pct_change().cumsum()
            highx1 = pd.DataFrame(corrhighx1,columns=['HighPc']).fillna(0.0)
            highx1 = highx1.reset_index()

            corrlowx1 = curpat['Low'][gcurday-9:gcurday+1].pct_change().cumsum()
            lowx1 = pd.DataFrame(corrlowx1,columns=['LowPc']).fillna(0.0)
            lowx1 = lowx1.reset_index()

            corrvolx1 = curpat['Volume'][gcurday-9:gcurday+1].pct_change().cumsum()
            volx1 = pd.DataFrame(corrvolx1,columns=['VolPc']).fillna(0.0)
            volx1 = volx1.reset_index()
            
            targetpatNum +=1 
            
            targetpatFound = False
            # startnum = targetdf['targetCol3'][numcnt]
            # for patternnum in range(startnum,len(gpatternAr)):
            for patternnum in range(len(gpatternAr)):

                pattern0 = lpatternAr[patternnum].patterndf.reset_index()
                
                pattern0_closepc = pattern0['Close'].pct_change().cumsum()   
                closepc0 = pd.DataFrame(pattern0_closepc,index = pattern0.index,columns=['ClosePc']).fillna(0.0)

                closecorr = closex1['ClosePc'].corr(closepc0['ClosePc'])
                if closecorr <= 0.8:
                    continue

                pattern0_openpc = pattern0['Open'].pct_change().cumsum()   
                openpc0 = pd.DataFrame(pattern0_openpc,index = pattern0.index,columns=['OpenPc']).fillna(0.0)

                pattern0_highpc = pattern0['High'].pct_change().cumsum()   
                highpc0 = pd.DataFrame(pattern0_highpc,index = pattern0.index,columns=['HighPc']).fillna(0.0)

                pattern0_lowpc = pattern0['Low'].pct_change().cumsum()   
                lowpc0 = pd.DataFrame(pattern0_lowpc,index = pattern0.index,columns=['LowPc']).fillna(0.0)
                
                pattern0_volpc = pattern0['Volume'].pct_change().cumsum()   
                volpc0 = pd.DataFrame(pattern0_volpc,index = pattern0.index,columns=['VolPc']).fillna(0.0)
                
                # patternall = pd.concat([pattern0,openpc0,closepc0,highpc0,lowpc0,volpc0],axis=1)
                
                opencorr = openx1['OpenPc'].corr(openpc0['OpenPc'])
                highcorr = highx1['HighPc'].corr(highpc0['HighPc'])
                lowcorr = lowx1['LowPc'].corr(lowpc0['LowPc'])
                volcorr = volx1['VolPc'].corr(volpc0['VolPc'])

                corrsum = (closecorr+opencorr+highcorr+lowcorr+volcorr)/5
                if corrsum > 0.8:
                    allselectpattern.append(PatternExtractData(lpatternAr[patternnum].patterndf,patternnum,gcurday,targetpatNum)) 
                    targetpatFound = True
            if targetpatFound == True:
                searchpatNum += 1   
            else:
                missingpat.append(targetpatNum)     

        elif targetnp[daycnt] == 0 and stance == 'holding':
            stance = 'none'


    print 'targetpatNum:',targetpatNum,'searchpatNum:',searchpatNum 
    # print 'missingpat:',missingpat
    elapsed = (time.clock() - start)
    print 'patternAllRun elapsed time:',elapsed
    return allselectpattern


def patternAllRun(targetdf,barsdf,gpatternAr):
    # global targetdf 
    # global targetlen
    # global gbarsdf

    # global gpatternAr
    lpatternAr = gpatternAr    
    # global patterntotallen

    targetpatNum = 0
    searchpatNum = 0
    missingpat = []
    allselectpattern = []
    start = time.clock()
    for numcnt in range(len(targetdf)):
        # print targetdf['targetCol'][numcnt] 
        if targetdf['targetCol2'][numcnt] == True:
            gcurday = numcnt

            curpat = barsdf.reset_index()

            corrclosex1 = curpat['Close'][gcurday-9:gcurday+1].pct_change().cumsum()
            closex1 = pd.DataFrame(corrclosex1,columns=['ClosePc']).fillna(0.0)
            closex1 = closex1.reset_index()
            
            corropenx1 = curpat['Open'][gcurday-9:gcurday+1].pct_change().cumsum()
            openx1 = pd.DataFrame(corropenx1,columns=['OpenPc']).fillna(0.0)
            openx1 = openx1.reset_index()
            
            corrhighx1 = curpat['High'][gcurday-9:gcurday+1].pct_change().cumsum()
            highx1 = pd.DataFrame(corrhighx1,columns=['HighPc']).fillna(0.0)
            highx1 = highx1.reset_index()

            corrlowx1 = curpat['Low'][gcurday-9:gcurday+1].pct_change().cumsum()
            lowx1 = pd.DataFrame(corrlowx1,columns=['LowPc']).fillna(0.0)
            lowx1 = lowx1.reset_index()

            corrvolx1 = curpat['Volume'][gcurday-9:gcurday+1].pct_change().cumsum()
            volx1 = pd.DataFrame(corrvolx1,columns=['VolPc']).fillna(0.0)
            volx1 = volx1.reset_index()
            
            targetpatNum +=1 

            # global allselectpattern
            
            targetpatFound = False
            # startnum = targetdf['targetCol3'][numcnt]
            # for patternnum in range(startnum,len(gpatternAr)):
            for patternnum in range(len(gpatternAr)):

                pattern0 = lpatternAr[patternnum].patterndf.reset_index()
                
                pattern0_closepc = pattern0['Close'].pct_change().cumsum()   
                closepc0 = pd.DataFrame(pattern0_closepc,index = pattern0.index,columns=['ClosePc']).fillna(0.0)

                closecorr = closex1['ClosePc'].corr(closepc0['ClosePc'])
                if closecorr <= 0.8:
                    continue

                pattern0_openpc = pattern0['Open'].pct_change().cumsum()   
                openpc0 = pd.DataFrame(pattern0_openpc,index = pattern0.index,columns=['OpenPc']).fillna(0.0)

                pattern0_highpc = pattern0['High'].pct_change().cumsum()   
                highpc0 = pd.DataFrame(pattern0_highpc,index = pattern0.index,columns=['HighPc']).fillna(0.0)

                pattern0_lowpc = pattern0['Low'].pct_change().cumsum()   
                lowpc0 = pd.DataFrame(pattern0_lowpc,index = pattern0.index,columns=['LowPc']).fillna(0.0)
                
                pattern0_volpc = pattern0['Volume'].pct_change().cumsum()   
                volpc0 = pd.DataFrame(pattern0_volpc,index = pattern0.index,columns=['VolPc']).fillna(0.0)
                
                # patternall = pd.concat([pattern0,openpc0,closepc0,highpc0,lowpc0,volpc0],axis=1)
                
                opencorr = openx1['OpenPc'].corr(openpc0['OpenPc'])
                highcorr = highx1['HighPc'].corr(highpc0['HighPc'])
                lowcorr = lowx1['LowPc'].corr(lowpc0['LowPc'])
                volcorr = volx1['VolPc'].corr(volpc0['VolPc'])

                corrsum = (closecorr+opencorr+highcorr+lowcorr+volcorr)/5
                if corrsum > 0.8:
                    allselectpattern.append(PatternExtractData(lpatternAr[patternnum].patterndf,patternnum,gcurday,targetpatNum)) 
                    targetpatFound = True
            if targetpatFound == True:
                searchpatNum += 1   
            else:
                missingpat.append(targetpatNum)     

    print 'targetpatNum:',targetpatNum,'searchpatNum:',searchpatNum 
    # print 'missingpat:',missingpat
    elapsed = (time.clock() - start)
    print 'patternAllRun elapsed time:',elapsed
    return allselectpattern

'''
find how many times the current pattern generates in the all patterns 
'''
def patternCompareAndExtract(allselectpattern):
    # global allselectpattern
    allsellen = len(allselectpattern)
    start = time.clock()
    for patternnum in range(allsellen):    
        # print allselectpattern[patternnum].patterndf
        # print 'patternid:',allselectpattern[patternnum].patternid
        # print 'parentid:',allselectpattern[patternnum].parentid
        # print 'targetpatNum:',allselectpattern[patternnum].targetpatNum

        for searchnum in range(allsellen):
            if allselectpattern[patternnum].parentid != allselectpattern[searchnum].parentid:
                if allselectpattern[patternnum].patternid == allselectpattern[searchnum].patternid:
                    allselectpattern[patternnum].foundnum = allselectpattern[patternnum].getFoundCount() + 1
                    
                    # print 'allselectpattern[patternnum].foundnum:',allselectpattern[patternnum].foundnum,'patternnum :',patternnum

    foundnumlist = []
    for patternnum in range(allsellen):    
        foundnumlist.append(allselectpattern[patternnum].getFoundCount()+1)
    # print len(foundnumlist),len(allselectpattern)    
    # global gfoundnumlist
    # gfoundnumlist = foundnumlist
    elapsed = (time.clock() - start)
    print 'patternCompareAndExtract elapsed time:',elapsed
    return foundnumlist

'''
find the extracted candidats for patterns
'''
def patternExtractCandidates(foundnumlist,allselectpattern,patternAr):
    
    # global allselectpattern
    gfoundnumlist = foundnumlist
    start = time.clock()
    # print 'gfoundnumlist:',gfoundnumlist
    gpatternAr = patternAr
    # lpatternAr_org = deepcopy(gpatternAr)
    allselectpattern_org = deepcopy(allselectpattern)

    maxindex = gfoundnumlist.index(max(gfoundnumlist))
    maxvalue = gfoundnumlist[maxindex] 
    extractid = []
    whilecnt = 0

    # for patternnum in range(len(allselectpattern)):
    #     print 'patternid:',allselectpattern[patternnum].patternid
    #     print 'parentid:',allselectpattern[patternnum].parentid
    #     print 'targetpatNum:',allselectpattern[patternnum].targetpatNum

    while maxvalue != 0:#whilecnt < 2:

        extractid.append(allselectpattern[maxindex].patternid)
        extractid0 = allselectpattern[maxindex].patternid

        # print 'len(foundnumlist):',len(gfoundnumlist),'allselectpattern :',len(allselectpattern),'maxvalue:',maxvalue,'patternid:',extractid0

        deleteitem = []
        parentid = -1
        prevparentid = -1
        
        for cnt in range(maxvalue):
            # print 'loop count:',cnt,'parentid:',parentid,'prevparentid:',prevparentid
            for searchnum in range(len(allselectpattern)):
                parentidFound = False
                if allselectpattern[searchnum].patternid == extractid0:
                    parentid = allselectpattern[searchnum].parentid
                    # print 'searching parentid:',allselectpattern[searchnum].parentid
                    if prevparentid >= parentid:
                        continue
                    prevparentid = parentid
                    parentidFound = True
                    # print 'found parentid:',parentid
                    break
            if parentidFound == True:
                for searchnum in range(len(allselectpattern)):
                    if allselectpattern[searchnum].parentid == parentid:
                        deleteitem.append(searchnum)
                        # print 'deletitem:',searchnum
                    

        # print len(deleteitem),deleteitem
        delcnt = 0
        for item in deleteitem:
            # print 'del allselectpattern[item]:',item,'parentid:',allselectpattern[item-delcnt].parentid
            del allselectpattern[item-delcnt]
            delcnt += 1

        for patternnum in range(len(allselectpattern)):    
            allselectpattern[patternnum].setFoundCount(0) 

        for patternnum in range(len(allselectpattern)):    

            for searchnum in range(len(allselectpattern)):
                if allselectpattern[patternnum].parentid != allselectpattern[searchnum].parentid:
                    if allselectpattern[patternnum].patternid == allselectpattern[searchnum].patternid:
                        allselectpattern[patternnum].foundnum = allselectpattern[patternnum].getFoundCount() + 1

        foundnumlist = []
        for patternnum in range(len(allselectpattern)):    
            foundnumlist.append(allselectpattern[patternnum].getFoundCount()+1)
        
        gfoundnumlist = foundnumlist

        if len(gfoundnumlist) > 0 :
            maxindex = gfoundnumlist.index(max(gfoundnumlist))
            maxvalue = gfoundnumlist[maxindex] 
            # print 'reorganize len(foundnumlist):',len(foundnumlist),'len(allselectpattern):',len(allselectpattern) ,'maxvalue:',maxvalue
            whilecnt += 1
        else:
            break


    # print 'extractid:',extractid,'len',len(extractid)    
    sortextractid = []
    foundcntlist = []
    foundidlist = []

    sellen = len(allselectpattern_org)
    # print 'len(allselectpattern_org):',len(allselectpattern_org)
    for idnum in extractid:  
        for patternnum in range(sellen):    
            if allselectpattern_org[patternnum].patternid == idnum:
                foundcntlist.append(allselectpattern_org[patternnum].getFoundCount()+1)
                foundidlist.append(idnum)
                # print 'patternid:',allselectpattern_org[patternnum].patternid,'parentid:,',allselectpattern_org[patternnum].parentid
                break    
    # print 'foundcntlist:',foundcntlist,'len:',len(foundcntlist)

    maxindex = foundcntlist.index(max(foundcntlist))
    maxvalue = foundidlist[maxindex] 
    stopvalue = max(foundcntlist)
    while stopvalue != 0:  
        sortextractid.append(maxvalue)
        
        del foundcntlist[maxindex]
        del foundidlist[maxindex]
        
        # print 'foundcntlist:',foundcntlist
        if len(foundcntlist) > 0 :
            maxindex = foundcntlist.index(max(foundcntlist))
            maxvalue = foundidlist[maxindex] 
            stopvalue = max(foundcntlist)
            # print 'max found patternid:',maxvalue,'max count:',max(foundcntlist)
        else:
            break
    
    print 'sortextractid:',sortextractid,'len:',len(sortextractid)
    # for idnum in sortextractid:  
    #     # print 'idnum',idnum 
    #     for patternnum in range(sellen):    
    #         # print 'patternid:',allselectpattern_org[num].patternid
    #         if allselectpattern_org[patternnum].patternid == idnum:
    #             print 'id:',idnum,'foundnum:',allselectpattern_org[patternnum].foundnum,'parentid:',allselectpattern_org[patternnum].parentid
                
                    



    # extractid = list(set(extractid))
    # print 'extractid:',extractid,'len',len(extractid)
    
    # global gextractid
    # gextractid = extractid
    elapsed = (time.clock() - start)
    print 'patternExtractCandidates elapsed time:',elapsed
    return sortextractid#extractid

def patternSelect(currentday, barsdf, gpatternAr, gextractid):

    lpatternAr = gpatternAr
    global gcurday
    gcurday = currentday
    global gselectedpattern

    global gbarsdf
    gbarsdf = barsdf
    curpat = barsdf.reset_index()
    gcurday = currentday

    print 'patternSelect gcurday:', gcurday

    corrclosex1 = curpat['Close'][gcurday - 9:gcurday + 1].pct_change().cumsum()
    closex1 = pd.DataFrame(corrclosex1, columns=['ClosePc']).fillna(0.0)
    closex1 = closex1.reset_index()

    corropenx1 = curpat['Open'][gcurday - 9:gcurday + 1].pct_change().cumsum()
    openx1 = pd.DataFrame(corropenx1, columns=['OpenPc']).fillna(0.0)
    openx1 = openx1.reset_index()

    corrhighx1 = curpat['High'][gcurday - 9:gcurday + 1].pct_change().cumsum()
    highx1 = pd.DataFrame(corrhighx1, columns=['HighPc']).fillna(0.0)
    highx1 = highx1.reset_index()

    corrlowx1 = curpat['Low'][gcurday - 9:gcurday + 1].pct_change().cumsum()
    lowx1 = pd.DataFrame(corrlowx1, columns=['LowPc']).fillna(0.0)
    lowx1 = lowx1.reset_index()

    corrvolx1 = curpat['Volume'][gcurday - 9:gcurday + 1].pct_change().cumsum()
    volx1 = pd.DataFrame(corrvolx1, columns=['VolPc']).fillna(0.0)
    volx1 = volx1.reset_index()

    selectpattern = []
    for patternnum in gextractid:

        pattern0 = lpatternAr[patternnum].patterndf.reset_index()

        pattern0_openpc = pattern0['Open'].pct_change().cumsum()
        openpc0 = pd.DataFrame(
            pattern0_openpc, index=pattern0.index, columns=['OpenPc']).fillna(0.0)

        pattern0_closepc = pattern0['Close'].pct_change().cumsum()
        closepc0 = pd.DataFrame(
            pattern0_closepc, index=pattern0.index, columns=['ClosePc']).fillna(0.0)

        pattern0_highpc = pattern0['High'].pct_change().cumsum()
        highpc0 = pd.DataFrame(
            pattern0_highpc, index=pattern0.index, columns=['HighPc']).fillna(0.0)

        pattern0_lowpc = pattern0['Low'].pct_change().cumsum()
        lowpc0 = pd.DataFrame(
            pattern0_lowpc, index=pattern0.index, columns=['LowPc']).fillna(0.0)

        pattern0_volpc = pattern0['Volume'].pct_change().cumsum()
        volpc0 = pd.DataFrame(
            pattern0_volpc, index=pattern0.index, columns=['VolPc']).fillna(0.0)

        patternall = pd.concat(
            [pattern0, openpc0, closepc0, highpc0, lowpc0, volpc0], axis=1)

        closecorr = closex1['ClosePc'].corr(closepc0['ClosePc'])
        opencorr = openx1['OpenPc'].corr(openpc0['OpenPc'])
        highcorr = highx1['HighPc'].corr(highpc0['HighPc'])
        lowcorr = lowx1['LowPc'].corr(lowpc0['LowPc'])
        volcorr = volx1['VolPc'].corr(volpc0['VolPc'])

        corrsum = (closecorr + opencorr + highcorr + lowcorr + volcorr) / 5
        # print 'corrsum:', corrsum
        if corrsum > 0.8 and corrsum > 0.8 and volcorr > 0.8:
            selectpattern.append(lpatternAr[patternnum])
            print 'found corrsum:', corrsum
            return True

    if len(selectpattern) == 0:
        print 'pattern not found'
        return False




def getMinMaxPoint(searchtype,fromday,today,mincount,maxcount,minmaxdf,signaldf,includetoday):
#     searchtype = 1 #1: max 2:min
#     mincnt = 1
#     maxcnt = 0
    mincnt = mincount
    maxcnt = maxcount
    minpoint = 0
    maxpoint = 0
    if includetoday == 1:
        today = today +1
    for daycnt in range(0,today):
        dayidx = today - daycnt-1
    #     print dayidx
        if searchtype == 1:
            if minmaxdf['MAXSignals'][minmaxdf.index[dayidx]] == 1:   
                if maxpoint == maxcnt:
                    # print 'MAX',minmaxdf.index[dayidx],'Found:',minmaxdf['MAXSignals'][minmaxdf.index[dayidx]]
                    dfidx = minmaxdf.index[dayidx]
                    p0 = signaldf['Value'][dfidx]
                    return dayidx,p0
                    break
                maxpoint = maxpoint + 1    
        elif searchtype == 2:            
            if minmaxdf['MINSignals'][minmaxdf.index[dayidx]] == 1:   
                # print 'MIN',minmaxdf.index[dayidx],'Found:',minmaxdf['MINSignals'][minmaxdf.index[dayidx]]
                if minpoint == mincnt:
                    # print 'MIN',minmaxdf.index[dayidx],'Found:',minmaxdf['MINSignals'][minmaxdf.index[dayidx]]
                    dfidx = minmaxdf.index[dayidx]
                    p0 = signaldf['Value'][dfidx]
                    return dayidx,p0
                    break
                minpoint = minpoint + 1
        if dayidx == 0 or dayidx <= fromday:
            return -1,-1


def getMinMaxPointAngle(pointnum,searchtype,fromday,today,mincount,maxcount,minmaxdf,signaldf,includetoday):
    day0 = []
    value0 = []
    curday = today
    if searchtype == 1:
        totalcount = len(minmaxdf[minmaxdf['MAXSignals']==1][0:minmaxdf.index[curday]])
        # print 'max totalcount:',totalcount,'today:',minmaxdf.index[curday]
    else:
        totalcount = len(minmaxdf[minmaxdf['MINSignals']==1][0:minmaxdf.index[curday]])
        # print 'min totalcount:',totalcount,'today:',minmaxdf.index[curday]
    rangecount = 0

    if totalcount >=2 :
        for daycnt in range(0,totalcount):
            if searchtype == 1:
                maxcount =  daycnt
            else:
                mincount =  daycnt
            retday,retvalue = getMinMaxPoint(searchtype,fromday,curday,mincount,maxcount,minmaxdf,signaldf,includetoday)
            if not retday == -1 and not retvalue == -1:
                day0.append(retday)
                value0.append(retvalue)
                rangecount = rangecount +1
            if rangecount == pointnum:
                break

        # if searchtype == 1:
        #     print 'max day',minmaxdf.index[day0]
        #     print 'max value',value0
        # else:
        #     print 'min day',minmaxdf.index[day0]
        #     print 'min value',value0
        # print 'rangecount:',rangecount
        p0 = value0
        p1 = sm.add_constant(day0, prepend=False)
        slope, intercept = sm.OLS(p0, p1).fit().params
        angle = np.arctan(slope)*57.3
        # print slope,',',intercept,'angle:',angle
        return angle,minmaxdf.index[day0],value0
    else:   
        return 0,0,0        



def inflectionPoint_ver1(signalnp,dayslim,barsdf):
    print 'inflectionPoint ver1 start'

    maxqueue =Deque()
    maxqueue.front = 0
    maxqueue.rear = 0

    minqueue =Deque()
    minqueue.front = 0
    minqueue.rear = 0
    
    print len(barsdf['Close']),len(signalnp)
    # print dayslim
    nextsearch = 'none'
    minday = 0
    maxday = 0
    
    for day in range(len(signalnp)):

        if day >dayslim:
            if nextsearch == 'none':
                
                for daycnt in range(dayslim-2):
                    searchday = day-daycnt
                    if (signalnp[searchday] <= signalnp[searchday-1] > signalnp[searchday-2])\
                        or (signalnp[searchday] < signalnp[searchday-1] >= signalnp[searchday-2]):
                        maxday = searchday -1
                        maxqueue.addRear(maxday)
                        maxqueue.rear += 1
                        nextsearch = 'min'
                        # print 'maxday:',barsdf.index[maxday],'signalnp[maxday]:',signalnp[maxday],'day:',barsdf.index[day]
                        break
                    if (signalnp[searchday] >= signalnp[searchday-1] < signalnp[searchday-2])\
                        or (signalnp[searchday] > signalnp[searchday-1] <= signalnp[searchday-2]):
                        minday = searchday -1
                        minqueue.addRear(minday)
                        minqueue.rear += 1
                        nextsearch = 'max'
                        # print 'minday:',barsdf.index[minday],'signalnp[minday]:',signalnp[minday],'day:',barsdf.index[day]
                        break

            elif nextsearch == 'min':
                for daycnt in range((day-maxday)-2+1):
                    searchday = day-daycnt
                    if (signalnp[searchday] >= signalnp[searchday-1] < signalnp[searchday-2])\
                        or (signalnp[searchday] > signalnp[searchday-1] <= signalnp[searchday-2]):
                        minday = searchday -1
                        minqueue.addRear(minday)
                        minqueue.rear += 1
                        nextsearch = 'max'
                        # print 'minday:',barsdf.index[minday],'signalnp[minday]:',signalnp[minday],'day:',barsdf.index[day]
                        break

            elif nextsearch == 'max':                    
                for daycnt in range((day-minday)-2+1):
                    searchday = day-daycnt
                    if (signalnp[searchday] <= signalnp[searchday-1] > signalnp[searchday-2])\
                        or (signalnp[searchday] < signalnp[searchday-1] >= signalnp[searchday-2]):
                        maxday = searchday -1
                        maxqueue.addRear(maxday)
                        maxqueue.rear += 1
                        nextsearch = 'min'
                        # print 'maxday:',barsdf.index[maxday],'signalnp[maxday]:',signalnp[maxday],'day:',barsdf.index[day]
                        break



    maxsignp = np.zeros(len(barsdf['Close']))
    minsignp = np.zeros(len(barsdf['Close']))

    # print 'maxqueue.size:',maxqueue.size()
    for maxcnt in range(maxqueue.size()):
        # print 'maxqueue day:',mixedSigMA1np[maxqueue.getValue(maxcnt)],CloseLD.index[maxqueue.getValue(maxcnt)]
        maxsignp[maxqueue.getValue(maxcnt)] = 1

    maxsigdf = pd.DataFrame(maxsignp,index = barsdf.index,columns=['MAXSignals']).fillna(0.0)
    maxsigdf.index.name = 'Date'    

    # print mixedSigMA1[maxsigdf.ix[maxsigdf.MAXSignals ==1].index]

    # print 'minqueue.size:',minqueue.size()
    for mincnt in range(minqueue.size()):
        # print 'minqueue day:',mixedSigMA1np[minqueue.getValue(mincnt)],CloseLD.index[minqueue.getValue(mincnt)]
        minsignp[minqueue.getValue(mincnt)] = 1

    minsigdf = pd.DataFrame(minsignp,index = barsdf.index,columns=['MINSignals']).fillna(0.0)
    minsigdf.index.name = 'Date'    
    
    mmsignp = np.zeros(len(barsdf['Close']))
    mmstance = 'none'

    for day in range(len(barsdf['Close'])):
        if day > dayslim:
            if mmstance == 'none':
                if minsignp[day] == 1:
                    mmsignp[day] = 0
                    mmstance = 'candidate'
            elif mmstance == 'candidate':                
                mmsignp[day] = 1
                mmstance = 'holding'
                if maxsignp[day] == 1:
                    mmsignp[day] = 0
                    mmstance = 'minmax'

            elif mmstance == 'holding':
                if minsignp[day] == 0:
                    mmsignp[day] = 1
                if maxsignp[day] == 1:
                    mmsignp[day] = 1
                    mmstance = 'none'
            elif mmstance == 'minmax':
                mmsignp[day] = 0
                mmstance = 'none'                                
                

    mmsigdf = pd.DataFrame(mmsignp,index = barsdf.index,columns=['GoodMMSignals']).fillna(0.0)
    mmsigdf.index.name = 'Date'    

    print 'inflectionPoint ver1 end'
    return mmsigdf,mmsignp,maxsigdf,minsigdf,maxsignp,minsignp,maxqueue,minqueue

    
def inflectionPoint(signalnp,dayslim,barsdf):
    print 'inflectionPoint start'

    maxqueue =Deque()
    maxqueue.front = 0
    maxqueue.rear = 0

    minqueue =Deque()
    minqueue.front = 0
    minqueue.rear = 0
    
    print len(barsdf['Close']),len(signalnp)
    # print dayslim
    ipclosep = barsdf['Close'].values
    
    nextsearch = 'none'
    minday = 0
    maxday = 0
    
    # print barsdf['2014-03-25':'2014-04-10']
    for day in range(len(signalnp)):

        if day >= dayslim:#5:
            
            if day >dayslim:
                if nextsearch == 'none':
                    
                    for daycnt in range(dayslim-2):
                        searchday = day-daycnt
                        if (signalnp[searchday] <= signalnp[searchday-1] > signalnp[searchday-2])\
                            or (signalnp[searchday] < signalnp[searchday-1] >= signalnp[searchday-2]):
                            maxday = searchday -1
                            maxqueue.addRear(maxday)
                            maxqueue.rear += 1
                            nextsearch = 'min'
                            # print 'maxday:',barsdf.index[maxday],'signalnp[maxday]:',signalnp[maxday],'day:',barsdf.index[day]
                            break
                        if (signalnp[searchday] >= signalnp[searchday-1] < signalnp[searchday-2])\
                            or (signalnp[searchday] > signalnp[searchday-1] <= signalnp[searchday-2]):
                            minday = searchday -1
                            minqueue.addRear(minday)
                            minqueue.rear += 1
                            nextsearch = 'max'
                            # print 'minday:',barsdf.index[minday],'signalnp[minday]:',signalnp[minday],'day:',barsdf.index[day]
                            break

                elif nextsearch == 'min':
                    for daycnt in range((day-maxday)-2+1):
                        searchday = day-daycnt
                        if (signalnp[searchday] >= signalnp[searchday-1] < signalnp[searchday-2])\
                            or (signalnp[searchday] > signalnp[searchday-1] <= signalnp[searchday-2]):
                            minday = searchday -1
                            minqueue.addRear(minday)
                            minqueue.rear += 1
                            nextsearch = 'max'
                            # print 'minday:',barsdf.index[minday],'signalnp[minday]:',signalnp[minday],'day:',barsdf.index[day]
                            break

                elif nextsearch == 'max':                    
                    for daycnt in range((day-minday)-2+1):
                        searchday = day-daycnt
                        if (signalnp[searchday] <= signalnp[searchday-1] > signalnp[searchday-2])\
                            or (signalnp[searchday] < signalnp[searchday-1] >= signalnp[searchday-2]):
                            maxday = searchday -1
                            maxqueue.addRear(maxday)
                            maxqueue.rear += 1
                            nextsearch = 'min'
                            # print 'maxday:',barsdf.index[maxday],'signalnp[maxday]:',signalnp[maxday],'day:',barsdf.index[day]
                            break


                
                
    maxsignp = np.zeros(len(signalnp))
    minsignp = np.zeros(len(signalnp))

    # print 'maxqueue.size:',maxqueue.size()
    for maxcnt in range(maxqueue.size()):
        # if barsdf.index[maxqueue.getValue(maxcnt)] > dt.datetime(2014,3,1):
        #     print 'maxqueue day:',barsdf.index[maxqueue.getValue(maxcnt)]
        maxsignp[maxqueue.getValue(maxcnt)] = 1

    maxsigdf = pd.DataFrame(maxsignp,index = barsdf.index,columns=['MAXSignals']).fillna(0.0)
    maxsigdf.index.name = 'Date'    


    # print mixedSigMA1[maxsigdf.ix[maxsigdf.MAXSignals ==1].index]

    # print 'minqueue.size:',minqueue.size()
    for mincnt in range(minqueue.size()):
        # if barsdf.index[minqueue.getValue(mincnt)] > dt.datetime(2014,3,1):
        #     print 'minqueue day:',barsdf.index[minqueue.getValue(mincnt)]
        minsignp[minqueue.getValue(mincnt)] = 1

    minsigdf = pd.DataFrame(minsignp,index = barsdf.index,columns=['MINSignals']).fillna(0.0)
    minsigdf.index.name = 'Date'    
    

    mmsignp = np.zeros(len(signalnp))
    mmstance = 'none'
    
    mmStancequeue =Deque()
    mmStancequeue.front = 0
    mmStancequeue.rear = 0

    mmGainqueue =Deque()
    mmGainqueue.front = 0
    mmGainqueue.rear = 0

    curGain = 0
    totalGain = 0
    buyprice = 0

    searchtype=2 # 1:max,2:min
    fromday = 0
    curday=0
    
    curangle = 0
    
    minangle2 = 0
    maxangle2 = 0
    
    buyangle =0 
    buylowerslope = 0
    buylowerintercept = 0
    buyupperslope = 0
    buyupperintercept = 0
    
    minmaxdf = pd.concat([maxsigdf,minsigdf,barsdf],axis=1).fillna(0.0)
    signaldf2 = pd.DataFrame(signalnp,index = barsdf.index,columns=['Value']).fillna(0.0)
    signaldf2.index.name = 'Date'    

    retday0 = 0
    retvalue0 = 0
    pchigh = 0
    pclow = 0

    benchgain = barsdf['Close'].pct_change().cumsum()   
    benchgainnp =benchgain.values
    for day in range(len(signalnp)):
        # if barsdf.index[day] > dt.datetime(2013,1,1)\
        #     and barsdf.index[day] < dt.datetime(2013,12,1):
        if day > dayslim:
            
            if mmstance == 'none':

                if minsignp[day-1] == 1:

                    curday = day
                    pointnum = 2
                    mincount=0
                    maxcount=0
                    searchtype=2 # 1:max,2:min

                    
                    minday0,minvalue0 = getMinMaxPoint(searchtype,0,curday,0,0,minmaxdf,signaldf2,0)
                    minday1,minvalue1 = getMinMaxPoint(searchtype,0,curday,1,0,minmaxdf,signaldf2,0)
                    searchtype=1 # 1:max,2:min                    
                    maxday0,maxvalue0 = getMinMaxPoint(searchtype,0,curday,0,0,minmaxdf,signaldf2,0)
                    maxday1,maxvalue1 = getMinMaxPoint(searchtype,0,curday,0,1,minmaxdf,signaldf2,0)
                    caly = 0


                    if not minday0 == -1 and not minday1 == -1 and not maxday0 == -1 and not maxday1 == -1:
                        mmsignp[day] = 1
                        mmstance = 'holding'
                        mmStancequeue.addRear('min')#today status
                        mmStancequeue.rear += 1
                        buyprice = signalnp[day]
                        curGain = 0
                        mmGainqueue.addRear(curGain)    
                        mmGainqueue.rear += 1

                        # print 'mmstance buy today: ',barsdf.index[day],', minangle2:',minangle2,\
                        # ',maxangle2:',maxangle2,', price:',barsdf['Close'][barsdf.index[day]],'maxmean:',maxmean,',minmean:',minmean
                    else:
                        mmsignp[day] = 0
                        mmStancequeue.addRear('none')#today status
                        mmStancequeue.rear += 1  
                        curGain = 0
                        mmGainqueue.addRear(curGain)    
                        mmGainqueue.rear += 1  
                        # print 'mmstance skip today: ',barsdf.index[day],', minangle2:',minangle2,\
                        # ',maxangle2:',maxangle2,', price:',barsdf['Close'][barsdf.index[day]],', curangle:',curangle
                else:
                    mmsignp[day] = 0
                    mmStancequeue.addRear('none')#today status
                    mmStancequeue.rear += 1
                    
                    curGain = 0
                    mmGainqueue.addRear(curGain)    
                    mmGainqueue.rear += 1 
            elif mmstance == 'holding':                
        
                curday = day
                if maxsignp[day-1] == 1:

                    mmsignp[day] = 0
                    mmstance = 'none'    
                    mmStancequeue.addRear('none')#today status
                    mmStancequeue.rear += 1

                    # curGain = (signalnp[day]-buyprice)/buyprice

                    # print 'mmstance point sell plus today: ',barsdf.index[day],', gain:',curGain\
                    #     ,', price:',barsdf['Close'][barsdf.index[day]],', buyangle:',buyangle
                    curGain = (signalnp[day]-buyprice)/buyprice                
                    totalGain = totalGain + curGain
                    buyprice =0
                    mmGainqueue.addRear(curGain)    
                    mmGainqueue.rear += 1 
                else:
                    mmsignp[day] = 1
                    mmStancequeue.addRear('minholding')#today status
                    mmStancequeue.rear += 1
                    curGain = 0
                    mmGainqueue.addRear(curGain)    
                    mmGainqueue.rear += 1 
                        # print 'mmstance holding today: ',barsdf.index[day]\
                        #     ,', price:',barsdf['Close'][barsdf.index[day]],', minangle2:',minangle2,', buyangle:',buyangle
        else:
            mmsignp[day] = 0
            mmStancequeue.addRear('none')#today status
            mmStancequeue.rear += 1

            curGain = 0
            mmGainqueue.addRear(curGain)    
            mmGainqueue.rear += 1 


    # print 'mmStancequeue.size()',mmStancequeue.size(),len(signalnp)
    # mmstancenp = np.zeros(len(signalnp))
    # idx = mmStancequeue.rear
    # for stancecnt in range(len(signalnp)):
    #     idx = mmStancequeue.size()-stancecnt-1
    #     # print 'mmstancenp day:',barsdf.index[stancecnt],', stance:',mmStancequeue.getValue(idx)
    #     if mmStancequeue.getValue(idx) == 'min':
    #         mmstancenp[stancecnt] = 1

    # mmstancedf = pd.DataFrame(mmstancenp,index = barsdf.index,columns=['Stance']).fillna(0.0)
    # mmstancedf.index.name = 'Date'    

    # print mmstancedf[0:30]
    # global gmmstancedf
    # gmmstancedf =mmstancedf
    
    # idx = mmStancequeue.rear
    # for day in range(len(signalnp)):
    #     idx = mmStancequeue.size()-day-1
    #     print barsdf.index[day],ipclosep[day],'mmsignp signal:',mmsignp[day],'minsignp:',minsignp[day],'maxsignp:',maxsignp[day],\
    #     mmStancequeue.getValue(idx),'gain:',mmGainqueue.getValue(idx),'totalGain:',totalGain,'benchgain:',benchgainnp[day]

    mmsigdf = pd.DataFrame(mmsignp,index = barsdf.index,columns=['MMSignals']).fillna(0.0)
    mmsigdf.index.name = 'Date'    

    print 'inflectionPoint end'
    return mmsigdf,mmsignp,maxsigdf,minsigdf,maxsignp,minsignp,maxqueue,minqueue

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
    # print adjustdf
    return adjustdf




def ReadPatternsFromDB(codearg,typearg,namearg,mode):

    code = codearg #'097950'#'005930' #'005380'#009540 #036570
    if typearg == 1:
        symbol = 'GOOG/KRX_'+code
    elif typearg == 2:
        symbol = 'GOOG/KOSDAQ_'+code
    elif typearg == 3:
        symbol = 'GOOG/INDEXKRX_KOSPI200'  

    # symbol = 'GOOG/INDEXKRX_KOSPI200'
    # startdate = '2014-01-01'
    # enddate = '2008-12-30'
    # print symbol
    
    '''
    pattern read
    '''
    
    dbname = 'pattern_db_'+codearg+'_'+namearg+'.sqlite'

    if typearg == 1 or typearg == 4:
        title = "../../data/pattern/" + dbname
    elif typearg == 2:
        title = "../../data/pattern/kosdaq/" + dbname
    
    print 'ReadPatternsFromDB', title
    if not os.path.isfile(title):
        print 'no down trend db pattern',title
        return -1,-1

    con = sqlite3.connect(title)

    query = "SELECT * FROM sqlite_master WHERE type='table'"
    df = pd.io.sql.read_frame(query,con)

    tablelen = len(df)
    print 'tablelen:',tablelen    
    tablename_base = 'result_'+codearg+'_'+namearg
    
    readlist = []    
    for cnt in range(tablelen):
        tablename = tablename_base+'_'+str(cnt)
        # print 'readtable:',tablename
        patterndf = pd_sql.read_frame("SELECT * from "+tablename, con)
        readlist.append(PatternData(patterndf))
        readlist[cnt].patterndf.index = readlist[cnt].patterndf['Date']
        readlist[cnt].patterndf = readlist[cnt].patterndf.drop('Date',1)


    # print 'read pattern:',readlist[0].patterndf
    # print 'org patternAr:',patternAr_org[0].patterndf
    
    # con.close()    
    dbname = 'extractid_db_'+codearg+'_'+namearg+'.sqlite'
    if typearg == 1 or typearg == 4:
        title = "../../data/pattern/" + dbname
    elif typearg == 2:
        title = "../../data/pattern/kosdaq/" + dbname
    print 'ReadPatternsFromDB', title
    con2 = sqlite3.connect(title)
    tablename = 'result_'+codearg+'_'+namearg
    extractdf = pd_sql.read_frame("SELECT * from "+tablename, con2)
    extractids = extractdf['ExtractId'].values

    # print 'read pattern:'
    # print readlist[0].patterndf
    print 'extractids:',extractids,len(extractids)
    
    con.close()        
    con2.close()        
    
    return readlist,extractids




def ReadUpPatternsFromDB(codearg,typearg,namearg,mode):

    code = codearg #'097950'#'005930' #'005380'#009540 #036570
    if typearg == 1:
        symbol = 'GOOG/KRX_'+code
    elif typearg == 2:
        symbol = 'GOOG/KOSDAQ_'+code
    elif typearg == 3:
        symbol = 'GOOG/INDEXKRX_KOSPI200'  
    # symbol = 'GOOG/INDEXKRX_KOSPI200'
    # startdate = '2014-01-01'
    # enddate = '2008-12-30'
    print symbol
    
    '''
    pattern read
    '''
    
    dbname = 'pattern_db_'+codearg+'_'+namearg+'.sqlite'

    title = "../../data/pattern/up/" + dbname
    
    if not os.path.isfile(title):
        print 'no up trend db pattern'
        return -1,-1

    con = sqlite3.connect("../../data/pattern/up/"+dbname)

    query = "SELECT * FROM sqlite_master WHERE type='table'"
    df = pd.io.sql.read_frame(query,con)

    tablelen = len(df)
    print 'tablelen:',tablelen    
    tablename_base = 'result_'+codearg+'_'+namearg
    
    readlist = []    
    for cnt in range(tablelen):
        tablename = tablename_base+'_'+str(cnt)
        # print 'readtable:',tablename
        patterndf = pd_sql.read_frame("SELECT * from "+tablename, con)
        readlist.append(PatternData(patterndf))
        readlist[cnt].patterndf.index = readlist[cnt].patterndf['Date']
        readlist[cnt].patterndf = readlist[cnt].patterndf.drop('Date',1)


    # print 'read pattern:',readlist[0].patterndf
    # print 'org patternAr:',patternAr_org[0].patterndf
    
    # con.close()    
    dbname = 'extractid_db_'+codearg+'_'+namearg+'.sqlite'
    con2 = sqlite3.connect("../../data/pattern/up/"+dbname)
    tablename = 'result_'+codearg+'_'+namearg
    extractdf = pd_sql.read_frame("SELECT * from "+tablename, con2)
    extractids = extractdf['ExtractId'].values

    # print 'read pattern:'
    # print readlist[0].patterndf
    print 'up extractids:',extractids,len(extractids)
    
    con.close()        
    con2.close()        
    
    return readlist,extractids



def ReadHistFromDB(codearg,typearg,namearg,mode):
    print 'read hist data from DB'

    
    dbname = 'hist_db_'+codearg+'_'+namearg+'.sqlite'
    con = sqlite3.connect("../../data/hist/"+dbname)

    query = "SELECT * FROM sqlite_master WHERE type='table'"
    df = pd.io.sql.read_frame(query,con)

    tablelen = len(df)
    # print 'hist tablelen:',tablelen    
    tablename = 'result_'+codearg+'_'+namearg

    histdf = pd_sql.read_frame("SELECT * from "+tablename, con)
    
    from pandas.lib import Timestamp
    histdf.Date = histdf.Date.apply(Timestamp)
    histdf2 = histdf.set_index('Date')

    # histdf.index = histdf['Date']
    # histdf = histdf.drop('Date',1)
    # print 'histdf from db:'
    # print histdf2.head()
    # print 'hist index type:',type(histdf2.index)
    con.close()
    return histdf2

def fetchHistData(codearg,namearg,symbol,startdate):
    print 'fetchHistData'
    try:
        
        dbname = 'hist_db_'+codearg+'.sqlite'
        print 'histdb name',dbname,type(dbname)

        tablename = 'result_'+codearg
        # constpath = 'C:/Users/Administrator/Documents/IPython Notebooks/stock_simul/data/hist/hist_db_005930.sqlite'
        # con = sqlite3.connect(constpath, check_same_thread=False)
        global office
        if office == False:
            con = sqlite3.connect('C:/Users/Administrator/Documents/IPython Notebooks/stock_simul/data/hist/'+dbname)#,check_same_thread=False)
        elif office == True:
            con = sqlite3.connect('C:/Users/AUTRON/Documents/IPython Notebooks/simul/data/hist/'+dbname)#,check_same_thread=False)
        # con = sqlite3.connect("../../data/hist/"+dbname,check_same_thread=False)

        histdf = pd_sql.read_frame("SELECT * from "+tablename, con)

        '''
        con = sqlite3.connect("../../data/hist/"+dbname)
        

        query = "SELECT * FROM sqlite_master WHERE type='table'"
        df = pd.io.sql.read_frame(query,con)

        tablelen = len(df)
        # print 'hist tablelen:',tablelen    
        tablename = 'result_'+codearg+'_'+namearg

        histdf = pd_sql.read_frame("SELECT * from "+tablename, con)
        '''
        print 'histdb read done'
        from pandas.lib import Timestamp
        histdf.Date = histdf.Date.apply(Timestamp)
        histdf2 = histdf.set_index('Date')

        histdf2 = histdf2[histdf2.index >= startdate]
        # histdf.index = histdf['Date']
        # histdf = histdf.drop('Date',1)
        # print 'histdf from db:'
        # print histdf2.head()
        # print histdf2.tail()
        # print 'hist index type:',type(histdf2.index)
        con.close()
    except Exception,e:
        PrintException()
        con.close()
        pass

    return histdf2


def fetchRealData(code,symbol,typearg,startdate):
    # enddate = '2014-05-30'
    # bars_org =  Quandl.get(symbol, collapse='Daily', trim_start=startdate, trim_end=datetime.today(),authtoken="")
    
    bars_org =  Quandl.get(symbol,  trim_start=startdate, trim_end=datetime.today(),authtoken="")

    # bars = Quandl.get(symbol, collapse='Daily', trim_start=startdate, trim_end=enddate,authtoken="")
    # print bars[-10:]
    # print bars_org.tail()
    print '---------'
    #print len(bars)

    
    today = datetime.today()
    startday = today- timedelta(days=7 )
    # print today.year,today.month,today.day
    # print startday.year,startday.month,startday.day
    
    if typearg == 4:
        histurl = 'http://ichart.yahoo.com/table.csv?s=^KQ11'+'&a='+str(startday.month-1)+\
        '&b='+str(startday.day)+'&c='+str(startday.year)+'&d='+str(today.month-1)+'&e='+str(today.day)+'&f='+str(today.year)+'&ignore=.csv'
        # print histurl
    elif typearg == 3:
        histurl = 'http://ichart.yahoo.com/table.csv?s=^KS11'+'&a='+str(startday.month-1)+\
        '&b='+str(startday.day)+'&c='+str(startday.year)+'&d='+str(today.month-1)+'&e='+str(today.day)+'&f='+str(today.year)+'&ignore=.csv'
        # print histurl
    elif typearg ==2:
        histurl = 'http://ichart.yahoo.com/table.csv?s='+code+'.KQ'+'&a='+str(startday.month-1)+\
        '&b='+str(startday.day)+'&c='+str(startday.year)+'&d='+str(today.month-1)+'&e='+str(today.day)+'&f='+str(today.year)+'&ignore=.csv'
    else:
        histurl = 'http://ichart.yahoo.com/table.csv?s='+code+'.KS'+'&a='+str(startday.month-1)+\
        '&b='+str(startday.day)+'&c='+str(startday.year)+'&d='+str(today.month-1)+'&e='+str(today.day)+'&f='+str(today.year)+'&ignore=.csv'
        # print histurl
    '''
    yahoo scrape api 
    '''
    try:
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
        # print bars_org.tail()
        print '----date adjust end-----'

    except:
        ''' 
        naver scrape for yahoo ichart alternative
        '''
        histdf = fetchData(code) 
        histdf = histdf[histdf.Volume != 0]
        # print histdf
        print '---- exception date adjust start---'
        bars_new_unique = histdf[~histdf.index.isin(bars_org.index)]
        bars_org = pd.concat([bars_org, bars_new_unique])
        # print bars_org.tail()
        print '----date adjust end-----'
        
        print 'fetch real data return'
        # print bars_org.tail()        
    return bars_org




def DataFetch(codearg,typearg,namearg,mode,dbmode,histmode,srcsite,updbpattern,startdate):
    
    code = codearg #'097950'#'005930' #'005380'#009540 #036570
    if codearg == '005490' or codearg == '000660':
        srcsite = 2
        
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

    # startdate = '2014-01-01'
    # startday =  datetime.today() - timedelta(days=30)
    # startdate = str(startday).split(' ')[0]
    # print 'startdate',startdate
    if mode =='dbpattern'  or dbmode == 'dbpattern':        
        if updbpattern == 'none':
            print 'read DB patterns'
            patternAr, extractid= ReadPatternsFromDB(codearg,typearg,namearg,mode)
        elif updbpattern == 'updbpattern':
            print 'read UP DB patterns'
            patternAr, extractid= ReadUpPatternsFromDB(codearg,typearg,namearg,mode)

        if patternAr == -1:
            print 'real time gen db pattern'
            startdate = '2011-01-01'
            dbmode  = 'none'

    
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
    # print realtimeURL
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
        # print 'rtsymbol:',rtsymbol,'rtclose:',rtclose,rtdate,rttime,rtchange,'rtopen:',rtopen,'rthigh:',rthigh,'rtlow:',rtlow,'rtvolume:',rtvolume

    # print date2num(datetime.strptime(rtdate.replace('/',' '),'%m %d %Y'))

    # print bars.index[-1]
    # print date_object > bars.index[-1]
    # date_object  = date_object- dt.timedelta(days=1) 
    # print date_object > bars.index[-1]
    date_object = datetime.strptime(rtdate.replace('/',' '), '%m %d %Y')
    rtdf = pd.date_range(date_object, date_object)

    date_append = False
    bars_org = bars_org.sort_index()
    # print len(bars_org),len(bars_org['Close']),len(bars_org['Volume'])
    if date_object > bars_org.index[-1]:
        d={'Open':rtopen, 'High':rthigh,'Low':rtlow,'Close':rtclose,'Volume':rtvolume }
        appenddf = pd.DataFrame(d,index=rtdf)
        appenddf.index.name = 'Date'
        date_append = True
        # print appenddf,date_append
        # print bars_org.index.isin(appenddf.index)
    
        bars = pd.concat([bars_org,appenddf])
        # print '----------'
        # print bars.tail()
    else:
        bars = bars_org
    
    
    # bars = []
    # startdate = '2013-01-01'
    # enddate = '2013-06-01'
    # bars =  Quandl.get(symbol, collapse='Daily', trim_start=startdate, trim_end=enddate,authtoken="")
    bars = bars.drop_duplicates()
    bars = bars.sort_index()
    # print '----------final sorted bars---------'
    # print bars.tail()

    return bars



def readStdargs(typearg,namearg):
    print 'read Std DB args , 25p 50p 75p 90p'
    try:

        if typearg == 1:
            dbname = '../../data/analysis/analysis_std_db.sqlite'
        elif typearg == 2:
            dbname = '../../data/analysis/analysis_kosdaq_std_db.sqlite'
        con = sqlite3.connect(dbname)

        tablename = 'analysis_table'
        print 'read table'
        analysisdf = pd_sql.read_frame("SELECT * from "+tablename, con)
        con.close()
        
        # print 'readStdargs namearg',namearg,analysisdf[analysisdf['title'] == namearg]
        return analysisdf[analysisdf['title'] == namearg]['bars_25p'].values[0],\
                analysisdf[analysisdf['title'] == namearg]['bars_50p'].values[0],\
                analysisdf[analysisdf['title'] == namearg]['bars_75p'].values[0],\
                analysisdf[analysisdf['title'] == namearg]['bars_90p'].values[0]
    except Exception,e:
        print e
        con.close()


def readTangentargs(typearg,namearg):
    print 'read Tangent DB args , 25p 50p'
    try:

        if typearg == 1:
            dbname = '../../data/analysis/analysis_tangent_db.sqlite'
        elif typearg == 2:
            dbname = '../../data/analysis/analysis_kosdaq_tangent_db.sqlite'
        con = sqlite3.connect(dbname)

        tablename = 'tangent_table'
        print 'read table'
        analysisdf = pd_sql.read_frame("SELECT * from "+tablename, con)
        con.close()
        
        # print 'readStdargs namearg',namearg,analysisdf[analysisdf['title'] == namearg]
        return analysisdf[analysisdf['title'] == namearg]['tangent_25p'].values[0],\
                analysisdf[analysisdf['title'] == namearg]['tangent_50p'].values[0]
                
    except Exception,e:
        print e
        con.close()



def writeResultDataToDB_algo1(typearg,codearg,namearg,*args):
    print 'write algo1 result data to DB'

    # writeResultDataToDB_algo1(typearg,codearg,namearg,newsigstr_arr1_algo1,newsigstr_arr2_algo1,newsigstr_arr3_algo1,newsigstr_arr4_algo1,newsigstr_arr5_algo1,newsigstr_arr6_algo1 \
    #     ,returns2 ,closepgainnp ,closep\
    #     ,initial_capital ,totalreturngain ,closepgain ,bars ,runcount )

    arg_indx =0
    newsigstr_arr1 = args[arg_indx]
    arg_indx +=1
    newsigstr_arr2 = args[arg_indx]
    arg_indx +=1
    newsigstr_arr3 = args[arg_indx]
    arg_indx +=1
    newsigstr_arr4 = args[arg_indx]
    arg_indx +=1
    newsigstr_arr5 = args[arg_indx]
    arg_indx +=1
    newsigstr_arr6 = args[arg_indx]
    arg_indx +=1
    returns2 = args[arg_indx]
    arg_indx +=1
    closepgainnp = args[arg_indx]
    arg_indx +=1
    closep = args[arg_indx]
    arg_indx +=1
    initial_capital = args[arg_indx]
    arg_indx +=1
    totalreturngain = args[arg_indx]
    arg_indx +=1
    closepgain = args[arg_indx]
    arg_indx +=1
    bars = args[arg_indx]
    arg_indx +=1
    runcount = args[arg_indx]
    arg_indx +=1
    smallarg = args[arg_indx]
    arg_indx +=1
    barsstddf = args[arg_indx]
    arg_indx +=1
    MA1arg = args[arg_indx]
    arg_indx +=1
    MA2arg = args[arg_indx]


    newsigstr_df = pd.DataFrame({'Date':newsigstr_arr1,'Stance':newsigstr_arr2,'Price':newsigstr_arr3,'BuyorSell':newsigstr_arr4,
            'totalGain':newsigstr_arr5,'currentgain':newsigstr_arr6})
    # print newsigstr_df.tail()
    todaydate = datetime.today()
    try:
        if typearg == 1:

            title = '../../data/result/result_db_'+str(todaydate.year)+'{0:02d}'.format(int(todaydate.month))+'{0:02d}'.format(int(todaydate.day))+'.sqlite'

            con = sqlite3.connect(title)
            tablename = 'result_'+codearg+'_'+namearg+'_signal_'+str(todaydate.year)+str(todaydate.month)+str(todaydate.day)+'_'+str(runcount)

            con.execute("DROP TABLE IF EXISTS "+tablename)
            # newsigstr_df.to_sql(tablename, con,False)
            
            pd_sql.write_frame(newsigstr_df, tablename, con)
            
            newsigsum_df = pd.DataFrame({'Benchmark':[closepgainnp[-1]],'Start':[closep[0]],'Current':[closep[-1]]
                ,'totalGain':[(returns2.total[returns2.total.index[-1]] - initial_capital)/initial_capital],'tradingGain':[totalreturngain]})
            # newsigsum_df = pd.DataFrame({'Benchmark':[closepgainnp[-1]]})
            
            tablename = 'result_'+codearg+'_'+namearg+'_summary'+'_'+str(todaydate.year)+str(todaydate.month)+str(todaydate.day)+'_'+str(runcount)
            con.execute("DROP TABLE IF EXISTS "+tablename)
            pd.io.sql.write_frame(newsigsum_df, tablename, con)
          

            newsigbench_df = pd.DataFrame({'Date':bars.index,'Benchmark':closepgain,'stddf':barsstddf,'MA1':MA1arg,'MA2':MA2arg \
                ,'totalReturn':returns2['total'].pct_change().cumsum()},index= bars.index)
            tablename = 'result_'+codearg+'_'+namearg+'_benchmark'+'_'+str(todaydate.year)+str(todaydate.month)+str(todaydate.day)+'_'+str(runcount)
            con.execute("DROP TABLE IF EXISTS "+tablename)
            pd.io.sql.write_frame(newsigbench_df, tablename, con)
            con.close()

            if smallarg == 'small':
                title = '../../data/result/small/result_db_'+str(todaydate.year)+'{0:02d}'.format(int(todaydate.month))+'{0:02d}'.format(int(todaydate.day))+'.sqlite'

                con = sqlite3.connect(title)
                tablename = 'result_'+codearg+'_'+namearg+'_signal_'+str(todaydate.year)+str(todaydate.month)+str(todaydate.day)+'_'+str(runcount)

                con.execute("DROP TABLE IF EXISTS "+tablename)
                # newsigstr_df.to_sql(tablename, con,False)
                
                pd_sql.write_frame(newsigstr_df, tablename, con)
                
                newsigsum_df = pd.DataFrame({'Benchmark':[closepgainnp[-1]],'Start':[closep[0]],'Current':[closep[-1]]
                    ,'totalGain':[(returns2.total[returns2.total.index[-1]] - initial_capital)/initial_capital],'tradingGain':[totalreturngain]})
                # newsigsum_df = pd.DataFrame({'Benchmark':[closepgainnp[-1]]})
                
                tablename = 'result_'+codearg+'_'+namearg+'_summary'+'_'+str(todaydate.year)+str(todaydate.month)+str(todaydate.day)+'_'+str(runcount)
                con.execute("DROP TABLE IF EXISTS "+tablename)
                pd.io.sql.write_frame(newsigsum_df, tablename, con)
              

                newsigbench_df = pd.DataFrame({'Date':bars.index,'Benchmark':closepgain,'stddf':barsstddf,'MA1':MA1arg,'MA2':MA2arg \
                     ,'totalReturn':returns2['total'].pct_change().cumsum()},index= bars.index)
                tablename = 'result_'+codearg+'_'+namearg+'_benchmark'+'_'+str(todaydate.year)+str(todaydate.month)+str(todaydate.day)+'_'+str(runcount)
                con.execute("DROP TABLE IF EXISTS "+tablename)
                pd.io.sql.write_frame(newsigbench_df, tablename, con)

        elif typearg == 2:
            title = '../../data/result/kosdaq/result_kosdaq_db_'+str(todaydate.year)+'{0:02d}'.format(int(todaydate.month))+'{0:02d}'.format(int(todaydate.day))+'.sqlite'
            con = sqlite3.connect(title)
            tablename = 'result_'+codearg+'_'+namearg+'_signal_'+str(todaydate.year)+str(todaydate.month)+str(todaydate.day)+'_'+str(runcount)
            con.execute("DROP TABLE IF EXISTS "+tablename)
            # newsigstr_df.to_sql(tablename, con,False)
            pd_sql.write_frame(newsigstr_df, tablename, con)

            newsigsum_df = pd.DataFrame({'Benchmark':[closepgainnp[-1]],'Start':[closep[0]],'Current':[closep[-1]]
                ,'totalGain':[(returns2.total[returns2.total.index[-1]] - initial_capital)/initial_capital],'tradingGain':[totalreturngain]})
            # newsigsum_df = pd.DataFrame({'Benchmark':[closepgainnp[-1]]})
            
            tablename = 'result_'+codearg+'_'+namearg+'_summary'+'_'+str(todaydate.year)+str(todaydate.month)+str(todaydate.day)+'_'+str(runcount)
            con.execute("DROP TABLE IF EXISTS "+tablename)
            pd.io.sql.write_frame(newsigsum_df, tablename, con)
          

            newsigbench_df = pd.DataFrame({'Date':bars.index,'Benchmark':closepgain,'stddf':barsstddf,'MA1':MA1arg,'MA2':MA2arg \
                ,'totalReturn':returns2['total'].pct_change().cumsum()},index= bars.index)
            tablename = 'result_'+codearg+'_'+namearg+'_benchmark'+'_'+str(todaydate.year)+str(todaydate.month)+str(todaydate.day)+'_'+str(runcount)
            con.execute("DROP TABLE IF EXISTS "+tablename)
            pd.io.sql.write_frame(newsigbench_df, tablename, con)
            con.close()

            if smallarg == 'small':
                title = '../../data/result/kosdaq/small/result_kosdaq_db_'+str(todaydate.year)+'{0:02d}'.format(int(todaydate.month))+'{0:02d}'.format(int(todaydate.day))+'.sqlite'
                con = sqlite3.connect(title)
                tablename = 'result_'+codearg+'_'+namearg+'_signal_'+str(todaydate.year)+str(todaydate.month)+str(todaydate.day)+'_'+str(runcount)
                con.execute("DROP TABLE IF EXISTS "+tablename)
                # newsigstr_df.to_sql(tablename, con,False)
                pd_sql.write_frame(newsigstr_df, tablename, con)

                newsigsum_df = pd.DataFrame({'Benchmark':[closepgainnp[-1]],'Start':[closep[0]],'Current':[closep[-1]]
                    ,'totalGain':[(returns2.total[returns2.total.index[-1]] - initial_capital)/initial_capital],'tradingGain':[totalreturngain]})
                # newsigsum_df = pd.DataFrame({'Benchmark':[closepgainnp[-1]]})
                
                tablename = 'result_'+codearg+'_'+namearg+'_summary'+'_'+str(todaydate.year)+str(todaydate.month)+str(todaydate.day)+'_'+str(runcount)
                con.execute("DROP TABLE IF EXISTS "+tablename)
                pd.io.sql.write_frame(newsigsum_df, tablename, con)
              

                newsigbench_df = pd.DataFrame({'Date':bars.index,'Benchmark':closepgain,'stddf':barsstddf,'MA1':MA1arg,'MA2':MA2arg \
                    ,'totalReturn':returns2['total'].pct_change().cumsum()},index= bars.index)
                tablename = 'result_'+codearg+'_'+namearg+'_benchmark'+'_'+str(todaydate.year)+str(todaydate.month)+str(todaydate.day)+'_'+str(runcount)
                con.execute("DROP TABLE IF EXISTS "+tablename)
                pd.io.sql.write_frame(newsigbench_df, tablename, con)
        elif typearg == 3:
            title = '../../data/result/result_kospi200_db_'+str(todaydate.year)+'{0:02d}'.format(int(todaydate.month))+'{0:02d}'.format(int(todaydate.day))+'_'+'{0:02d}'.format(int(todaydate.hour))\
            +'{0:02d}'.format(int(todaydate.minute))+'.sqlite'
            con = sqlite3.connect(title)
            tablename = 'result_'+codearg+'_'+namearg+'_signal_'+str(todaydate.year)+str(todaydate.month)+str(todaydate.day)+'_'+str(runcount)
            con.execute("DROP TABLE IF EXISTS "+tablename)
            # newsigstr_df.to_sql(tablename, con,False)
            pd_sql.write_frame(newsigstr_df, tablename, con)

            newsigsum_df = pd.DataFrame({'Benchmark':[closepgainnp[-1]],'Start':[closep[0]],'Current':[closep[-1]]
                ,'totalGain':[(returns2.total[returns2.total.index[-1]] - initial_capital)/initial_capital],'tradingGain':[totalreturngain]})
            # newsigsum_df = pd.DataFrame({'Benchmark':[closepgainnp[-1]]})
            
            tablename = 'result_'+codearg+'_'+namearg+'summary'+'_'+str(todaydate.year)+str(todaydate.month)+str(todaydate.day)+'_'+str(runcount)
            con.execute("DROP TABLE IF EXISTS "+tablename)
            pd.io.sql.write_frame(newsigsum_df, tablename, con)
          

            newsigbench_df = pd.DataFrame({'Date':bars.index,'Benchmark':closepgain,'totalReturn':returns2['total'].pct_change().cumsum()},index= bars.index)
            tablename = 'result_'+codearg+'_'+namearg+'benchmark'+'_'+str(todaydate.year)+str(todaydate.month)+str(todaydate.day)+'_'+str(runcount)
            con.execute("DROP TABLE IF EXISTS "+tablename)
            pd.io.sql.write_frame(newsigbench_df, tablename, con)        
    except Exception,e:
        print 'db write exception:',e 
        con.close()

    con.close()

def writeResultDataToDB_algo2(typearg,codearg,namearg,*args):
    print 'write algo2 result data to DB'

    arg_indx =0
    newsigstr_arr1 = args[arg_indx]
    arg_indx +=1
    newsigstr_arr2 = args[arg_indx]
    arg_indx +=1
    newsigstr_arr3 = args[arg_indx]
    arg_indx +=1
    newsigstr_arr4 = args[arg_indx]
    arg_indx +=1
    newsigstr_arr5 = args[arg_indx]
    arg_indx +=1
    newsigstr_arr6 = args[arg_indx]
    arg_indx +=1
    returns3 = args[arg_indx]
    arg_indx +=1
    closepgainnp = args[arg_indx]
    arg_indx +=1
    closep = args[arg_indx]
    arg_indx +=1
    initial_capital = args[arg_indx]
    arg_indx +=1
    totalreturngain = args[arg_indx]
    arg_indx +=1
    closepgain = args[arg_indx]
    arg_indx +=1
    bars = args[arg_indx]
    arg_indx +=1
    runcount = args[arg_indx]
    arg_indx +=1
    smallarg = args[arg_indx]
    arg_indx +=1
    barsstddf = args[arg_indx]
    arg_indx +=1
    MA1arg = args[arg_indx]
    arg_indx +=1
    MA2arg = args[arg_indx]

    newsigstr_df = pd.DataFrame({'Date':newsigstr_arr1,'Stance':newsigstr_arr2,'Price':newsigstr_arr3,'BuyorSell':newsigstr_arr4,
            'totalGain':newsigstr_arr5,'currentgain':newsigstr_arr6})
    # print newsigstr_df.tail()
    todaydate = datetime.today()
    try:
        if typearg == 1:

            title = '../../data/result2/result_db_'+str(todaydate.year)+'{0:02d}'.format(int(todaydate.month))+'{0:02d}'.format(int(todaydate.day))+'.sqlite'

            con = sqlite3.connect(title)
            tablename = 'result_'+codearg+'_'+namearg+'_signal_'+str(todaydate.year)+str(todaydate.month)+str(todaydate.day)+'_'+str(runcount)

            con.execute("DROP TABLE IF EXISTS "+tablename)
            # newsigstr_df.to_sql(tablename, con,False)
            
            pd_sql.write_frame(newsigstr_df, tablename, con)
            
            newsigsum_df = pd.DataFrame({'Benchmark':[closepgainnp[-1]],'Start':[closep[0]],'Current':[closep[-1]]
                ,'totalGain':[(returns3.total[returns3.total.index[-1]] - initial_capital)/initial_capital],'tradingGain':[totalreturngain]})
            # newsigsum_df = pd.DataFrame({'Benchmark':[closepgainnp[-1]]})
            
            tablename = 'result_'+codearg+'_'+namearg+'_summary'+'_'+str(todaydate.year)+str(todaydate.month)+str(todaydate.day)+'_'+str(runcount)
            con.execute("DROP TABLE IF EXISTS "+tablename)
            pd.io.sql.write_frame(newsigsum_df, tablename, con)
          

            newsigbench_df = pd.DataFrame({'Date':bars.index,'Benchmark':closepgain,'stddf':barsstddf,'MA1':MA1arg,'MA2':MA2arg \
                ,'totalReturn':returns3['total'].pct_change().cumsum()},index= bars.index)
            tablename = 'result_'+codearg+'_'+namearg+'_benchmark'+'_'+str(todaydate.year)+str(todaydate.month)+str(todaydate.day)+'_'+str(runcount)
            con.execute("DROP TABLE IF EXISTS "+tablename)
            pd.io.sql.write_frame(newsigbench_df, tablename, con)
            con.close()
    
            if smallarg == 'small':

                title = '../../data/result2/small/result_db_'+str(todaydate.year)+'{0:02d}'.format(int(todaydate.month))+'{0:02d}'.format(int(todaydate.day))+'.sqlite'            
                con = sqlite3.connect(title)
                tablename = 'result_'+codearg+'_'+namearg+'_signal_'+str(todaydate.year)+str(todaydate.month)+str(todaydate.day)+'_'+str(runcount)

                con.execute("DROP TABLE IF EXISTS "+tablename)
                # newsigstr_df.to_sql(tablename, con,False)
                
                pd_sql.write_frame(newsigstr_df, tablename, con)
                
                newsigsum_df = pd.DataFrame({'Benchmark':[closepgainnp[-1]],'Start':[closep[0]],'Current':[closep[-1]]
                    ,'totalGain':[(returns3.total[returns3.total.index[-1]] - initial_capital)/initial_capital],'tradingGain':[totalreturngain]})
                # newsigsum_df = pd.DataFrame({'Benchmark':[closepgainnp[-1]]})
                
                tablename = 'result_'+codearg+'_'+namearg+'_summary'+'_'+str(todaydate.year)+str(todaydate.month)+str(todaydate.day)+'_'+str(runcount)
                con.execute("DROP TABLE IF EXISTS "+tablename)
                pd.io.sql.write_frame(newsigsum_df, tablename, con)
              

                newsigbench_df = pd.DataFrame({'Date':bars.index,'Benchmark':closepgain,'stddf':barsstddf,'MA1':MA1arg,'MA2':MA2arg \
                    ,'totalReturn':returns3['total'].pct_change().cumsum()},index= bars.index)
                tablename = 'result_'+codearg+'_'+namearg+'_benchmark'+'_'+str(todaydate.year)+str(todaydate.month)+str(todaydate.day)+'_'+str(runcount)
                con.execute("DROP TABLE IF EXISTS "+tablename)
                pd.io.sql.write_frame(newsigbench_df, tablename, con)

            
        elif typearg == 2:
            title = '../../data/result2/kosdaq/result_kosdaq_db_'+str(todaydate.year)+'{0:02d}'.format(int(todaydate.month))+'{0:02d}'.format(int(todaydate.day))+'.sqlite'
            con = sqlite3.connect(title)
            tablename = 'result_'+codearg+'_'+namearg+'_signal_'+str(todaydate.year)+str(todaydate.month)+str(todaydate.day)+'_'+str(runcount)
            con.execute("DROP TABLE IF EXISTS "+tablename)
            # newsigstr_df.to_sql(tablename, con,False)
            pd_sql.write_frame(newsigstr_df, tablename, con)

            newsigsum_df = pd.DataFrame({'Benchmark':[closepgainnp[-1]],'Start':[closep[0]],'Current':[closep[-1]]
                ,'totalGain':[(returns3.total[returns3.total.index[-1]] - initial_capital)/initial_capital],'tradingGain':[totalreturngain]})
            # newsigsum_df = pd.DataFrame({'Benchmark':[closepgainnp[-1]]})
            
            tablename = 'result_'+codearg+'_'+namearg+'_summary'+'_'+str(todaydate.year)+str(todaydate.month)+str(todaydate.day)+'_'+str(runcount)
            con.execute("DROP TABLE IF EXISTS "+tablename)
            pd.io.sql.write_frame(newsigsum_df, tablename, con)
          

            newsigbench_df = pd.DataFrame({'Date':bars.index,'Benchmark':closepgain,'stddf':barsstddf,'MA1':MA1arg,'MA2':MA2arg \
                ,'totalReturn':returns3['total'].pct_change().cumsum()},index= bars.index)
            tablename = 'result_'+codearg+'_'+namearg+'_benchmark'+'_'+str(todaydate.year)+str(todaydate.month)+str(todaydate.day)+'_'+str(runcount)
            con.execute("DROP TABLE IF EXISTS "+tablename)
            pd.io.sql.write_frame(newsigbench_df, tablename, con)
            con.close()

            if smallarg == 'small':    
                title = '../../data/result2/kosdaq/small/result_kosdaq_db_'+str(todaydate.year)+'{0:02d}'.format(int(todaydate.month))+'{0:02d}'.format(int(todaydate.day))+'.sqlite'
                con = sqlite3.connect(title)
                tablename = 'result_'+codearg+'_'+namearg+'_signal_'+str(todaydate.year)+str(todaydate.month)+str(todaydate.day)+'_'+str(runcount)
                con.execute("DROP TABLE IF EXISTS "+tablename)
                # newsigstr_df.to_sql(tablename, con,False)
                pd_sql.write_frame(newsigstr_df, tablename, con)

                newsigsum_df = pd.DataFrame({'Benchmark':[closepgainnp[-1]],'Start':[closep[0]],'Current':[closep[-1]]
                    ,'totalGain':[(returns3.total[returns3.total.index[-1]] - initial_capital)/initial_capital],'tradingGain':[totalreturngain]})
                # newsigsum_df = pd.DataFrame({'Benchmark':[closepgainnp[-1]]})
                
                tablename = 'result_'+codearg+'_'+namearg+'_summary'+'_'+str(todaydate.year)+str(todaydate.month)+str(todaydate.day)+'_'+str(runcount)
                con.execute("DROP TABLE IF EXISTS "+tablename)
                pd.io.sql.write_frame(newsigsum_df, tablename, con)
              

                newsigbench_df = pd.DataFrame({'Date':bars.index,'Benchmark':closepgain,'stddf':barsstddf,'MA1':MA1arg,'MA2':MA2arg \
                    ,'totalReturn':returns3['total'].pct_change().cumsum()},index= bars.index)
                tablename = 'result_'+codearg+'_'+namearg+'_benchmark'+'_'+str(todaydate.year)+str(todaydate.month)+str(todaydate.day)+'_'+str(runcount)
                con.execute("DROP TABLE IF EXISTS "+tablename)
                pd.io.sql.write_frame(newsigbench_df, tablename, con)

        elif typearg == 3:
            title = '../../data/result2/result_kospi200_db_'+str(todaydate.year)+'{0:02d}'.format(int(todaydate.month))+'{0:02d}'.format(int(todaydate.day))+'_'+'{0:02d}'.format(int(todaydate.hour))\
            +'{0:02d}'.format(int(todaydate.minute))+'.sqlite'
            con = sqlite3.connect(title)
            tablename = 'result_'+codearg+'_'+namearg+'_signal_'+str(todaydate.year)+str(todaydate.month)+str(todaydate.day)+'_'+str(runcount)
            con.execute("DROP TABLE IF EXISTS "+tablename)
            # newsigstr_df.to_sql(tablename, con,False)
            pd_sql.write_frame(newsigstr_df, tablename, con)

            newsigsum_df = pd.DataFrame({'Benchmark':[closepgainnp[-1]],'Start':[closep[0]],'Current':[closep[-1]]
                ,'totalGain':[(returns3.total[returns3.total.index[-1]] - initial_capital)/initial_capital],'tradingGain':[totalreturngain]})
            # newsigsum_df = pd.DataFrame({'Benchmark':[closepgainnp[-1]]})
            
            tablename = 'result_'+codearg+'_'+namearg+'summary'+'_'+str(todaydate.year)+str(todaydate.month)+str(todaydate.day)+'_'+str(runcount)
            con.execute("DROP TABLE IF EXISTS "+tablename)
            pd.io.sql.write_frame(newsigsum_df, tablename, con)
          

            newsigbench_df = pd.DataFrame({'Date':bars.index,'Benchmark':closepgain,'totalReturn':returns3['total'].pct_change().cumsum()},index= bars.index)
            tablename = 'result_'+codearg+'_'+namearg+'benchmark'+'_'+str(todaydate.year)+str(todaydate.month)+str(todaydate.day)+'_'+str(runcount)
            con.execute("DROP TABLE IF EXISTS "+tablename)
            pd.io.sql.write_frame(newsigbench_df, tablename, con)        
    except Exception,e:
        print 'db write exception:',e 
        con.close()

    con.close()



def RunSimul_std(codearg,typearg,namearg,mode,dbmode,histmode,srcsite,startdatemode):


    code = codearg #'097950'#'005930' #'005380'#009540 #036570
    if codearg == '005490' or codearg == '000660' or codearg == '068870'\
        or codearg == '078520':
        srcsite = 2
        
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

    if startdatemode == 1:            
        startdate = '2011-01-01'
    else:
        startday =  datetime.today() - timedelta(days=150)
        startdate = str(startday).split(' ')[0]
    print 'startdate',startdate

    
    
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

    
    date_object = datetime.strptime(rtdate.replace('/',' '), '%m %d %Y')
    rtdf = pd.date_range(date_object, date_object)

    date_append = False
    
    if date_object > bars_org.index[-1]:
        d={'Open':rtopen, 'High':rthigh,'Low':rtlow,'Close':rtclose,'Volume':rtvolume }
        appenddf = pd.DataFrame(d,index=rtdf)
        appenddf.index.name = 'Date'
        date_append = True
        print appenddf,date_append
        bars = pd.concat([bars_org,appenddf])
        print '----------'
    
    else:
        bars = bars_org

    
    bars = bars.sort_index()
    print '----------final sorted bars---------'

    print bars.tail()
    

    bars['Std'] = 0
    bars['Avg'] = 0
    bars['CumStd'] = 0
    bars['VolPrice'] = 0
    print 'bars len',len(bars)

    for day in range(len(bars)):
        try:
        #     print bars['Close'][day]
            if day <=1:
                bars['Avg'][day] = bars['Close'][day]
            if day > 1 and day <= 20:
                bars['Std'][day] = bars['Close'][:day].std()
                bars['Avg'][day] = bars['Close'][:day].mean()
                if day == 2:
                    bars['Std'][0] = bars['Std'][2]
                    bars['Std'][1] = bars['Std'][2]
            if day > 20:
                bars['Std'][day] = bars['Close'][day-20:day].std()
                bars['Avg'][day] = bars['Close'][day-20:day].mean()

            if day > 1:
                bars['CumStd'] = bars['Close'][:day].std()

            ''' volume '''
            if day <= 1:
                bars['VolPrice'][day] = bars['Close'][day]
            if day > 1 and day <= 20:
                volmean = bars['Volume'][1:day].mean()
                if len(bars['Volume'][1:day][bars['Volume'][1:day] > volmean].index) > 0 :
                    bars['VolPrice'][day] = bars['Close'][bars['Volume'][1:day][bars['Volume'][1:day] > volmean].index].mean()
                else:
                    bars['VolPrice'][day] = bars['Close'][day]
                
                # print volmean,bars['Volume'][day]
                # print bars.index[day]
                # print bars['Close'][bars['Volume'][:day][bars['Volume'][:day] > volmean].index]
            elif day >20:
                volmean = bars['Volume'][day-20:day].mean()
                bars['VolPrice'][day] = bars['Close'][bars['Volume'][day-20:day][bars['Volume'][day-20:day] > volmean].index].mean()    
        except Exception,e:
            print 'error ',e


    ''' std inflection point'''    
    bars2 = deepcopy(bars)
    # bars2 = bars2.astype(np.float64,copy = False)
    # barsStdnp = (bars2['Std'].values + bars2['Std'].values) /(bars2['Avg'].values-bars2['Std'].values)

    bars2['Std'] = bars2['Std'].astype(np.float64)
    bars2['Avg'] = bars2['Avg'].astype(np.float64)
    barsStddf = bars2['Stdsig'] = (bars2['Std'] + bars2['Std']) /(bars2['Avg']-bars2['Std'])
    
    bars_25p = bars2['Stdsig'].describe()['25%']
    bars_50p = bars2['Stdsig'].describe()['50%']
    bars_75p = bars2['Stdsig'].describe()['75%']
    bars_90p = bars2['Stdsig'].quantile(0.9)
    
    print 'bars 25%:',bars_25p,'bars_50p:',bars_50p,'bars 75%:',bars_75p,'bars_90p:',bars_90p

    
    print '----------------------------Return Std Bars-------------------------'    
    return bars_25p,bars_50p,bars_75p,bars_90p



def RunSimul_realData(codearg,typearg,namearg,mode,dbmode,histmode,srcsite,startdatemode):


    code = codearg #'097950'#'005930' #'005380'#009540 #036570
    if codearg == '005490' or codearg == '000660' or codearg == '068870'\
        or codearg == '078520':
        srcsite = 2
        
    if srcsite == 1:
        if typearg == 1:
            symbol = 'GOOG/KRX_'+code
        elif typearg == 2:
            symbol = 'GOOG/KOSDAQ_'+code
        elif typearg == 3:
            symbol = 'GOOG/INDEXKRX_KOSPI200'  
        elif typearg == 4:
            symbol = 'GOOG/INDEXKRX_KOSDAQ'
    elif srcsite == 2:
        if typearg == 1:
            symbol = 'YAHOO/KS_'+code
        elif typearg == 2:
            symbol = 'YAHOO/KQ_'+code
        elif typearg == 3:
            symbol = 'YAHOO/INDEX_KS11'  

    if startdatemode == 1:            
        startdate = '2011-01-01'
    elif startdatemode == 3:            
        startdate = '2007-01-01'    
    else:
        startday =  datetime.today() - timedelta(days=150)
        startdate = str(startday).split(' ')[0]
    print 'startdate',startdate

    
    
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
    elif typearg == 4:
        rtsymbol = '^KQ11'
    # rtsymbol = '^KS200'
    realtimeURL = 'http://finance.yahoo.com/d/quotes.csv?s='+rtsymbol+'&f=sl1d1t1c1ohgv&e=.csv'
    # print realtimeURL
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
        # print 'rtsymbol:',rtsymbol,'rtclose:',rtclose,rtdate,rttime,rtchange,'rtopen:',rtopen,'rthigh:',rthigh,'rtlow:',rtlow,'rtvolume:',rtvolume

    
    date_object = datetime.strptime(rtdate.replace('/',' '), '%m %d %Y')
    rtdf = pd.date_range(date_object, date_object)

    date_append = False
    

    bars_org = bars_org.sort_index()
    # print len(bars_org),len(bars_org['Close']),len(bars_org['Volume'])
    if date_object > bars_org.index[-1]:
        d={'Open':rtopen, 'High':rthigh,'Low':rtlow,'Close':rtclose,'Volume':rtvolume }
        appenddf = pd.DataFrame(d,index=rtdf)
        appenddf.index.name = 'Date'
        date_append = True
        # print appenddf,date_append
        # print bars_org.index.isin(appenddf.index)
    
        bars = pd.concat([bars_org,appenddf])
        # print '----------'
        # print bars.tail()
    else:
        bars = bars_org

    # bars = []
    # startdate = '2013-01-01'
    # enddate = '2013-06-01'
    # bars =  Quandl.get(symbol, collapse='Daily', trim_start=startdate, trim_end=enddate,authtoken="")
    bars = bars.drop_duplicates()
    bars = bars.sort_index()
    
    # print '----------final sorted bars---------'
    # print bars.tail()

    return bars


from numpy import arange
from numpy import sin,linspace,power
from scipy import interpolate
from pylab import plot,show
def draw_tangent(x,y,a):
    # interpolate the data with a spline
    spl = interpolate.splrep(x,y)
    small_t = arange(a-5,a+5)
    fa = interpolate.splev(a,spl,der=0)     # f(a)
    fprime = interpolate.splev(a,spl,der=1) # f'(a)
    tan = fa+fprime*(small_t-a) # tangent
    # ax.plot(a,fa,'om',small_t,tan,'--r')
    return fprime


def RunSimul_tangent(codearg,typearg,namearg,mode,dbmode,histmode,srcsite,startdatemode):


    code = codearg #'097950'#'005930' #'005380'#009540 #036570
    if codearg == '005490' or codearg == '000660' or codearg == '068870'\
        or codearg == '078520':
        srcsite = 2
        
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

    if startdatemode == 1:            
        startdate = '2011-01-01'
    else:
        startday =  datetime.today() - timedelta(days=150)
        startdate = str(startday).split(' ')[0]
    print 'startdate',startdate

    
    
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

    
    date_object = datetime.strptime(rtdate.replace('/',' '), '%m %d %Y')
    rtdf = pd.date_range(date_object, date_object)

    date_append = False
    
    bars_org = bars_org.sort_index()
    # print len(bars_org),len(bars_org['Close']),len(bars_org['Volume'])
    if date_object > bars_org.index[-1]:
        d={'Open':rtopen, 'High':rthigh,'Low':rtlow,'Close':rtclose,'Volume':rtvolume }
        appenddf = pd.DataFrame(d,index=rtdf)
        appenddf.index.name = 'Date'
        date_append = True
        print appenddf,date_append
        # print bars_org.index.isin(appenddf.index)
    
        bars = pd.concat([bars_org,appenddf])
        print '----------'
        # print bars.tail()
    else:
        bars = bars_org

    # bars = []
    # startdate = '2013-01-01'
    # enddate = '2013-06-01'
    # bars =  Quandl.get(symbol, collapse='Daily', trim_start=startdate, trim_end=enddate,authtoken="")
    bars = bars.drop_duplicates()
    bars = bars.sort_index()
    print '----------final sorted bars---------'
    print bars.tail()

    bars2 = deepcopy(bars)
    print 'bars2 start'
    bars2['OpenCloseSig'] = 0
    bars2['OpenCloseSig'] = bars2['Close']- bars2['Open']
    bars2['OpenCloseSig'] = np.where(bars2['OpenCloseSig'] > 0.0, 1.0, -1.0)  
    bars2['OBV'] = bars2['Volume']*bars2['OpenCloseSig']
    bars2['OBV'] = bars2['OBV'].cumsum()
    # bars2['VolAvg'] =  [bars2['Volume'][day-20:day].mean() for day in range(len(bars['Close'])) ]
    # bars2['VolStdSig'] =  (bars2['VolStd'] + bars2['VolStd']) /(bars2['VolAvg']-bars2['VolStd'])
    # bars2['StdCorr'] = [bars2['Stdsig'][:day].corr(bars2['OBV'][:day]) for day in range(len(bars['Close'])) ]
    bars['OBV'] = bars2['OBV']
    barsOBVMA1 = pd.rolling_mean(bars['OBV'], 5, min_periods=1).fillna(0.0)
    barsobvma1np = barsOBVMA1.values 
    barsobvnp = bars['OBV'].values 
    print 'bars obv processing end'

    tix = np.arange(0,len(bars))
    # tix = [t.value/(10**9) for t in bars.index]
    tangents = []
    # for day in tix:
    for day in range(len(bars)):
        tangents.append(draw_tangent(tix,bars2['OBV'].values,day))

    nptangents = np.array(tangents)
    dftangents = pd.DataFrame({'tangent':nptangents})
    tangent_25p = dftangents[dftangents['tangent'] > 0.0].quantile(0.25).values[0]
    tangent_50p = dftangents[dftangents['tangent'] > 0.0].quantile(0.5).values[0]

    
    print 'bars tangent 25%:',tangent_25p,'tangent bars_50p:',tangent_50p

    
    print '----------------------------Return Tangent Bars-------------------------'    
    return tangent_25p,tangent_50p



def _inRunSimul_FetchData(codearg,typearg,namearg,mode,dbmode,histmode,runcount,srcsite,writedblog,updbpattern,appenddb,startdatemode 
    ,dbtradinghist,plotly,*args):
    print '_inRunSimul'
    arg_index = 0
    stdarg = args[arg_index]
    arg_index += 1
    smallarg = args[arg_index]
    arg_index += 1
    dayselect = args[arg_index]
    arg_index += 1
    tangentmode = args[arg_index]
    

    # print 'stdarg',stdarg,'smallarg',smallarg,'dayselect',dayselect,'tangentmode',tangentmode
    patternAr = 0
    extractid = 0
    patternAppendAr = 0
    bars_25p =0 
    bars_50p =0 
    bars_75p =0 
    bars_90p =0 
    tangent_25p =0 
    tangent_50p =0 
    startdate =0 
    

    # if stdarg == 'stddb':
    #     bars_25p,bars_50p,bars_75p,bars_90p = readStdargs(typearg,namearg)

    # print 'tangentmode',tangentmode    
    # if tangentmode == 'tangentdb':
    #     try:
    #         tangent_25p,tangent_50p = readTangentargs(typearg,namearg)
    #     except Exception,e:
    #         tangentmode = 'tan_gen'
        
    code = codearg #'097950'#'005930' #'005380'#009540 #036570
    if codearg == '005490' or codearg == '000660' or codearg == '068870'\
        or codearg == '078520' :
        srcsite = 2
        
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

    if startdatemode == 1:            
        startdate = '2011-01-01'
    elif startdatemode == 3:
        arg_index += 1
        startdate_arg = args[arg_index]
        arg_index += 1
        endday_arg = args[arg_index]
        startdate = startdate_arg
    else:
        # startday =  datetime.today() - timedelta(days=30*12)
        startday =  dt.date(2016,1,1)
        startdate = str(startday).split(' ')[0]
        
    print 'startdate',startdate
    # if mode =='dbpattern'  or dbmode == 'dbpattern':        
    #     if updbpattern == 'none':
    #         print 'read DB patterns'
    #         patternAr, extractid= ReadPatternsFromDB(codearg,typearg,namearg,mode)
    #         patternAppendAr = ReadPatternsAppendFromDB(codearg,namearg)
    #     elif updbpattern == 'updbpattern':
    #         print 'read UP DB patterns'
    #         patternAr, extractid= ReadUpPatternsFromDB(codearg,typearg,namearg,mode)

    #     if patternAr == -1:
    #         print 'real time gen db pattern'
    #         startdate = '2011-01-01'
    #         dbmode  = 'none'

    
    # enddate = '2008-12-30'
    print symbol,namearg,mode,histmode,startdate
    try:
        if mode == 'realtime':
            if histmode == 'none':
                bars_org = fetchRealData(code,symbol,typearg,startdate)
                # print 'fetch real data',bars_org.tail()
            elif histmode == 'histdb':
                try:
                    bars_org = fetchHistData(codearg,namearg,symbol,startdate)
                    bars = bars_org
                except:
                    bars_org = fetchRealData(code,symbol,typearg,startdate)
                    bars = bars_org
            elif histmode == 'histdb2':
                try:
                    bars_org = fetchHistData2(codearg,namearg,symbol,startdate)
                    bars = bars_org
                except:
                    bars_org = fetchRealData(code,symbol,typearg,startdate)
                    bars = bars_org        
        elif mode =='dbpattern':
            bars_org = ReadHistFromDB(codearg,typearg,namearg,mode)    
    except Exception,e:
        print 'fetch data error ',e

    start = time.clock()
    if histmode == 'none':
        
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
        
        bars_org = bars_org.sort_index()
        # print len(bars_org),len(bars_org['Close']),len(bars_org['Volume'])
        if date_object > bars_org.index[-1]:
            d={'Open':rtopen, 'High':rthigh,'Low':rtlow,'Close':rtclose,'Volume':rtvolume }
            appenddf = pd.DataFrame(d,index=rtdf)
            appenddf.index.name = 'Date'
            date_append = True
            print appenddf,date_append
            # print bars_org.index.isin(appenddf.index)
        
            bars = pd.concat([bars_org,appenddf])
            print '----------'
            # print bars.tail()
        else:
            bars = bars_org
        


    # bars = []
    # startdate = '2013-01-01'
    # enddate = '2013-06-01'
    # bars =  Quandl.get(symbol, collapse='Daily', trim_start=startdate, trim_end=enddate,authtoken="")
    bars = bars.drop_duplicates()
    bars = bars.sort_index()
    print '----------final sorted bars---------'
    if startdatemode == 3:
        # endday_arg = datetime.datetime.strptime(endday_arg, "%Y-%m-%d %H:%M:%S")
        bars = bars[bars.index <= endday_arg]
    print bars.tail()

    elapsed = (time.clock() - start)
    print 'real time data web gathering elapsed time:',elapsed

    return bars,patternAr, extractid,patternAppendAr,bars_25p,bars_50p\
    ,bars_75p,bars_90p,tangent_25p,tangent_50p,tangentmode\
    ,startdate,dbmode,stdarg,smallarg,dayselect,tangentmode
    
def _inRunSimul_AlgoRun(codearg,typearg,namearg,mode,dbmode,histmode,runcount,srcsite,writedblog,updbpattern,appenddb,startdatemode 
    ,dbtradinghist,plotly,*args):
    print '_inRunSimul_AlgoRun'
    
def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print 'EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj)


def assure_path_exists(path):
    # dir = os.path.dirname(path)
    if not os.path.exists(path):
        os.makedirs(path)
