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
# from guiqwt.plot import CurveDialog
# from guiqwt.builder import make


class PatternData:
    def __init__(self,df):
        self.patterndf = df


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
        positions[self.symbol] = 10*self.signals['signal']   # This strategy buys 100 shares
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
        
        positions[self.symbol] = 100*self.signals   # This strategy buys 10 shares
        #positions['test'] has 10 shares * (1 or 0)
        # print positions['2012-05-10':'2012-06-10']
        return positions
                    
    def backtest_portfolio(self):
        #print len(self.positions),len(self.ms['Price'])
        portfolio = self.positions*self.ms['Price']
        #portfolio has (10 shares or 0) * price 

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
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        portfolio['returns'] = portfolio['total'].pct_change()
        #print portfolio.tail()
        #print portfolio.head()
        return portfolio


def patternRecAuto(gpatternAr,gbarsdf):
    print 'pattern matching'

    # global gpatternAr
    lpatternAr = gpatternAr
    totallen = len(gpatternAr)
    # global gbarsdf
    # print lpatternAr[num].patterndf
    # print(num,showaccum)
    
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
    
    global gextractid
    for num in gextractid:
    # for num in range(totallen):

        pattern0 = lpatternAr[num].patterndf.reset_index()
        
        pattern0_openpc = pattern0['Open'].pct_change().cumsum()   
        openpc0 = pd.DataFrame(pattern0_openpc,index = pattern0.index,columns=['OpenPc']).fillna(0.0)

        pattern0_closepc = pattern0['Close'].pct_change().cumsum()   
        closepc0 = pd.DataFrame(pattern0_closepc,index = pattern0.index,columns=['ClosePc']).fillna(0.0)

        pattern0_highpc = pattern0['High'].pct_change().cumsum()   
        highpc0 = pd.DataFrame(pattern0_highpc,index = pattern0.index,columns=['HighPc']).fillna(0.0)

        pattern0_lowpc = pattern0['Low'].pct_change().cumsum()   
        lowpc0 = pd.DataFrame(pattern0_lowpc,index = pattern0.index,columns=['LowPc']).fillna(0.0)
        
        pattern0_volpc = pattern0['Volume'].pct_change().cumsum()   
        volpc0 = pd.DataFrame(pattern0_volpc,index = pattern0.index,columns=['VolPc']).fillna(0.0)
        
        patternall = pd.concat([pattern0,openpc0,closepc0,highpc0,lowpc0,volpc0],axis=1)
    
        closecorr = closex1['ClosePc'].corr(closepc0['ClosePc'])
        opencorr = openx1['OpenPc'].corr(openpc0['OpenPc'])    
        highcorr = highx1['HighPc'].corr(highpc0['HighPc'])    
        lowcorr = lowx1['LowPc'].corr(lowpc0['LowPc'])
        volcorr = volx1['VolPc'].corr(volpc0['VolPc'])

        # print 'close corr:',closecorr,' open corr:',opencorr,' high corr:',highcorr,' low corr:',lowcorr,' vol corr:',volcorr
        if closecorr > 0.8 \
            and volcorr > 0.8\
            and opencorr > 0.8\
            and highcorr > 0.8\
            and lowcorr > 0.8:
        # corrsum = (closecorr+opencorr+highcorr+lowcorr+volcorr)/5
        # if corrsum > 0.5:
            print 'found pattern :',num
            return num
            break
    print 'loop end!!'    
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
            print 'gain:',periodgain,barsdf.index[daycnt],npclose[daycnt],benchgainnp[daycnt]\
                ,'negnum:',negnum,' posnum:',posnum

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


def dateParser(s):
    #return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")
    return datetime(int(s[0:4]), int(s[5:7]), int(s[8:10]))*1000
    #return np.datetime64(s)
    #return pandas.Timestamp(s, "%Y-%m-%d %H:%M:%S.%f", tz='utc' )

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

global goodcasedf
goodcasedf = 0
global fixcasedf
fixcasedf = 0
global gbarsdf
gbarsdf = 0
global gpatternAr
gpatternAr = 0

def RunSimul(codearg,typearg,namearg):

    code = codearg #'097950'#'005930' #'005380'#009540 #036570
    if typearg == 1:
        symbol = 'GOOG/KRX_'+code
    elif typearg == 2:
        symbol = 'GOOG/KOSDAQ_'+code
    elif typearg == 3:
        symbol = 'GOOG/INDEXKRX_KOSPI200'  
    # symbol = 'GOOG/INDEXKRX_KOSPI200'
    startdate = '2009-01-01'
    # enddate = '2008-12-30'
    print symbol
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
        print rtsymbol,rtclose,rtdate,rttime,rtchange,rtopen,rthigh,rtlow,rtvolume

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


    mac1 = MovingAverageCrossStrategy(
    'test', bars, short_window=5, long_window=30)
    signals = mac1.generate_signals()
    # print signals.tail()
    portfolio = MarketOnClosePortfolio(
        'test', bars, signals, initial_capital=3000000.0)
     
    returns = portfolio.backtest_portfolio()


    '''
    make full time date index to fill saturday,sunday and etc
    '''
    # startdate = datetime(2011, 10, 1)
    # enddate = datetime.today()
    # startdate = datetime(2008, 01, 01)#'2007-10-01'
    # enddate = datetime(2008, 12, 30)#'2009-12-30'

    # npbars = Quandl.get(symbol, collapse='Daily', trim_start=startdate, trim_end=datetime.today(),returns='numpy',authtoken="")
    # new_values=[]
    # for row in npbars:
    #     date, open1,high,low,closep,volume = row
    #     new_values.append([date2num(datetime.strptime(date.replace('-',' '),'%Y %m %d'))
    #                        ,open1
    #                        ,high
    #                        ,low
    #                        ,closep
    #                        ,volume])

    # numArray = np.array(list(new_values))

    # date, open1,high,low,closep,volume = numArray[0:,0],numArray[0:,1],numArray[0:,2],numArray[0:,3],numArray[0:,4],numArray[0:,5]


    # if date_append == True:
    #     closep = np.append(closep,rtclose)
    #     open1 = np.append(open1,rtopen)
    #     high = np.append(high,rthigh)
    #     low = np.append(low,rtlow)
    #     volume = np.append(volume,rtvolume)
    #     date = np.append(date,date2num(datetime.strptime(rtdate.replace('/',' '),'%m %d %Y')))
        # print 'append:',date,date2num(datetime.strptime(rtdate.replace('/',' '),'%m %d %Y'))

    MA1 =5
    MA2 =30

    '''
    draw matplotlib chart
    '''
    #''' plot draw
    fig = plt.figure(figsize=(20, 20))

    fig.patch.set_facecolor('white')     # Set the outer colour to white
    ax1 = fig.add_subplot(511,  ylabel='Price in $')

    # # Plot the AAPL closing price overlaid with the moving averages
    bars['Close'].plot(ax=ax1, color='r', lw=2.)
    signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)

    # # Plot the "buy" trades against AAPL
    ax1.plot(signals.ix[signals.positions == 1.0].index,
              signals.short_mavg[signals.positions == 1.0],
              '^', markersize=10, color='m')

    # # Plot the "sell" trades against AAPL
    ax1.plot(signals.ix[signals.positions == -1.0].index,
              signals.short_mavg[signals.positions == -1.0],
              'v', markersize=10, color='k')


    from matplotlib.finance import candlestick
    from itertools import izip
    ax3 = fig.add_subplot(512,ylabel='Candlestick')
    ax3dates = bars.index.to_pydatetime() 
    ax3times = date2num(ax3dates)
    quotes = izip(ax3times,bars['Open'],bars['Close'],bars['High'],bars['Low'])
    candlestick(ax3,quotes,width=1.5, colorup='g', colordown='r', alpha=1.0)
    ax3.xaxis.set_major_locator(mticker.MaxNLocator(10))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax3.set_xlim(ax3times[0], ax3times[-1])

    ax4 = fig.add_subplot(513,  ylabel='Volume')
    bars['Volume'].plot(ax=ax4, color='#EF15C3', lw=2.)
    # ax4.plot(date,volume,'#EF15C3',label='VolAV1',linewidth=1.5)
    # ax4.xaxis.set_major_locator(mticker.MaxNLocator(10))
    # ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # ax4.set_xlim(date[0], date[-1])

    # datestr1 = datetime.fromtimestamp(time.mktime(mdates.num2date(date[0]).timetuple())).strftime('%Y %m %d')
    # print datestr1,len(bars['Close']),len(date),len(volume)
    # print bars.head()
    # plt.show()
    #'''

    # print pdvol1.ix[pdvol1.longsignal ==1]
    ''' 
    make data frame with buy,sell, poly cross signal
    '''
    sp = pd.DataFrame(bars['Close'],index = bars.index,columns=['Price'])

    ms = pd.concat([sp,signals],axis=1).fillna(0.0)

    '''
    add techinical signal
    '''
    macd, macdsignal, macdhist = talib.MACD(bars['Close'].values, 12, 26, 9)
    slowk,slowd =talib.STOCH(bars['High'].values,bars['Low'].values,bars['Close'].values,10,6,6)
    upperband,middleband,lowerband = talib.BBANDS(bars['Close'].values,10,2,2)
    obvout = talib.OBV(bars['Close'].values,bars['Volume'].values)
    rsiout = talib.RSI(bars['Close'].values,14)
    # wmaout = talib.WMA(bars['Close'],30)
    mfiout = talib.MFI(bars['High'].values,bars['Low'].values,bars['Close'].values,bars['Volume'].values,14)
    dojiout = talib.CDLDOJI(bars['Open'].values,bars['High'].values,bars['Low'].values,bars['Close'].values)
    # marubozuout = talib.CDLMARUBOZU(bars['Open'],bars['High'],bars['Low'],bars['Close'])
    # hammerout = talib.CDLHAMMER(bars['Open'],bars['High'],bars['Low'],bars['Close'])
    # engulfingout = talib.CDLENGULFING(bars['Open'],bars['High'],bars['Low'],bars['Close'])
    # print len(macd),len(obvout),len(mfiout)

    rsidf = pd.DataFrame(rsiout,index = bars.index,columns=['RSI'])
    slowkdf = pd.DataFrame(slowk,index = bars.index,columns=['Slowk'])
    slowddf = pd.DataFrame(slowd,index = bars.index,columns=['Slowd'])
    upperbanddf = pd.DataFrame(upperband,index = bars.index,columns=['upperband'])
    middlebanddf = pd.DataFrame(middleband,index = bars.index,columns=['middleband'])
    lowerbanddf = pd.DataFrame(lowerband,index = bars.index,columns=['lowerband'])
    obvdf = pd.DataFrame(obvout,index = bars.index,columns=['OBV'])
    mfidf = pd.DataFrame(mfiout,index = bars.index,columns=['MFI'])
    dojidf = pd.DataFrame(dojiout,index = bars.index,columns=['DOJI'])
    # marubozudf = pd.DataFrame(marubozuout,index = bars.index,columns=['MARUBOZU'])
    # hammerdf = pd.DataFrame(hammerout,index = bars.index,columns=['HAMMER'])
    # engulfingdf = pd.DataFrame(engulfingout,index = bars.index,columns=['ENGULFING'])
    obvMA1 = pd.rolling_mean(obvdf, MA1, min_periods=1).fillna(0.0)
    obvMA1np = obvMA1.values

    mfiMA1 = pd.rolling_mean(mfidf, MA1, min_periods=1).fillna(0.0)
    mfiMA1np = mfiMA1.values

    rsinp = rsidf.values
    obvnp = obvdf.values
    # mfinp = mfidf.values
    dojinp = dojidf.values
    # dojistarnp = dojistardf.values
    # marubozunp = marubozudf.values
    # hammernp = hammerdf.values
    # engulfingnp = engulfingdf.values



    CloseLD = bars['Close'] - signals['long_mavg']

    CloseLDPc = CloseLD.pct_change()

    CloseLDPc2 = CloseLD/bars['Close']
    ms['closeLDPc2'] = 0
    ms['closeLDPc2signal'] = np.where(CloseLDPc2<-0.15,1,0)

    cLDPMA1 = pd.rolling_mean(CloseLDPc2, MA1, min_periods=1).fillna(0.0)
    cLDPMA1np = cLDPMA1.values
    volpchange = bars['Volume'].pct_change()       
    volpchangenp = volpchange.values

    closepgain = bars['Close'].pct_change().cumsum()   
    closepgainMA1 = pd.rolling_mean(closepgain, 20, min_periods=1).fillna(0.0)
    closepgainnp  = closepgain.values    
    closepgainMA1np = closepgainMA1.values
    # CLDPc2VolPchange = np.where(CloseLDPc2 >0,CloseLDPc2*volpchange,CloseLDPc2*volpchange)
    # print CLDPc2VolPchange[0:10],volpchange[0:10],CloseLDPc2[0:10]
    # print ms[ms.closeLDPc2signal == 1]

    '''
    back algo testing
    '''
    '''
    calculate the buy,sell signals
    '''
    buysig = 0
    sellsig = 0
    holdsig = 0
    stance = 'none'
    closep = bars['Close'].values
    buyprice = 0
    sellprice = 0
    buycloseLD = 0
    sellcloseLD = 0
    buydate = 0
    selldate = 0
    buyprevPoscnt = 0
    negcnt = 0
    neggain = 0
    buygain = 0
    candidate =0
    # print type(bars['Close']),type(closep)
    # print type(ms['closeLDPc2signal'].values)

    closeLDPnp = CloseLDPc2.values

    newsigarr = np.zeros(len(bars['Close']))

    mixedSig = closepgain
    mixedSigMA1 = pd.rolling_mean(mixedSig, MA1, min_periods=1).fillna(0.0)
    mixedSigMA2 = pd.rolling_mean(mixedSig, MA2, min_periods=1).fillna(0.0)
    mixedSigMA1np = mixedSigMA1.values
    mixedSigMA2np = mixedSigMA2.values

    ''' algo 3'''
    signalnp = mixedSigMA1np
    dayslim = 5
    barsdf = bars
    mmsigdf,mmsignp,maxsigdf,minsigdf,maxsignp,minsignp,maxqueue,minqueue = inflectionPoint(signalnp,dayslim,barsdf)

    signalnp = mixedSigMA2np
    dayslim = 5#30
    barsdf = bars
    mmsigdf2,mmsignp2,maxsigdf2,minsigdf2,maxsignp2,minsignp2,maxqueue2,minqueue2 = inflectionPoint(signalnp,dayslim,barsdf)

    volumenp = bars['Volume'].values
    signalnp = volumenp
    dayslim = 5
    barsdf = bars
    mmsigdf3,mmsignp3,maxsigdf3,minsigdf3,maxsignp3,minsignp3,maxqueue3,minqueue3 = inflectionPoint(signalnp,dayslim,barsdf)


    signalnp = bars['Close'].values
    dayslim = 5
    barsdf = bars
    mmsigdf4,mmsignp4,maxsigdf4,minsigdf4,maxsignp4,minsignp4,maxqueue4,minqueue4 = inflectionPoint(signalnp,dayslim,barsdf)

    signalnp = closeLDPnp
    dayslim = 5
    barsdf = bars
    mmsigdf5,mmsignp5,maxsigdf5,minsigdf5,maxsignp5,minsignp5,maxqueue5,minqueue5 = inflectionPoint(signalnp,dayslim,barsdf)


    volumeday = bars['Volume'].values
    openday = bars['Open'].values
    closeday = bars['Close'].values
    highday = bars['High'].values
    lowday = bars['Low'].values
    closeopendiff = (bars['Close']-bars['Open']).values
    closeopendiff_pch = ((bars['Close']-bars['Open'])/bars['Close']).values
    dfvolumeMA1 = pd.rolling_mean(bars['Volume'],5,min_periods=1).fillna(0.0)
    npvolumeMA1 = dfvolumeMA1.values
    # print len(npvolumeMA1),len(volumeday),len(dfvolumeMA1)
    # print 'time,5day Av, 3day Av, gain,price,price inflection point,price 30MA inflection,volume,volume 5day,mfi,obv,close open diff'

    # for day in range(len(bars['Close'])):
    #     if CloseLD.index[day] > dt.datetime(2014,1,1) and CloseLD.index[day] < dt.datetime(2014,6,30):    
    #         print CloseLD.index[day],',', mixedSigMA1np[day],',',closepgain[day],',',closep[day],',',mmsignp4[day],',',mmsignp[day],',',mmsignp3[day],\
    #             ',',volumeday[day],',',npvolumeMA1[day],',',closeopendiff_pch[day],',',closeLDPnp[day],',',mmsignp5[day]


    '''
    pattern gen and recognition
    '''
    negsigdf,possigdf,patternAr = PatternSave(bars,mmsigdf4,mmsignp4)
    global gpatternAr
    gpatternAr = patternAr
    # patternnum = patternRecAuto(patternAr,bars)
    signalnp = bars['Close'].values
    dayslim = 5
    barsdf = bars
    mmsigdf_ver1,mmsignp_ver1,maxsigdf_ver1,minsigdf_ver1,maxsignp_ver1,minsignp_ver1,maxqueue_ver1,minqueue_ver1 =\
            inflectionPoint_ver1(signalnp,dayslim,barsdf)    

    global goodcasedf
    goodcasedf = mmsigdf_ver1
    global fixcasedf
    fixcasedf = mmsigdf4
    global gbarsdf
    gbarsdf = bars

    '''
    pattern end
    '''

    holdingcount =0
    uppermax = 0
    maxday = 0
    minday = 0
    buymax = 0
    buymin = 0
    totalreturngain = 0
    newsigstr_arr1 = []
    newsigstr_arr2 = []
    newsigstr_arr3 = []
    newsigstr_arr4 = []
    newsigstr_arr5 = []
    for day in range(len(bars['Close'])):

        if day > 10 :

            if stance == 'none':
                
                if mmsignp[day] == 0:
                    if mmsignp4[day] == 1 or (mmsignp4[day] == 0 and mmsignp5[day] == 1):
                        
                        patternnum = patternRecAuto(patternAr,bars[day-9:day+1])
                        if patternnum > 0:
                            print 'pattern found:',patternnum
                        if patternnum == -1:
                            print 'pattern not found:',patternnum

                            currentbuymax = mixedSigMA2np[day]
                            if buymax == 0  or (not buymax == 0 and not buymax == currentbuymax ):
                                stance = 'holdingA'
                                buyprice = closep[day]                        
                                buydate = day
                                buymax = mixedSigMA2np[day]
                                newsigarr[day] = 1
                                print 'buyprice1:',stance,buyprice, CloseLD.index[day],'day:',day
                                newsigstr_arr1.append(CloseLD.index[day])
                                newsigstr_arr2.append(stance)
                                newsigstr_arr3.append(buyprice)
                                newsigstr_arr4.append('buyprice1')
                                newsigstr_arr5.append(totalreturngain)

                elif mmsignp[day] == 1 :
                    
                    if mmsignp4[day] == 1 or (mmsignp4[day] == 0 and mmsignp5[day] == 1):
                        patternnum = patternRecAuto(patternAr,bars[day-9:day+1])
                        if patternnum > 0:
                            print 'pattern found:',patternnum
                        if patternnum == -1:
                            print 'pattern not found:',patternnum                        
                            currentbuymin = mixedSigMA2np[day]
                            if buymin == 0 or (not buymin == 0 and not buymin == currentbuymin ):
                                stance = 'holdingB'
                                buyprice = closep[day]                        
                                buydate = day
                                buymin = mixedSigMA2np[day]
                                newsigarr[day] = 1
                                print 'buyprice2:',stance,buyprice, CloseLD.index[day],'day:',day
                                newsigstr_arr1.append(CloseLD.index[day])
                                newsigstr_arr2.append(stance)
                                newsigstr_arr3.append(buyprice)
                                newsigstr_arr4.append('buyprice2')
                                newsigstr_arr5.append(totalreturngain)

            elif stance == 'holdingA':
                pchange = (closep[day] - buyprice)/buyprice

                if mmsignp4[day] == 0  or closeopendiff_pch[day] <0:
                # if closeopendiff_pch[day] <0:                    
                    newsigarr[day] = 0
                    stance = 'none'
                    sellprice = closep[day]
                    totalreturngain = totalreturngain+pchange
                    print 'sellprice1:',stance,sellprice,CloseLD.index[day],'currentgain:',pchange,'totalgain:',totalreturngain
                    newsigstr_arr1.append(CloseLD.index[day])
                    newsigstr_arr2.append(stance)
                    newsigstr_arr3.append(sellprice)
                    newsigstr_arr4.append('sellprice1')
                    newsigstr_arr5.append(totalreturngain)
                else:
                    newsigarr[day] = 1
            elif stance == 'holdingB':
                pchange = (closep[day] - buyprice)/buyprice

                if mmsignp4[day] == 0 or closeopendiff_pch[day] <0:
                # if closeopendiff_pch[day] <0:    
                    newsigarr[day] = 0
                    stance = 'none'
                    sellprice = closep[day]
                    totalreturngain = totalreturngain+pchange
                    print 'sellprice2:',stance,sellprice,CloseLD.index[day],'currentgain:',pchange,'totalgain:',totalreturngain
                    newsigstr_arr1.append(CloseLD.index[day])
                    newsigstr_arr2.append(stance)
                    newsigstr_arr3.append(sellprice)
                    newsigstr_arr4.append('sellprice2')
                    newsigstr_arr5.append(totalreturngain)
                else:
                    newsigarr[day] = 1


    newsigdf = pd.DataFrame(newsigarr,index = bars.index,columns=['Signals']).fillna(0.0)
    newsigdf.index.name = 'Date'

    ms['newsignals'] = 0

    ms['newsignals']=newsigdf['Signals']


    initial_capital = closep[0] *110 
    portfolio2 = MarketOnMixedPortfolio(
        'Hyundai', ms, initial_capital=initial_capital)
     
    returns2 = portfolio2.backtest_portfolio()
    print 'start closep:',closep[0],'current closep:',closep[day]
    print 'totalGain:', (returns2.total[returns2.total.index[-1]] - initial_capital)/initial_capital,'initial_capital:',initial_capital,'totalReturn:',returns2.total[returns2.total.index[-1]]
    print 'benchmark:',closepgainnp[-1]

    #''' plot draw
    ax5 = fig.add_subplot(514,  ylabel='Return')
    returns2['total'].plot(ax=ax5, color='r', lw=2.)
    ax6 = fig.add_subplot(515,  ylabel='Benchmark')
    closepgain.plot(ax=ax6, color='r', lw=2.)

    plt.show()
    fig.savefig("./data/"+name+".png", dpi = 100)
    #''' 
    # fig.savefig("./data/"+name+".png", dpi = 400)

    # print newsigstr_arr1,newsigstr_arr2,newsigstr_arr3,newsigstr_arr4,type(newsigstr_arr1)
    newsigstr_df = pd.DataFrame({'Date':newsigstr_arr1,'Stance':newsigstr_arr2,'Price':newsigstr_arr3,'BuyorSell':newsigstr_arr4,
        'totalGain':newsigstr_arr5},index=newsigstr_arr1)
    # print newsigstr_df.tail()
    import sqlite3
    import pandas.io.sql as pd_sql
    if typearg == 1:
        con = sqlite3.connect("./data/result_db.sqlite")
        tablename = 'result_'+codearg+'_'+namearg
        con.execute("DROP TABLE IF EXISTS "+tablename)
        # newsigstr_df.to_sql(tablename, con,False)
        pd_sql.write_frame(newsigstr_df, tablename, con)

        newsigsum_df = pd.DataFrame({'Benchmark':[closepgainnp[-1]],'Start':[closep[0]],'Current':[closep[-1]]
            ,'totalGain':[(returns2.total[returns2.total.index[-1]] - initial_capital)/initial_capital],'tradingGain':[totalreturngain]})
        # newsigsum_df = pd.DataFrame({'Benchmark':[closepgainnp[-1]]})
        
        tablename = 'result_'+codearg+'_'+namearg+'summary'
        con.execute("DROP TABLE IF EXISTS "+tablename)
        pd.io.sql.write_frame(newsigsum_df, tablename, con)
      

        newsigbench_df = pd.DataFrame({'Date':bars.index,'Benchmark':closepgain,'totalReturn':returns2['total'].pct_change().cumsum()},index= bars.index)
        tablename = 'result_'+codearg+'_'+namearg+'benchmark'
        con.execute("DROP TABLE IF EXISTS "+tablename)
        pd.io.sql.write_frame(newsigbench_df, tablename, con)
    elif typearg == 2:
        con = sqlite3.connect("./data/result_kosdaq_db.sqlite")
        tablename = 'result_'+codearg+'_'+namearg
        con.execute("DROP TABLE IF EXISTS "+tablename)
        # newsigstr_df.to_sql(tablename, con,False)
        pd_sql.write_frame(newsigstr_df, tablename, con)

        newsigsum_df = pd.DataFrame({'Benchmark':[closepgainnp[-1]],'Start':[closep[0]],'Current':[closep[-1]]
            ,'totalGain':[(returns2.total[returns2.total.index[-1]] - initial_capital)/initial_capital],'tradingGain':[totalreturngain]})
        # newsigsum_df = pd.DataFrame({'Benchmark':[closepgainnp[-1]]})
        
        tablename = 'result_'+codearg+'_'+namearg+'summary'
        con.execute("DROP TABLE IF EXISTS "+tablename)
        pd.io.sql.write_frame(newsigsum_df, tablename, con)
      

        newsigbench_df = pd.DataFrame({'Date':bars.index,'Benchmark':closepgain,'totalReturn':returns2['total'].pct_change().cumsum()},index= bars.index)
        tablename = 'result_'+codearg+'_'+namearg+'benchmark'
        con.execute("DROP TABLE IF EXISTS "+tablename)
        pd.io.sql.write_frame(newsigbench_df, tablename, con)
    elif typearg == 3:
        con = sqlite3.connect("./data/result_kospi200_db.sqlite")
        tablename = 'result_'+codearg+'_'+namearg
        con.execute("DROP TABLE IF EXISTS "+tablename)
        # newsigstr_df.to_sql(tablename, con,False)
        pd_sql.write_frame(newsigstr_df, tablename, con)

        newsigsum_df = pd.DataFrame({'Benchmark':[closepgainnp[-1]],'Start':[closep[0]],'Current':[closep[-1]]
            ,'totalGain':[(returns2.total[returns2.total.index[-1]] - initial_capital)/initial_capital],'tradingGain':[totalreturngain]})
        # newsigsum_df = pd.DataFrame({'Benchmark':[closepgainnp[-1]]})
        
        tablename = 'result_'+codearg+'_'+namearg+'summary'
        con.execute("DROP TABLE IF EXISTS "+tablename)
        pd.io.sql.write_frame(newsigsum_df, tablename, con)
      

        newsigbench_df = pd.DataFrame({'Date':bars.index,'Benchmark':closepgain,'totalReturn':returns2['total'].pct_change().cumsum()},index= bars.index)
        tablename = 'result_'+codearg+'_'+namearg+'benchmark'
        con.execute("DROP TABLE IF EXISTS "+tablename)
        pd.io.sql.write_frame(newsigbench_df, tablename, con)        
    # imageFile = open("./data/"+name+".png", 'rb')
    # b = sqlite3.Binary(imageFile.read())
    # con.execute("INSERT INTO "+tablename+" (image) values(?)", (b,))
    # imageFile.close()
    # from xlutils.copy import copy 
    # from xlrd import open_workbook 
    # from xlwt import easyxf 

    # if typearg == 1:
    #     rb = open_workbook('result_summary.xls',formatting_info=True)
    # elif typearg == 2:        
    #     rb = open_workbook('result_summary_kosdaq.xls',formatting_info=True)
    # r_sheet = rb.sheet_by_index(0) 
    # wb = copy(rb) 
    # w_sheet = wb.get_sheet(0) 
    # newrow =  r_sheet.nrows
    # w_sheet.write(newrow, 0, 'start closep')
    # w_sheet.write(newrow, 1, closep[0])
    # w_sheet.write(newrow, 2, 'currnet closep')
    # w_sheet.write(newrow, 3, closep[-1])
    # w_sheet.write(newrow, 4, 'total trading gain')
    # w_sheet.write(newrow, 5, totalreturngain)
    # w_sheet.write(newrow, 6, 'totalGain')
    # w_sheet.write(newrow, 7, (returns2.total[returns2.total.index[-1]] - initial_capital)/initial_capital)
    # w_sheet.write(newrow, 8, 'initial_capital')
    # w_sheet.write(newrow, 9, initial_capital)
    # w_sheet.write(newrow, 10, 'totalReturn')
    # w_sheet.write(newrow, 11, returns2.total[returns2.total.index[-1]])
    # w_sheet.write(newrow, 12, 'benchmark')
    # w_sheet.write(newrow, 13, closepgainnp[-1])
    # if typearg == 1:
    #     wb.save('result_summary.xls')    
    # elif typearg == 2:
    #     wb.save('result_summary_kosdaq.xls')                

