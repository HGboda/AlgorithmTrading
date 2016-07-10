%matplotlib inline
# %pylab inline 
# from matplotlib import interactive
# interactive(False)

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

'''test '''
global gmaxsigdf
gmaxsigdf = 0
global gmaxsignp
gmaxsignp = 0
global gminsignp
gminsignp = 0
global gminsigdf
gminsigdf = 0
global gbars
gbars = 0
global gmaxqueue
gmaxqueue = 0
global gminqueue
gminqueue = 0
global gmmstancedf
gmmstancedf =0
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
                            print 'maxday:',barsdf.index[maxday],'signalnp[maxday]:',signalnp[maxday],'day:',barsdf.index[day]
                            break
                        if (signalnp[searchday] >= signalnp[searchday-1] < signalnp[searchday-2])\
                            or (signalnp[searchday] > signalnp[searchday-1] <= signalnp[searchday-2]):
                            minday = searchday -1
                            minqueue.addRear(minday)
                            minqueue.rear += 1
                            nextsearch = 'max'
                            print 'minday:',barsdf.index[minday],'signalnp[minday]:',signalnp[minday],'day:',barsdf.index[day]
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
                            print 'minday:',barsdf.index[minday],'signalnp[minday]:',signalnp[minday],'day:',barsdf.index[day]
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
                            print 'maxday:',barsdf.index[maxday],'signalnp[maxday]:',signalnp[maxday],'day:',barsdf.index[day]
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
    
    ''' test'''
    global gmaxsigdf
    gmaxsigdf = maxsigdf
    global gmaxsignp
    gmaxsignp = maxsignp
    global gminsignp
    gminsignp = minsignp
    global gminsigdf
    gminsigdf = minsigdf
    global gbars
    gbars = barsdf
    global gmaxqueue
    gmaxqueue = maxqueue
    global gminqueue
    gminqueue = minqueue

    mmsignp = np.zeros(len(signalnp))
    mmstance = 'none'
    
    mmStancequeue =Deque()
    mmStancequeue.front = 0
    mmStancequeue.rear = 0

    mmGainqueue =Deque()
    mmGainqueue.front = 0
    mmGainqueue.rear = 0

    mmMaxmeanqueue =Deque()
    mmMaxmeanqueue.front = 0
    mmMaxmeanqueue.rear = 0

    mmMinmeanqueue =Deque()
    mmMinmeanqueue.front = 0
    mmMinmeanqueue.rear = 0

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
            if len(minmaxdf[minmaxdf['MAXSignals']==1][0:minmaxdf.index[day]]) > 1:
                dfidx = minmaxdf[minmaxdf['MAXSignals']==1][0:minmaxdf.index[day]].index
                maxmean = signaldf2['Value'][dfidx].mean()
                # print 'dfidx',dfidx,maxmean
                # print 'signaldf2,',signaldf2['Value'][dfidx]
            else:
                maxmean = 0    
            if len(minmaxdf[minmaxdf['MINSignals']==1][0:minmaxdf.index[day]]) > 1:
                dfidx = minmaxdf[minmaxdf['MINSignals']==1][0:minmaxdf.index[day]].index
                minmean = signaldf2['Value'][dfidx].mean()
                # print 'dfidx',dfidx,maxmean
                # print 'signaldf2,',signaldf2['Value'][dfidx]
            else:
                minmean = 0        
            if mmstance == 'none':

                if minsignp[day-1] == 1:

                    curday = day
                    pointnum = 2
                    mincount=0
                    maxcount=0
                    searchtype=2 # 1:max,2:min

                    #pointnum = len(minmaxdf[minmaxdf['MINSignals']==1][0:minmaxdf.index[curday]])

                    # minangle2,minretday0,minretvalue0 = getMinMaxPointAngle(pointnum,searchtype,fromday,curday,mincount,maxcount,minmaxdf,signaldf2,0)
                    # if not minretvalue0 == 0 and len(minretvalue0) > 1:
                        # print 'minangle2:',minangle2,',minretday0:',minretday0,',minretvalue0:',minretvalue0
                        # print retvalue0[0],retvalue0[1]
                    
                    minday0,minvalue0 = getMinMaxPoint(searchtype,0,curday,0,0,minmaxdf,signaldf2,0)
                    minday1,minvalue1 = getMinMaxPoint(searchtype,0,curday,1,0,minmaxdf,signaldf2,0)
                    searchtype=1 # 1:max,2:min                    
                    maxday0,maxvalue0 = getMinMaxPoint(searchtype,0,curday,0,0,minmaxdf,signaldf2,0)
                    maxday1,maxvalue1 = getMinMaxPoint(searchtype,0,curday,0,1,minmaxdf,signaldf2,0)
                    caly = 0

                    
                    # if barsdf.index[day] > dt.datetime(2014,4,10) \
                    #     and barsdf.index[day] < dt.datetime(2014,4,30):
                    #     print 'maxday0:',barsdf.index[maxday0],',minday1:',barsdf.index[minday1],barsdf.index[day]
                    if not minday0 == -1 and not minday1 == -1 and not maxday0 == -1 and not maxday1 == -1\
                        and maxday0 - minday1 > 1:# and maxmean < signaldf2['Value'][signaldf2.index[curday]]:

                        pchigh = abs((maxvalue0 -  minvalue1)/minvalue1)
                        pclow = abs((minvalue0 - maxvalue0)/maxvalue0)
                        p0 = [minvalue0 ,maxvalue0,minvalue1]
                        p1 = sm.add_constant([minday0 , maxday0,minday1], prepend=False)
                        slope, intercept = sm.OLS(p0, p1).fit().params
                        # print [minday0 , maxday0, minday1],[minvalue0 ,maxvalue0,minvalue1],slope,intercept
                        caly = curday*slope+intercept
                        
                        # print 'mmstance buy today: ',barsdf.index[day],'value:',signaldf2['Value'][signaldf2.index[curday]] ,\
                        # ',caly:',caly,',pchigh:',pchigh,',pclow:',pclow,\
                        # 'benchgain:',benchgainnp[day]

                        

                        # if not pchigh == 0:# and pchigh>pclow :#and pclow/pchigh < 0.6:#signaldf2['Value'][signaldf2.index[curday]] < caly :

                        mmsignp[day] = 1
                        mmstance = 'holding'
                        mmStancequeue.addRear('min')#today status
                        mmStancequeue.rear += 1
                        buyprice = signalnp[day]
                        curGain = 0
                        mmGainqueue.addRear(curGain)    
                        mmGainqueue.rear += 1
                        mmMaxmeanqueue.addRear(maxmean)    
                        mmMaxmeanqueue.rear += 1
                        mmMinmeanqueue.addRear(minmean)    
                        mmMinmeanqueue.rear += 1
                        print 'mmstance buy today: ',barsdf.index[day],', minangle2:',minangle2,\
                        ',maxangle2:',maxangle2,', price:',barsdf['Close'][barsdf.index[day]],'maxmean:',maxmean,',minmean:',minmean
                    else:
                        
                        mmsignp[day] = 0
                        mmStancequeue.addRear('none')#today status
                        mmStancequeue.rear += 1  
                        curGain = 0
                        mmGainqueue.addRear(curGain)    
                        mmGainqueue.rear += 1  
                        mmMaxmeanqueue.addRear(maxmean)    
                        mmMaxmeanqueue.rear += 1
                        mmMinmeanqueue.addRear(minmean)    
                        mmMinmeanqueue.rear += 1
                        # print 'mmstance skip today: ',barsdf.index[day],', minangle2:',minangle2,\
                        # ',maxangle2:',maxangle2,', price:',barsdf['Close'][barsdf.index[day]],', curangle:',curangle
                else:
                    mmsignp[day] = 0
                    mmStancequeue.addRear('none')#today status
                    mmStancequeue.rear += 1
                    
                    curGain = 0
                    mmGainqueue.addRear(curGain)    
                    mmGainqueue.rear += 1 
                    mmMaxmeanqueue.addRear(maxmean)    
                    mmMaxmeanqueue.rear += 1
                    mmMinmeanqueue.addRear(minmean)    
                    mmMinmeanqueue.rear += 1
            elif mmstance == 'holding':                
        
                curday = day
                # curdx = (curday - buyday0)
                # curdy = (barsdf['Close'][barsdf.index[day]] - buyvalue0)
                # buyangle = np.arctan(curdy/curdx)*57.3            

                
                # if curGain < 0:
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
                    mmMaxmeanqueue.addRear(maxmean)    
                    mmMaxmeanqueue.rear += 1
                    mmMinmeanqueue.addRear(minmean)    
                    mmMinmeanqueue.rear += 1
                else:
                    mmsignp[day] = 1
                    mmStancequeue.addRear('minholding')#today status
                    mmStancequeue.rear += 1
                    curGain = 0
                    mmGainqueue.addRear(curGain)    
                    mmGainqueue.rear += 1 
                    mmMaxmeanqueue.addRear(maxmean)    
                    mmMaxmeanqueue.rear += 1
                    mmMinmeanqueue.addRear(minmean)    
                    mmMinmeanqueue.rear += 1                    
                        # print 'mmstance holding today: ',barsdf.index[day]\
                        #     ,', price:',barsdf['Close'][barsdf.index[day]],', minangle2:',minangle2,', buyangle:',buyangle
        else:
            mmsignp[day] = 0
            mmStancequeue.addRear('none')#today status
            mmStancequeue.rear += 1

            curGain = 0
            mmGainqueue.addRear(curGain)    
            mmGainqueue.rear += 1 
            mmMaxmeanqueue.addRear(0)    
            mmMaxmeanqueue.rear += 1
            mmMinmeanqueue.addRear(0)    
            mmMinmeanqueue.rear += 1

    print 'mmStancequeue.size()',mmStancequeue.size(),len(signalnp)

    mmstancenp = np.zeros(len(signalnp))
    idx = mmStancequeue.rear
    for stancecnt in range(len(signalnp)):
        idx = mmStancequeue.size()-stancecnt-1
        # print 'mmstancenp day:',barsdf.index[stancecnt],', stance:',mmStancequeue.getValue(idx)
        if mmStancequeue.getValue(idx) == 'min':
            mmstancenp[stancecnt] = 1

    mmstancedf = pd.DataFrame(mmstancenp,index = barsdf.index,columns=['Stance']).fillna(0.0)
    mmstancedf.index.name = 'Date'    
    # print mmstancedf[0:30]
    global gmmstancedf
    gmmstancedf =mmstancedf
    
    idx = mmStancequeue.rear
    for day in range(len(signalnp)):
        idx = mmStancequeue.size()-day-1
        print barsdf.index[day],ipclosep[day],'mmsignp signal:',mmsignp[day],'minsignp:',minsignp[day],'maxsignp:',maxsignp[day],\
        mmStancequeue.getValue(idx),'gain:',mmGainqueue.getValue(idx),'totalGain:',totalGain,'benchgain:',benchgainnp[day]
        # print ',maxmean:',mmMaxmeanqueue.getValue(idx),',minmean:',mmMinmeanqueue.getValue(idx)



    mmsigdf = pd.DataFrame(mmsignp,index = barsdf.index,columns=['MMSignals']).fillna(0.0)
    mmsigdf.index.name = 'Date'    

    print 'inflectionPoint end'
    return mmsigdf,mmsignp,maxsigdf,minsigdf,maxsignp,minsignp,maxqueue,minqueue

def RunSimul(codearg,typearg,namearg):

    
    code = codearg #'097950'#'005930' #'005380'#009540 #036570
    if typearg == 1:
        symbol = 'GOOG/KRX_'+code
    elif typearg == 2:
        symbol = 'GOOG/KOSDAQ_'+code
    elif typearg == 3:
        symbol = 'GOOG/INDEXKRX_KOSPI200'  
    # symbol = 'GOOG/INDEXKRX_KOSPI200'
    startdate = '2013-01-01'
        
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

    proxy_support = urllib2.ProxyHandler({"https":"https://50.58.251.66:3128"})
    opener = urllib2.build_opener(proxy_support)
    urllib2.install_opener(opener)

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
    startdate = datetime(2013, 01, 01)#'2007-10-01'
    # enddate = datetime(2008, 12, 30)#'2009-12-30'


    '''
    calc polyfit and plot cross point 
    '''

    npbars = Quandl.get(symbol, collapse='Daily', trim_start=startdate, trim_end=datetime.today(),returns='numpy',authtoken="")
    # npbars = Quandl.get(symbol, collapse='Daily', trim_start=startdate, trim_end=enddate,returns='numpy',authtoken="")


    new_values=[]
    for row in npbars:
        date, open1,high,low,closep,volume = row
        new_values.append([date2num(datetime.strptime(date.replace('-',' '),'%Y %m %d'))
                           ,open1
                           ,high
                           ,low
                           ,closep
                           ,volume])

    numArray = np.array(list(new_values))

    date, open1,high,low,closep,volume = numArray[0:,0],numArray[0:,1],numArray[0:,2],numArray[0:,3],numArray[0:,4],numArray[0:,5]


    if date_append == True:
        closep = np.append(closep,rtclose)
        open1 = np.append(open1,rtopen)
        high = np.append(high,rthigh)
        low = np.append(low,rtlow)
        volume = np.append(volume,rtvolume)
        date = np.append(date,date2num(datetime.strptime(rtdate.replace('/',' '),'%m %d %Y')))
        # print 'append:',date,date2num(datetime.strptime(rtdate.replace('/',' '),'%m %d %Y'))

    MA1 =5
    MA2 =30

    '''
    draw matplotlib chart
    '''
    #''' plot disable
    fig = plt.figure(figsize=(20, 20))

    fig.patch.set_facecolor('white')     # Set the outer colour to white
    ax1 = fig.add_subplot(311,  ylabel='Price in $')

    # # Plot the AAPL closing price overlaid with the moving averages
    bars['Close'].plot(ax=ax1, color='black', lw=2.)
    signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)

    # # Plot the "buy" trades against AAPL
    # ax1.plot(signals.ix[signals.positions == 1.0].index,
    #           signals.short_mavg[signals.positions == 1.0],
    #           '^', markersize=10, color='m')

    # # # Plot the "sell" trades against AAPL
    # ax1.plot(signals.ix[signals.positions == -1.0].index,
    #           signals.short_mavg[signals.positions == -1.0],
    #           'v', markersize=10, color='k')


    from matplotlib.finance import candlestick
    from itertools import izip
    ax3 = fig.add_subplot(312,ylabel='Candlestick')
    ax3dates = bars.index.to_pydatetime() 
    ax3times = date2num(ax3dates)
    quotes = izip(ax3times,bars['Open'],bars['Close'],bars['High'],bars['Low'])
    candlestick(ax3,quotes,width=1.5, colorup='g', colordown='r', alpha=1.0)
    ax3.xaxis.set_major_locator(mticker.MaxNLocator(10))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax3.set_xlim(ax3times[0], ax3times[-1])

    ax4 = fig.add_subplot(313,  ylabel='Volume')
    ax4.plot(date,volume,'#EF15C3',label='VolAV1',linewidth=1.5)
    ax4.xaxis.set_major_locator(mticker.MaxNLocator(10))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax4.set_xlim(date[0], date[-1])
    #'''#plot disable
    # datestr1 = datetime.fromtimestamp(time.mktime(mdates.num2date(date[0]).timetuple())).strftime('%Y %m %d')
    # print datestr1,len(bars['Close']),len(date),len(volume)
    # print bars.head()
    # plt.show()


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

    
    closeopendiff_pch_df = ((bars['Close']-bars['Open'])/bars['Close'])
    closeopendiff_pch_MA1_df = pd.rolling_mean(closeopendiff_pch_df, MA1, min_periods=1).fillna(0.0)

    ''' algo 3'''

    print '----inflectionPoint 4-------'
    signalnp = bars['Close'].values
    dayslim = 5
    barsdf = bars
    mmsigdf4,mmsignp4,maxsigdf4,minsigdf4,maxsignp4,minsignp4,maxqueue4,minqueue4 = inflectionPoint(signalnp,dayslim,barsdf)

    global gmmstancedf

    ax1.plot(gmmstancedf.ix[gmmstancedf.Stance == 1.0].index,
              bars['Close'][gmmstancedf.ix[gmmstancedf.Stance == 1.0].index],
              '.',markersize=10, color='r')    
    print 'gmmstancedf size:',len(gmmstancedf.ix[gmmstancedf.Stance == 1.0].index)

    gmmstancedf =0    
    
    # signalnp = closeLDPnp
    # dayslim = 5
    # barsdf = bars
    # mmsigdf5,mmsignp5,maxsigdf5,minsigdf5,maxsignp5,minsignp5,maxqueue5,minqueue5 = inflectionPoint(signalnp,dayslim,barsdf)
    

    # ax1.plot(gmmstancedf.ix[gmmstancedf.Stance == 1.0].index,
    #           bars['Close'][gmmstancedf.ix[gmmstancedf.Stance == 1.0].index],
    #           '*',markersize=10, color='r')    
    # print 'gmmstancedf size:',len(gmmstancedf.ix[gmmstancedf.Stance == 1.0].index)

    # gmmstancedf =0

    
    # signalnp = mixedSigMA1np
    # dayslim = 5
    # barsdf = bars
    # mmsigdf,mmsignp,maxsigdf,minsigdf,maxsignp,minsignp,maxqueue,minqueue = inflectionPoint(signalnp,dayslim,barsdf)

    # ax1.plot(gmmstancedf.ix[gmmstancedf.Stance == 1.0].index,
    #           bars['Close'][gmmstancedf.ix[gmmstancedf.Stance == 1.0].index],
    #           's',markersize=10, color='r')    
    # print 'gmmstancedf size:',len(gmmstancedf.ix[gmmstancedf.Stance == 1.0].index)    
    # plt.show()

    # for day in range(len(bars['Close'])):
    #     if CloseLD.index[day] > dt.datetime(2013,1,1) and CloseLD.index[day] < dt.datetime(2014,6,30):    
    #         print CloseLD.index[day],',',closep[day],',',mmsignp4[day]

    # ax1.plot(mmsigdf4.ix[mmsigdf4.MMSignals == 1.0].index,
    #           bars['Close'][mmsigdf4.ix[mmsigdf4.MMSignals == 1.0].index],
    #           '.',markersize=10, color='r')

    # print mmsigdf4.ix[mmsigdf4.MMSignals == 1.0].index[0:60]
    # print bars['Close'][mmsigdf4.ix[mmsigdf4.MMSignals == 1.0].index][0:60]
# code = '005930'
# name = '삼성전자'

code = '010140'
name = '삼성중공업'
# code = '005380'
# name = '현대차'
RunSimul(str(code),1,name)    