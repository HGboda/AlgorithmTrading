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
# from guiqwt.plot import CurveDialog
# from guiqwt.builder import make


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




'''
pattern functions
'''



'''
check if targetCol2 pattern has a down trend pattern or not with downtrend pattern data 
'''
global targetnp
targetnp = 0
def patternAllRun(targetdf,targetnp,barsdf,gpatternAr):
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
    startdate = '2014-01-01'
    # enddate = '2008-12-30'
    print symbol
    
    '''
    pattern read
    '''
    
    dbname = 'pattern_db_'+codearg+'_'+namearg+'.sqlite'
    con = sqlite3.connect("../data/pattern/"+dbname)

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
    con2 = sqlite3.connect("../data/pattern/"+dbname)
    tablename = 'result_'+codearg+'_'+namearg
    extractdf = pd_sql.read_frame("SELECT * from "+tablename, con2)
    extractids = extractdf['ExtractId'].values

    # print 'read pattern:'
    # print readlist[0].patterndf
    print 'extractids:',extractids,len(extractids)
    
    con.close()        
    con2.close()        
    print 'extractid save done'
    return readlist,extractids

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

global gbarsdf
gbarsdf = 0
global gpatternAr
gpatternAr = 0
global gextractid
gextractid = 0
def RunSimul(codearg,typearg,namearg,mode,dbmode,histmode,srcsite):

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
    # symbol = 'GOOG/INDEXKRX_KOSPI200'
    startdate = '2011-01-01'
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
        # print bars.tail()
    else:
        bars = bars_org

    bars = bars.sort_index()
    print '----------final sorted bars---------'
    print bars.tail()   

    signalnp = bars['Close'].values
    dayslim = 5
    barsdf = bars
    mmsigdf4,mmsignp4,maxsigdf4,minsigdf4,maxsignp4,minsignp4,maxqueue4,minqueue4 = inflectionPoint(signalnp,dayslim,barsdf)



    '''
    pattern gen and recognition
    '''
    negsigdf,possigdf,patternAr = PatternSave(bars,mmsigdf4,mmsignp4)
    global gpatternAr
    gpatternAr = patternAr

    allselectpattern = patternAllRun(mmsigdf4,mmsignp4,bars,patternAr)

    foundnumlist = patternCompareAndExtract(allselectpattern)

    extractid = patternExtractCandidates(foundnumlist,allselectpattern,patternAr)

    print 'extractid done'
    global targetnp
    targetnp = mmsignp4
    global gbarsdf
    gbarsdf = bars
    global gextractid
    gextractid = extractid

    '''
    pattern end
    '''
    basepos = u"../data/pattern/up/"
    
    # deletename = basepos+u'pattern_db_'+codearg+u'_'+namearg+u'.sqlite'
    # if os.path.isfile(deletename):
    #     os.remove(deletename)
    # deletename = basepos+u'extractid_db_'+codearg+'_'+namearg+u'.sqlite'
    # if os.path.isfile(deletename):
    #     os.remove(deletename)
    # print 'deletename',deletename

    print 'save starts !!'
    dbname = 'pattern_db_'+codearg+'_'+namearg+'.sqlite'
    con2 = sqlite3.connect("../data/pattern/up/"+dbname)
    dblen = len(patternAr)
    tablename_base = 'result_'+codearg+'_'+namearg

    for cnt in range(dblen):
        tablename = tablename_base+'_'+str(cnt)
        # print 'writetable:',tablename
        con2.execute("DROP TABLE IF EXISTS "+tablename)
        patternAr[cnt].patterndf = patternAr[cnt].patterndf.reset_index()
        pd_sql.write_frame(patternAr[cnt].patterndf, tablename, con2)

    
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
    
    dbname = 'extractid_db_'+codearg+'_'+namearg+'.sqlite'
    con3 = sqlite3.connect("../data/pattern/up/"+dbname)
    tablename = 'result_'+codearg+'_'+namearg
    con3.execute("DROP TABLE IF EXISTS "+tablename)
    print 'extractid tablename:',tablename
    listdf = pd.DataFrame({'ExtractId':extractid})
    # print listdf.tail()
    # print listdf['ExtractId'].values
    pd_sql.write_frame(listdf, tablename, con3)
    
    # con.close()        
    con2.close()        
    con3.close()      
    print 'extractid save done'


'''
pattern auto saving
'''    
import xlrd

book = xlrd.open_workbook("../symbols.xls")
sheet = book.sheet_by_name('kospi')
# sheet = book.sheet_by_name('kosdaq')
length = sheet.nrows

# basepos = u"../data/pattern/"
# filelist = [ f for f in os.listdir(basepos) if f.endswith(u".sqlite") ]
# for f in filelist:
#     deletename = basepos+f
#     if os.path.isfile(deletename):
#         os.remove(deletename)


for rowcnt in range(30,100):
    code = '{0:06d}'.format(int(sheet.row_values(rowcnt)[1]))
    name = sheet.row_values(rowcnt)[2]
    try:
        # RunSimul(str(code),1,name,'one')
        RunSimul(str(code),1,name,'realtime','none','none',1)
    except:
        pass            