import pandas.io.data as web
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as wd
import sklearn.hmm as lrn
from IPython.html.widgets import interact, interactive, fixed
from IPython.html import widgets
from IPython.display import clear_output, display, HTML
import time
import datetime as dt
from datetime import datetime,timedelta
import os,sys
import linecache
import scipy.interpolate as sp
from scipy import signal, fftpack
import scipy
from numpy import fft
import scipy.stats as stats
import matplotlib.gridspec as gridspec
from scipy.optimize import fsolve
import warnings
# %matplotlib inline
# pd.set_option('display.width',500)
def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print 'EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj)


def findIntersection(fun1,fun2,x0):
    return fsolve(lambda x : fun1(x) - fun2(x),x0)

def findSolveTest():
    x1 = np.arange(-10,5,1)
    y1 = lambda x : 1*x
    y2 = lambda x : 2.1*x+1
    result = findIntersection(y1,y2,0.0)
    # x = numpy.linspace(-2,2,50)
    # pylab.plot(x1,y1(x1),x1,y2(x1),result,y2(result),'ro')
    print result



def fourierExtrapolation(x, n_predict):
    n = x.size
    n_harm = 10                     # number of harmonics in model
    t = np.arange(0, n)
    p = np.polyfit(t, x, 1)         # find linear trend in x
    x_notrend = x - p[0] * t        # detrended x
    x_freqdom = fft.fft(x_notrend)  # detrended x in frequency domain
    f = fft.fftfreq(n)              # frequencies
    indexes = range(n)
    # sort indexes by frequency, lower -> higher
    indexes.sort(key = lambda i: np.absolute(f[i]))
 
    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)
    for i in indexes[:1 + n_harm * 2]:
        ampli = np.absolute(x_freqdom[i]) / n   # amplitude
        phase = np.angle(x_freqdom[i])          # phase
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    return restored_sig + p[0] * t


def stkHMM(lrndata, n_components):

    model = lrn.GaussianHMM(n_components, covariance_type="tied", n_iter=20)
    model.fit([lrndata])

    hidden_states = model.predict(lrndata)
    return [model, hidden_states]


def findMinSlope(bars,day):
    'finding min low list '
    minlowlist = []
    backupminlowlist = []
    prevminval = 0
    for tmpday in range(len(bars[:day+1])-40,len(bars[:day+1]),10):
        minval = bars['bench'][tmpday:tmpday+10].min()
        # print 'minval',minval
        if prevminval == 0:
            prevminval = minval
            minlowlist.append(minval)
        else:
            if prevminval < minval:
                minlowlist.append(minval)    
                
            else:
                minlowlist.pop(-1)
                minlowlist.append(minval)    
        prevminval = minval        
        backupminlowlist.append(minval)
    # print 'minlowlist',minlowlist,min(minlowlist)
    if len(minlowlist) == 1:
        minlowlist = backupminlowlist
    minlowdaylist = []
    for tmpmin in minlowlist:
        tmpdaycnt = 0
        for val in bars['bench'][day-40:day+1]:
            if val == tmpmin:
                break
            tmpdaycnt += 1
        minlowdaylist.append(tmpdaycnt+day-40)

    fitx_dir = minlowdaylist
    paramfit_dir = np.polyfit(fitx_dir,bars['Close'][minlowdaylist],1)
    fity_dir = np.polyval(paramfit_dir,fitx_dir)

    return paramfit_dir[0]

def findMinBenchSlope(bars,day):
    'finding min low list '
    minlowlist = []
    backupminlowlist = []
    prevminval = 0
    for tmpday in range(len(bars[:day+1])-40,len(bars[:day+1]),10):
        minval = bars['bench'][tmpday:tmpday+10].min()
        # print 'minval',minval
        if prevminval == 0:
            prevminval = minval
            minlowlist.append(minval)
        else:
            if prevminval < minval:
                minlowlist.append(minval)    
                
            else:
                minlowlist.pop(-1)
                minlowlist.append(minval)    
        prevminval = minval        
        backupminlowlist.append(minval)
    # print 'minlowlist',minlowlist,min(minlowlist)
    if len(minlowlist) == 1:
        minlowlist = backupminlowlist
    minlowdaylist = []
    for tmpmin in minlowlist:
        tmpdaycnt = 0
        for val in bars['bench'][day-40:day+1]:
            if val == tmpmin:
                break
            tmpdaycnt += 1
        minlowdaylist.append(tmpdaycnt+day-40)

    fitx_dir = minlowdaylist
    paramfit_bench_dir = np.polyfit(fitx_dir,bars['bench'][minlowdaylist],1)

    return paramfit_bench_dir

def findMaxBenchSlope(bars,day):
    'finding max up list '
    maxuplist = []
    backupmaxuplist = []
    prevmaxval = 0
    for tmpday in range(len(bars[:day+1])-40,len(bars[:day+1]),10):
        maxval = bars['bench'][tmpday:tmpday+10].max()
        # print 'maxval',maxval
        if prevmaxval == 0:
            prevmaxval = maxval
            maxuplist.append(maxval)
        else:
            if prevmaxval > maxval:
                maxuplist.append(maxval)    
                
            else:
                maxuplist.pop(-1)
                maxuplist.append(maxval)    
        prevmaxval = maxval 
        backupmaxuplist.append(maxval)
    if len(maxuplist) == 1:
        maxuplist = backupmaxuplist

    maxupdaylist = []
    for tmpmax in maxuplist:
        tmpdaycnt = 0
        for val in bars['bench'][day-40:day+1]:
            if val == tmpmax:
                break
            tmpdaycnt += 1
        maxupdaylist.append(tmpdaycnt+day-40)

    fitx_max_dir = maxupdaylist
    paramfit_max_bench_dir = np.polyfit(fitx_max_dir,bars['bench'][maxupdaylist],1)

    return paramfit_max_bench_dir


def tradeSimul(extrapolation_Startday,day,bars,tmpminday):
    try:

        global gfilterbars
        tradesignals = []
        tradestatus = 'none'

        g_interFallYoutVal = 0
        g_interFallXinDay = 0
        gDelay_Day = False
        gTradeStatus = False
        for tmpday in range(extrapolation_Startday,day+1,1):
            
            daywindow = tmpday
            
            """  --------------  range found -------------------- """
            rangeFound = False
            for eDate in reversed(barsMAranges['eDate']):

                if eDate<bars.index[tmpday]:
                    selrange = barsMAranges[barsMAranges['eDate'] == eDate]
                    selindex = selrange.index[0]
                    rangeFound = True
                    break

            for nday in range(len(bars)):
                if eDate == bars.index[nday]:
                    lasteDateDay = nday
                    break       
           

            if rangeFound == True:
                               
                scut = selrange['scut'][selindex]
                ecut = selrange['ecut'][selindex]
                esbench = selrange['eBench'][selindex] - selrange['sBench'][selindex]
                # print 'esbench',esbench
               
                if scut == -1 and esbench > 0.0 :
                    subtstate = 'low_rising'
                    barsMAranges['subtstate'][selindex] = subtstate
                    
                if scut == 1 and esbench > 0.0 :
                    subtstate = 'high_rising'
                    barsMAranges['subtstate'][selindex] = subtstate
                    
                if scut == -1 and esbench < 0.0:   
                    subtstate = 'low_falling'
                    barsMAranges['subtstate'][selindex] = subtstate
                    
                if scut == 1 and esbench < 0.0 :   
                    subtstate = 'high_falling'   
                    barsMAranges['subtstate'][selindex] = subtstate   
                        
            """  --------------  range found -------------------- """

            rangegain = bars['Close'][lasteDateDay:tmpday+1].pct_change().cumsum()[-1:].values
            # print 'lasteDateDay',lasteDateDay,'tmpday',tmpday,'rangegain',rangegain,'extrapolation_Startday',extrapolation_Startday
            vertex = lasteDateDay
            """  --------------  coef, coeflowlim-------------------- """        
            disbars = bars[:daywindow+1]
            # print 'vertex',vertex,'tmpday',tmpday
            if vertex > tmpday:
                continue
            fitx12 = np.arange(vertex,tmpday+1,1.0)
            paramfit = np.polyfit(fitx12,bars['bench'][vertex:tmpday+1],1)

            if tmpday <= 30 or vertex <=30:
                tall = np.arange(tmpday-30, tmpday+1, 1.0)      
                t1 = np.arange(tmpday-30, tmpday+10, 1.0)  
            else:
                tall = np.arange(vertex-30, tmpday+1, 1.0)      
                t1 = np.arange(vertex-30, tmpday+10, 1.0)  
            
            

            if paramfit[0] <= 0.0:
                coef = 0.0002
            if 0.0 < paramfit[0] <= 0.003:
                coef = 0.0002
            if 0.003 <= paramfit[0] <= 0.004:    
                coef = 0.0001
            if 0.004 < paramfit[0] :    
                coef = 0.00007
            interpx = [0.0,0.001,0.002,0.003,0.004,0.005,0.006,0.007]
            interpy = [0.0002,0.0002,0.0002,0.00018,0.00017,0.00016,0.00015,0.00014]

            interlowlimpx = [-0.002,-0.001,0.0,0.001,0.002,0.003,0.004,0.005,0.006,0.007]
            interlowlimpy = [0.0003,0.00035,0.0004,0.0005,0.00055,0.0006,0.00065,0.0007,0.00075,0.0008]
            coeflowlim = 0.0

            if paramfit[0] >= 0.007:
                coef = 0.00015
                coeflowlim = 0.0008
            if paramfit[0] <= 0.0:
                coef = 0.0002
                if  -0.002 < paramfit[0] <= 0.0:
                    interlowlimfl = sp.interp1d(interlowlimpx, interlowlimpy,kind='linear',bounds_error=False)
                    coeflowlim = interlowlimfl(paramfit[0])    
                else:
                    coeflowlim = 0.0003
            if 0.0< paramfit[0] < 0.007:
                # interpolation
                interfl = sp.interp1d(interpx, interpy,kind='linear',bounds_error=False)
                coef = interfl(paramfit[0])
                
                interlowlimfl = sp.interp1d(interlowlimpx, interlowlimpy,kind='linear',bounds_error=False)
                coeflowlim = interlowlimfl(paramfit[0])    

            

            y1= coef*(t1-(vertex))**2+disbars['bench'][vertex]
            """  --------------  coef, coeflowlim-------------------- """        

            """  --------------  plot ,right side of vertex over y2 , maxlim_y2 -------------------- """                
            if vertex <= 50:
                t2 = np.arange(0, vertex+1, 1.0)      
                y2 = coef*(t2-(vertex))**2+disbars['bench'][vertex]        
                maxlim_y2_idx = bars['bench'][:vertex+1] > y2
                maxlim_y2 = bars['bench'][:vertex+1][maxlim_y2_idx == True].max()

                
                    
                if np.isnan(maxlim_y2) == True:
                    maxlim_y2 = bars['bench'][:vertex+1].max()
                    maxlim_y2_prev = maxlim_y2
                maxlim_y2_day = 0
                for tmpbench in bars['bench'][:vertex+1]:
                    if tmpbench == maxlim_y2:
                        break
                    maxlim_y2_day+=1    
                t_right = np.arange(vertex, tmpday+1, 1.0)          
                y2 = coef*(t_right-(vertex))**2+disbars['bench'][vertex]        
                y_right_idx = bars['bench'][vertex:tmpday+1] > y2
                
            else:
                t2 = np.arange(vertex-50, vertex+1, 1.0)      
                y2 = coef*(t2-(vertex))**2+disbars['bench'][vertex]        
                maxlim_y2_idx = bars['bench'][vertex-50:vertex+1] > y2
                maxlim_y2 = bars['bench'][vertex-50:vertex+1][maxlim_y2_idx == True].max()


                if len(bars[:vertex-50]) < 100:
                    maxlim_y2_prev = bars['bench'][vertex-100:vertex-50].max()
                else:
                    maxlim_y2_prev = bars['bench'][:vertex-50].max()

                if np.isnan(maxlim_y2) == True:
                    maxlim_y2 = bars['bench'][vertex-50:vertex+1].max()
                
                maxlim_y2_day = 0
                for tmpbench in bars['bench'][:vertex+1]:
                    if tmpbench == maxlim_y2:
                        break
                    maxlim_y2_day+=1

                t_right = np.arange(vertex, tmpday+1, 1.0)          
                y2 = coef*(t_right-(vertex))**2+disbars['bench'][vertex]        
                y_right_idx = bars['bench'][vertex:tmpday+1] > y2
                
                
            """  --------------  plot ,right side of vertex over y2 , maxlim_y2 -------------------- """                
            if tmpday > 40:
                tmpmin = bars['bench'][tmpday-40:tmpday+1].min()
                tmpdaycnt = 0
                for val in bars['bench'][tmpday-40:tmpday+1]:
                    if val == tmpmin:
                        break
                    tmpdaycnt += 1
                tmpminday = tmpdaycnt+tmpday-40    
                minvertexy = coef*(tmpday-(vertex))**2+bars['bench'][vertex]
                
                if minvertexy > bars['bench'][tmpday]:
                    tmpdaycnt = 0.0
                    tmpdaysum = 0.0
                    for inday,tmpval in reversed(zip(np.arange(vertex,tmpday+1,1),bars['bench'][vertex:tmpday+1])):    
                        if tmpval < coef*(inday-(vertex))**2+bars['bench'][vertex]:
                            tmpdaysum += inday    
                            tmpdaycnt += 1
                    tmpdayvertex = tmpdaysum/tmpdaycnt
                    
                    t1_minvertex = np.arange(tmpday-10, tmpday+10, 1.0)  
                    y1_minvertex = coeflowlim*(t1_minvertex-tmpdayvertex)**2+\
                                    bars['bench'][vertex] - (bars['bench'][vertex] - bars['bench'][tmpminday])*0.4

            """  --------------  plot ,right side of vertex over y2 , maxlim_y2 -------------------- """                

            ''' calc paramfit_bench_predict '''
            
            fitx_predict = np.arange(extrapolation_Startday,tmpday+1,1.0)
            paramfit_bench_predict = np.polyfit(fitx_predict,bars['bench'][extrapolation_Startday:tmpday+1],1)
            ''' calc paramfit_bench_predict '''

            ''' calc param_setBenchSlope '''
            setbenchSelectDay = 0
            tmprisingSelectdays = []                                        
            tmpSelectdays = []                            
            for loopday in range(tmpday-40,tmpday+1,1):
            # for loopday in range(extrapolation_Startday ,tmpday+1,1):                
                if bars['benchselect'][loopday] == 0:
                    tmpSelectdays.append(loopday)
                if bars['benchselect'][loopday] == 1:
                    tmprisingSelectdays.append(loopday)
            
            tmpSelectMean = 0
            if len(tmpSelectdays) > 0:
                setbenchSelectDay = int(np.mean(tmpSelectdays))
            else:
                setbenchSelectDay = tmpday
            # print 'setbenchSelectDay',setbenchSelectDay,'tmpday',tmpday,'tmpminday',tmpminday
            setBenchSlopeX = np.arange(setbenchSelectDay,tmpday+1,1.0)
            param_setBenchSlope = np.polyfit(setBenchSlopeX,bars['bench'][setbenchSelectDay:tmpday+1],1)

            ''' calc param_setBenchSlope '''
            
            paramfit_bench_dir = findMinBenchSlope(bars,tmpday)
            paramfit_max_bench_dir = findMaxBenchSlope(bars,tmpday)
            
            y_bench_lowlim_prev = paramfit_bench_dir[0]*(tmpday)+paramfit_bench_dir[1]
            y_bench_lowlim_next = paramfit_bench_dir[0]*(tmpday+10)+paramfit_bench_dir[1]

            yup_bench_lim_prev = paramfit_max_bench_dir[0]*(tmpday)+paramfit_max_bench_dir[1]     
            yup_bench_lim_next = paramfit_max_bench_dir[0]*(tmpday+10)+paramfit_max_bench_dir[1]        

            ' find solve '
            solvey1 = lambda x : paramfit_bench_dir[0]*x+paramfit_bench_dir[1]
            solvey2 = lambda x : paramfit_max_bench_dir[0]*x+paramfit_max_bench_dir[1]
            solveresult = findIntersection(solvey1,solvey2,0.0)
            solv_intsec_MinMaxDir = solveresult
            solv_intsec_MinMaxDir_Val = solvey1(solveresult)

                        
            ''' calc param_MinSlope '''
            extraMinflag = 'none'
            extraMinDayarrs = []
            extraMinVararrs = []
            extraMinDay = 0
            extraMinVar = 0
            
            extraMaxflag = 'none'
            extraMaxDayarrs = []
            extraMaxVararrs = []
            extraMaxDay = 0
            extraMaxVar = 0


            interXinArrs = []
            interYoutArrs = []
            interFallXinArrs = []
            interFallYoutArrs = []
            offset = 0.01
            for loopday in range(extrapolation_Startday,tmpday+1,1):
                slopeY = paramfit_bench_predict[0]*loopday + paramfit_bench_predict[1]
                # print 'slopeY',slopeY,'bench',bars['bench'][loopday],'extrapolation_Startday',extrapolation_Startday,'tmpday',tmpday,'tmpday',tmpday
                # print 'slopeY',slopeY,'loopday',loopday
                if bars['bench'][loopday] < slopeY - offset:
                    if extraMinflag == 'progress':
                        if slopeY - bars['bench'][loopday] > extraMinVar:
                            extraMinDay = loopday
                            extraMinVar = slopeY - bars['bench'][loopday]    
                            extraMinDayarrs.pop(-1)
                            extraMinDayarrs.append(loopday)
                            extraMinVararrs.append(extraMinVar)
                            # print 'progress extraMinVar',extraMinVar,loopday
                    if extraMinflag == 'none':
                        extraMinflag = 'progress'
                        extraMinDay = loopday
                        extraMinVar = slopeY - bars['bench'][loopday]
                        extraMinDayarrs.append(loopday)
                        extraMinVararrs.append(extraMinVar)
                        
                        # print 'none extraMinVar',extraMinVar,loopday
                    
                    if len(extraMinDayarrs) > 0 and len(extraMaxDayarrs) > 0:
                        interMaxDay = min(extraMaxDayarrs, key=lambda x:abs(x-tmpday))
                        tmpExtraMinDayarrs = [x for x in extraMinDayarrs if x < interMaxDay]
                        if len(tmpExtraMinDayarrs) == 0:
                            continue
                        interMinDay = min(tmpExtraMinDayarrs, key=lambda x:abs(x-interMaxDay))
                        interXin = [interMinDay,interMaxDay]
                        param_interSlope = np.polyfit(interXin,bars['bench'][interXin],1)
                        interYout = np.polyval(param_interSlope,interXin)

                        ' find solve '
                        solvey1 = lambda x : paramfit_bench_predict[0]*x+paramfit_bench_predict[1]
                        solvey2 = lambda x : param_interSlope[0]*x+param_interSlope[1]
                        solveresult = findIntersection(solvey1,solvey2,0.0)
                        solv_intsec_Xin = solveresult
                        solv_intsec_Yout = solvey1(solveresult)
                        if len(interXinArrs) == 0 or (len(interXinArrs) > 0 and solv_intsec_Xin != interXinArrs[-1]):
                            interXinArrs.append(solv_intsec_Xin)
                            interYoutArrs.append(solv_intsec_Yout)
                        # if loopday == day:
                        #     print 'extraMaxDayarrs',extraMaxDayarrs,'extraMinDayarrs',extraMinDayarrs
                        #     print 'tmpday',tmpday,'interMaxDay',interMaxDay,'interMinDay',interMinDay,'solv_intsec_Xin',solv_intsec_Xin,'solv_intsec_Yout',solv_intsec_Yout
                elif bars['bench'][loopday] > slopeY:
                    if extraMinflag == 'progress':
                        extraMinflag = 'none'
                        extraMinDay = 0
                        extraMinVar = 0
                        


                if bars['bench'][loopday] > slopeY + offset:
                    if extraMaxflag == 'progress':
                        if  bars['bench'][loopday] - slopeY > extraMaxVar:
                            extraMaxDay = loopday
                            extraMaxVar = bars['bench'][loopday]-  slopeY 
                            extraMaxDayarrs.pop(-1)
                            extraMaxDayarrs.append(loopday)
                            extraMaxVararrs.append(extraMaxVar)
                            # print 'progress extraMaxVar',extraMaxVar,loopday
                    if extraMaxflag == 'none':
                        extraMaxflag = 'progress'
                        extraMaxDay = loopday
                        extraMaxVar = bars['bench'][loopday] - slopeY
                        extraMaxDayarrs.append(loopday)
                        extraMaxVararrs.append(extraMaxVar)
                        
                        # print 'none extraMaxVar',extraMaxVar,loopday
                    if len(extraMinDayarrs) > 0 and len(extraMaxDayarrs) > 0:
                        # print 'extraMinDayarrs extraMaxDayarrs','tmpday',tmpday,'loopday',loopday
                        interMinDay = min(extraMinDayarrs, key=lambda x:abs(x-tmpday))
                        tmpextraMaxDayarrs = [x for x in extraMaxDayarrs if x < interMinDay]
                        if len(tmpextraMaxDayarrs) == 0:
                            continue
                        interMaxDay = min(tmpextraMaxDayarrs, key=lambda x:abs(x-interMinDay))
                        interXin = [interMaxDay,interMinDay]
                        param_interSlope = np.polyfit(interXin,bars['bench'][interXin],1)
                        interYout = np.polyval(param_interSlope,interXin)

                        ' find solve '
                        solvey1 = lambda x : paramfit_bench_predict[0]*x+paramfit_bench_predict[1]
                        solvey2 = lambda x : param_interSlope[0]*x+param_interSlope[1]
                        solveresult = findIntersection(solvey1,solvey2,0.0)
                        solv_intsec_Xin = solveresult
                        solv_intsec_Yout = solvey1(solveresult)
                        if len(interXinArrs) == 0 or (len(interXinArrs) > 0 and solv_intsec_Xin != interXinArrs[-1]):
                            interFallXinArrs.append(solv_intsec_Xin)
                            interFallYoutArrs.append(solv_intsec_Yout)
                elif bars['bench'][loopday] < slopeY:
                    if extraMaxflag == 'progress':
                        extraMaxflag = 'none'
                        extraMaxDay = 0
                        extraMaxVar = 0        
            
            figaxecnt = 0
            disfig = 0
            for i in plt.get_fignums():
                if figaxecnt == 1:
                    disfig = plt.figure(i)
                    break
                figaxecnt += 1  

            tx1 = np.arange(extrapolation_Startday,tmpday+1,1)
            ty1=  gfilterbars['ma2bench'][extrapolation_Startday:tmpday+1].values.tolist()
            disfig.axes[1].plot(tx1,ty1,color='g')        

            if len(interXinArrs) > 0:
                
            
                if day == tmpday:
                    if len(interXinArrs) > 0 :
                        print 'interXinArrs',interXinArrs[-1],'interYoutArrs',interYoutArrs[-1]    
                        disfig.axes[1].plot(interXinArrs,interYoutArrs,'o')        
                    if len(interFallXinArrs) > 0:
                        print 'interFallXinArrs',interFallXinArrs[-1],'interFallYoutArrs',interFallYoutArrs[-1]    
                        disfig.axes[1].plot(interFallXinArrs,interFallYoutArrs,'*')        
            
                        
                
            
            MinSlopeX = extraMinDayarrs
            MaxSlopeX = extraMaxDayarrs
            if len(extraMinDayarrs) > 0:
                param_MinSlope = np.polyfit(MinSlopeX,bars['bench'][MinSlopeX],1)
                MinSlopeY = np.polyval(param_MinSlope,MinSlopeX)

                if len(extraMaxDayarrs) > 0:
                    param_MaxSlope = np.polyfit(MaxSlopeX,bars['bench'][MaxSlopeX],1)
                    MaxSlopeY = np.polyval(param_MaxSlope,MaxSlopeX)

                    
                    if len(interXinArrs) > 0 and len(interFallXinArrs) > 0:
                        if g_interFallYoutVal != 0.0 and g_interFallYoutVal < interFallYoutArrs[-1]:
                            g_interFallYoutVal = interFallYoutArrs[-1]        
                            g_interFallXinDay = interFallXinArrs[-1]
                        if g_interFallYoutVal == 0.0:
                            g_interFallYoutVal = interFallYoutArrs[-1] 
                            g_interFallXinDay = interFallXinArrs[-1]
                     
                    if tmpday - extrapolation_Startday > 35:
                        ma3benchAbovePct = float(len(bars['bench'][extrapolation_Startday+35:tmpday+1][bars['ma2bench'] < bars['bench']]))/float(tmpday-(extrapolation_Startday+35)+1)        
                    else:
                        ma3benchAbovePct = float(len(bars['bench'][extrapolation_Startday:tmpday+1][bars['ma2bench'] < bars['bench']]))/float(tmpday-extrapolation_Startday+1)        
                    # if  182 <= tmpday :
                    #     print 'ma3benchAbovePct',ma3benchAbovePct
                    benchselectPct = len(bars['benchselect'][tmpday - 10:tmpday][bars['benchselect'] == 1])/10.0
                    # print 'benchselectPct',benchselectPct
                    # if 74 <= tmpday <= 90:
                    #     print 'param_MinSlope',param_MinSlope[0], 'paramfit_bench_dir',paramfit_bench_dir[0],'param_MaxSlope',param_MaxSlope[0]\
                    #         ,'paramfit_bench_predict',paramfit_bench_predict[0],'param_setBenchSlope',param_setBenchSlope[0]\
                    #         ,'param_MaxSlope',param_MaxSlope[0]
                    tmp_tstatus = 'none'
                    tmp_buyday = 0
                    tmp_gainsum = 0.0
                    tmp_buybench = 0.0
                    
                    for sigday in range(extrapolation_Startday,tmpday+1):
                        # print 'sigday',sigday,tradesignals[-1]
                        if tmp_tstatus == 'none' and bars['benchselect'][sigday] == 1:
                            tmp_tstatus = 'holding'
                            tmp_buybench = bars['bench'][sigday]
                            tmp_buyday = sigday
                        elif tmp_tstatus == 'holding':
                            
                            if bars['benchselect'][sigday] == 0:
                                # print 'bench tmp_buyday',bars['bench'][tmp_buyday+1],'sellday',tmp_buyday+1,'tmp_buybench',tmp_buybench
                                tmp_gainsum += bars['bench'][sigday] - tmp_buybench
                                tmp_tstatus = 'none'
                                # print 'sigday gain calc',tmp_gainsum,'sigday',sigday,'extrapolation_Startday',extrapolation_Startday
                            if bars['benchselect'][sigday] == 1:
                                continue
                    
                    

                    maxgapdist = 0.0
                    maxgapday =0
                    for sigday in range(extrapolation_Startday,tmpday+1):
                                
                        slopeY = param_MinSlope[0]*sigday + param_MinSlope[1]        
                        if bars['bench'][sigday] > slopeY:
                            tmpdist = abs(param_MinSlope[0]*sigday - bars['bench'][sigday] +param_MinSlope[1])\
                                    /np.sqrt(param_MinSlope[0]**2+param_MinSlope[1]**2)
                            if maxgapdist != 0.0 and maxgapdist < tmpdist:
                                maxgapdist = tmpdist
                                maxgapday = sigday
                            if maxgapdist == 0.0:
                                maxgapdist = tmpdist
                                maxgapday = sigday
                        

                    min_bench_sol_X  = (g_interFallYoutVal - param_MinSlope[1])/param_MinSlope[0]   
                    # max_bench_sol_X  = (bars['bench'][tmpday] - paramfit_bench_predict[1])/paramfit_bench_predict[0]                                                             
                    max_bench_sol_lim_Y = param_MaxSlope[0]*min_bench_sol_X+param_MaxSlope[1]
                    recentStd = bars['bench'][extrapolation_Startday-5:tmpday].std()    
                    curgapdist = abs(param_MinSlope[0]*tmpday - bars['bench'][tmpday] +param_MinSlope[1])\
                                    /np.sqrt(param_MinSlope[0]**2+param_MinSlope[1]**2)
                    # if  paramfit_bench_predict[0] > 0.0 and ma3benchAbovePct > 0.5 and param_MinSlope[0] >= 0.00 and paramfit_bench_dir[0] >= 0.00\
                    #     and ( (min_bench_sol_X+30 > tmpday and g_interFallYoutVal !=0) \
                    #          and ((param_MinSlope[0] < 0.006 and param_MinSlope[0] > 0.002) or (param_MaxSlope[0] < 0.006 and param_MaxSlope[0] > 0.002)))\
                    #     and barsMAranges['eBench'][selindex] < bars['bench'][tmpday]\
                    #     and recentStd + max_bench_sol_lim_Y  - bars['bench'][tmpday]  > 0.0\
                    #     and g_interFallYoutVal <= bars['bench'][tmpday]\
                    #     and maxgapday + 15 > tmpday:
                        # and y_bench_lowlim_next - y_bench_lowlim_prev >= 0.02:
                        
                        

                        # tradesignals.append(tmpday)
                        # if gTradeStatus == False:
                        #     gTradeStatus = True
                        # print 'ma3bench len',ma3benchAbovePct
                    disbarsdf = pd.DataFrame({ 'g_interFallYoutVal':[g_interFallYoutVal],'bench':[bars['bench'][tmpday]]\
                    ,'tmpday':[tmpday] ,'CloseMAsig':[gfilterbars['CloseMAsig'][tmpday]]\
                    ,'ma3benchAbovePct':[ma3benchAbovePct],'param_MinSlope':[param_MinSlope[0]]\
                    ,'paramfit_bench_predict':[paramfit_bench_predict[0]]\
                    ,'param_setBenchSlope':[param_setBenchSlope[0]]
                    ,'param_MaxSlope':[param_MaxSlope[0]]\
                    ,'next-prev':[(yup_bench_lim_next - yup_bench_lim_prev)]\
                    ,'tmp_gainsum':[tmp_gainsum]\
                    ,'min_bench_sol_X':[min_bench_sol_X]\
                    ,'maxlim_y2':[maxlim_y2]\
                    ,'max_bench_sol_lim_Y':[max_bench_sol_lim_Y]\
                    ,'recentStd':[recentStd]\
                    ,'maxgapday':[maxgapday]\
                    ,'maxgapdist':[maxgapdist]\
                    ,'curgapdist':[curgapdist]} 
                    , columns= ['g_interFallYoutVal','bench','tmpday'\
                        ,'CloseMAsig','ma3benchAbovePct','param_MinSlope','paramfit_bench_predict'\
                        ,'param_setBenchSlope','param_MaxSlope','next-prev','tmp_gainsum'\
                        ,'min_bench_sol_X'\
                        ,'maxlim_y2','max_bench_sol_lim_Y','recentStd'\
                        ,'maxgapday','maxgapdist','curgapdist'])
                    display(HTML(disbarsdf.to_html()))
                        # print ' g_interFallXinDay',g_interFallXinDay,'g_interFallYoutVal',g_interFallYoutVal,bars['bench'][tmpday],'tmpday',tmpday ,'CloseMAsig',\
                        #     gfilterbars['CloseMAsig'][tmpday],'param_MinSlope',param_MinSlope[0],'paramfit_bench_predict',paramfit_bench_predict[0],\
                        #     'param_setBenchSlope',param_setBenchSlope[0],'param_MaxSlope',param_MaxSlope[0]
                    #     print ''
                    #     continue
                    # else:
                        

                    #     disbarsdf = pd.DataFrame({ 'hold_False_2':[False]\
                    #         ,'g_interFallYoutVal':[g_interFallYoutVal],'bench':[bars['bench'][tmpday]]\
                    #     ,'tmpday':[tmpday] ,'CloseMAsig':[gfilterbars['CloseMAsig'][tmpday]]\
                    #     ,'ma3benchAbovePct':[ma3benchAbovePct],'param_MinSlope':[param_MinSlope[0]]\
                    #     ,'paramfit_bench_predict':[paramfit_bench_predict[0]]\
                    #     ,'param_setBenchSlope':[param_setBenchSlope[0]]
                    #     ,'param_MaxSlope':[param_MaxSlope[0]]\
                    #     ,'next-prev':[(yup_bench_lim_next - yup_bench_lim_prev)]\
                    #     ,'tmp_gainsum':[tmp_gainsum]\
                    #     ,'min_bench_sol_X':[min_bench_sol_X]\
                    #     ,'maxlim_y2':[maxlim_y2]\
                    #     ,'max_bench_sol_lim_Y':[max_bench_sol_lim_Y]\
                    #     ,'recentStd':[recentStd]\
                    #     ,'maxgapday':[maxgapday]\
                    #     ,'maxgapdist':[maxgapdist]\
                    #     ,'curgapdist':[curgapdist]\
                    #     } 

                    #     , columns= ['hold_False_2','g_interFallYoutVal','bench','tmpday'\
                    #         ,'CloseMAsig','ma3benchAbovePct','param_MinSlope','paramfit_bench_predict'\
                    #         ,'param_setBenchSlope','param_MaxSlope','next-prev','tmp_gainsum'\
                    #         ,'min_bench_sol_X'\
                    #         ,'maxlim_y2','max_bench_sol_lim_Y','recentStd'\
                    #         ,'maxgapday','maxgapdist','curgapdist'])
                    #     display(HTML(disbarsdf.to_html()))        
                    #     gTradeStatus = False
                    #     gDelay_Day = 0        
                    #     continue
                
                
                else:
                    gTradeStatus = False
                    gDelay_Day = 0  
                    # disbarsdf = pd.DataFrame({ 'hold_False_3':[False]\
                    #     ,'g_interFallYoutVal':[g_interFallYoutVal],'bench':[bars['bench'][tmpday]]\
                    # ,'tmpday':[tmpday] ,'CloseMAsig':[gfilterbars['CloseMAsig'][tmpday]]\
                    # ,'ma3benchAbovePct':[False],'param_MinSlope':[param_MinSlope[0]]\
                    # ,'paramfit_bench_predict':[paramfit_bench_predict[0]]\
                    # ,'param_setBenchSlope':[param_setBenchSlope[0]]
                    # ,'param_MaxSlope':[param_MaxSlope[0]]\
                    # ,'next-prev':[(yup_bench_lim_next - yup_bench_lim_prev)]\
                    # ,'tmp_gainsum':[tmp_gainsum]\
                    # ,'bench-g_interFallYoutVal':[bars['bench'][tmpday] - g_interFallYoutVal]\
                    # ,'max_bench_sol_X':[max_bench_sol_X]} 
                    # , columns= ['hold_False_3','g_interFallYoutVal','bench','tmpday'\
                    #     ,'CloseMAsig','ma3benchAbovePct','param_MinSlope','paramfit_bench_predict'\
                    #     ,'param_setBenchSlope','param_MaxSlope','next-prev','tmp_gainsum'\
                    #     ,'bench-g_interFallYoutVal','max_bench_sol_X'])
                    # display(HTML(disbarsdf.to_html()))              
                    continue
                ''' calc param_MinSlope '''
                
                """
                benchEndDay = 0
                benchStartDay = 0
                for rday in range(tmpday,tmpday-40,-1):
                    # print 'rday',rday
                    if bars['benchselect'][rday] == 0 and bars['benchselect'][rday-1]== 1:
                        benchEndDay = rday
                    if bars['benchselect'][rday] == 1 and bars['benchselect'][rday-1]== 0:
                        benchStartDay = rday
                        break
                if benchStartDay == 0:
                    benchStartDay = rday
                if benchEndDay == 0:
                    benchEndDay = tmpday

                previous_bench_gain = bars['bench'][benchEndDay] - bars['bench'][benchStartDay]        
                # print 'previous_bench_gain',previous_bench_gain,'benchEndDay',benchEndDay,'benchStartDay',benchStartDay
                
                # print 'bench', bars['bench'][tmpday] ,'param_MinSlope[0]*tmpday+param_MinSlope[1]', param_MinSlope[0]*tmpday+param_MinSlope[1]
                # print 'solv_intsec_MinMaxDir',solv_intsec_MinMaxDir ,'paramfit_bench_predict[0] ',paramfit_bench_predict[0]
                # print 'paramfit_bench_dir[0]',paramfit_bench_dir[0]
                # print 'paramfit_bench_predict[0]*extrapolation_Startday+paramfit_bench_predict[1]',paramfit_bench_predict[0]*extrapolation_Startday+paramfit_bench_predict[1]
                # print 'param_setBenchSlope[0]*setbenchSelectDay +param_setBenchSlope[1]',param_setBenchSlope[0]*setbenchSelectDay +param_setBenchSlope[1]
                # print 'paramfit_bench_predict[0] < paramfit_bench_dir[0]',paramfit_bench_predict[0] < paramfit_bench_dir[0]
                # print 'round(param_setBenchSlope[0],5) >= round(paramfit_bench_predict[0],5)',round(param_setBenchSlope[0],5) >= round(paramfit_bench_predict[0],5)
                # print 'paramfit_max_bench_dir[0] > 0.0  and paramfit_max_bench_dir[0] < paramfit_bench_predict[0]',paramfit_max_bench_dir[0] > 0.0  and paramfit_max_bench_dir[0] < paramfit_bench_predict[0]
                # print 'previous_bench_gain',previous_bench_gain
                if bars['bench'][tmpday] > param_MinSlope[0]*tmpday+param_MinSlope[1] :
                    # print 'rangegain',rangegain
                    # print 'solv_intsec_MinMaxDir > tmpday',solv_intsec_MinMaxDir > tmpday
                    # print 'solv_intsec_MinMaxDir <  extrapolation_Startday ',solv_intsec_MinMaxDir <  extrapolation_Startday 
                    # print '(paramfit_bench_predict[0]*extrapolation_Startday+paramfit_bench_predict[1] < param_setBenchSlope[0]*setbenchSelectDay +param_setBenchSlope[1])',(paramfit_bench_predict[0]*extrapolation_Startday+paramfit_bench_predict[1] < param_setBenchSlope[0]*setbenchSelectDay +param_setBenchSlope[1])
                    if solv_intsec_MinMaxDir > tmpday or solv_intsec_MinMaxDir <  extrapolation_Startday :
                        if paramfit_bench_predict[0] > 0.0 and paramfit_bench_dir[0] > 0.0\
                            and ((paramfit_bench_predict[0]*extrapolation_Startday+paramfit_bench_predict[1] <= param_setBenchSlope[0]*setbenchSelectDay +param_setBenchSlope[1])\
                            ):
                            # print  'extraMinDayarrs',extraMinDayarrs 
                            # if len(extraMinDayarrs) > 2:

                            if round(paramfit_bench_predict[0],5) <= round(paramfit_bench_dir[0],5) or round(param_setBenchSlope[0],5) >= round(paramfit_bench_predict[0],5) \
                                or (paramfit_max_bench_dir[0] > 0.0  and round(paramfit_max_bench_dir[0],5) <= round(paramfit_bench_predict[0])):
                                # if bars['benchselect'][tmpday] == 1:
                                if previous_bench_gain != 0 and previous_bench_gain > 0.0:
                                    tradesignals.append(tmpday)
                """    

                

        figaxecnt = 0
        disfig = 0
        for i in plt.get_fignums():
            if figaxecnt == 1:
                disfig = plt.figure(i)
                break
            figaxecnt += 1
        tmpbenchX = np.arange(extrapolation_Startday,day+1,1)
        disfig.axes[1].plot(tmpbenchX,bars['bench'][tmpbenchX])          
        if len(tradesignals) > 0:
            disfig.axes[1].plot(tradesignals,bars['bench'][tradesignals],'o')      
            print 'tradesignal',tradesignals[-1],tradesignals

        tstatus = 'none'
        buyday = 0
        gainsum = 0.0
        global totalgain
        for tsig in tradesignals:
            # print 'tsig',tsig,tradesignals[-1]
            if tstatus == 'none':
                tstatus = 'holding'
                buybench = bars['bench'][tsig]
                buyday = tsig
            elif tstatus == 'holding':
                if buyday + 1 == tsig:
                    buyday = tsig
                if tsig == tradesignals[-1] or buyday + 1 < tsig:
                    # print 'bench buyday',bars['bench'][buyday+1],'sellday',buyday+1,'buybench',buybench
                    gainsum += bars['bench'][buyday+1] - buybench
                    # global totalgain
                    # totalgain += gainsum
                    buybench = bars['bench'][tsig]
                    buyday = tsig
                    print 'tsig gain calc',gainsum,'tsig',tsig
    except Exception,e:
        print PrintException()

global bars
bars = 0
global barsMAranges
barsMAranges = 0
global gfilterbars
gfilterbars = 0
global tday
tday = 0
global predictdf
predictdf = 0

global extrapolation_Startday
extrapolation_Startday = 0

global daywindow
daywindow = 0
global setupvertex 
setupvertex = 0
global f_setupvertex
f_setupvertex = False
global g_setup_mindir
global g_setup_maxdir
g_setup_mindir = 0
g_setup_maxdir = 0
def setup():
    import tradingalgo as tal
    import ver2_0_i as v20
    import data_mani as dmani

    global bars
    # bars = pd.DataFrame(pd.read_csv('bars.csv'))
    bars = v20.plot_bars
    bars['A'] = 0
    bars['B'] = 0
    bars['C'] = 0
    bars['D'] = 0
    bars['E'] = 0
   
    # bars.Date =[ datetime.strptime(tdate, '%Y-%m-%d %H:%M:%S')  for tdate in bars.Date ]

    global barsMAranges
    # barsMAranges = pd.DataFrame(pd.read_csv('barsMAranges.csv'))
    barsMAranges = dmani.gbarMAranges
    global gfilterbars
    
    gfilterbars = tal.gfilterbars
    # print gfilterbars['filter_hmm_real'][-50:]
    # gfilterbars = pd.DataFrame(pd.read_csv('gfilterbars.csv'))
    # display(HTML(gfilterbars.to_html()))
    global predictdf
    predictdf = pd.DataFrame()
    global daywindow
    daywindow = len(bars)

def showPredictMinMaxSameStateMinMax_hmmrealSignal(predictdf,day,daycnt):
    fig1 = plt.figure(figsize=(15, 2))
    gs = gridspec.GridSpec(1, 3,width_ratios=[1,1,1])
        
    ax1 = fig1.add_subplot(gs[0],  ylabel='predict Min line')
    ttmp = np.arange(0,daycnt+1,1)
    ax1.plot(ttmp,predictdf['predict Min'][:daycnt+1])
    ax1.plot(ttmp,predictdf['predict Min'][:daycnt+1],'o', markersize=7)
    ax1.plot(ttmp,predictdf['same Min'][:daycnt+1],color='r')
    ax1.plot(ttmp,predictdf['same Min'][:daycnt+1],'o', markersize=7,color='r')


    ax1 = fig1.add_subplot(gs[1],  ylabel='predict Max line')
    ax1.plot(ttmp,predictdf['predict Max'][:daycnt+1])
    ax1.plot(ttmp,predictdf['predict Max'][:daycnt+1],'o', markersize=7)
    ax1.plot(ttmp,predictdf['same Max'][:daycnt+1],color='r')
    ax1.plot(ttmp,predictdf['same Max'][:daycnt+1],'o', markersize=7,color='r')    

    ax1 = fig1.add_subplot(gs[2],  ylabel='filter_hmm_real')
    try:
        disbars = gfilterbars['bench'][day-10:day+1]
        ax1.plot(disbars.index,disbars,lw=2)
        # gfilterbars['bench'][day-10:day+1].plot(ax=ax1,lw=2)
        
    except  Exception,e:
        print 'showPredictMinMaxSameStateMinMax_hmmrealSignal error',e    
    ax1.plot(gfilterbars['bench'][day-10:day+1].ix[gfilterbars['filter_hmm_real'] == 1].index,
                 gfilterbars['bench'][day-10:day+1].ix[gfilterbars['filter_hmm_real'] == 1],
                 '.', markersize=10, color='r')         

def showtotalStdVolandVol(gfilterbars,day,daywindow):
    # """
    fig1 = plt.figure(figsize=(15, 2))
    ax5 = fig1.add_subplot(131,  ylabel='filter_totalstdvolsum')
    disbars = gfilterbars['filter_totalstdvolsum'][day-20:daywindow+1]
    disbars.plot(ax=ax5,lw=2)
    ax5.plot(gfilterbars['filter_totalstdvolsum'][day-20:daywindow+1].index,
         gfilterbars['filter_totalstdvolsum'][day-20:daywindow+1],
         'o', markersize=5, color='r')        
    ax5 = fig1.add_subplot(132,  ylabel='filter_totalvolsum')
    disbars = gfilterbars['filter_totalvolsum'][day-20:daywindow+1]
    disbars.plot(ax=ax5,lw=2)
    ax5.plot(gfilterbars['filter_totalvolsum'][day-20:daywindow+1].index,
         gfilterbars['filter_totalvolsum'][day-20:daywindow+1],
         'o', markersize=5, color='r')        
    # """
def showSelectSignal(fig,bars,gfilterbars,day,daywindow):
    
    ax1 = fig.add_subplot(122,  ylabel='benchselect')
    disbars = bars[['bench','benchselect']]
    # disbars[['bench']][day-30:day+1].plot(ax=ax1,lw=2)
    ax1.plot(disbars['bench'][day-30:day+1].index,disbars['bench'][day-30:day+1])
    ax1.plot(disbars['bench'][day-30:day+1].ix[disbars['benchselect'][day-30:day+1] == 1].index,
             disbars['bench'][day-30:day+1][disbars['benchselect'][day-30:day+1] == 1],
             '.', markersize=10, color='r')                
    ax1.grid(True)
    # print 'benchselect',bars['benchselect'][day]

def show_args(Code= 41):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Polyfit may be poorly conditioned')

        global daywindow
        global bars
        global barsMAranges
        global gfilterbars
        global predictdf
        global gCode        
        global setupvertex 

        try:
            # fig1 = plt.figure(figsize=(20, 3))
            # ax1 = fig1.add_subplot(111,  ylabel='bench')
            gCode  = Code
            daywindow = Code
            day = daywindow   
            # disbars = bars['bench'][daywindow-20:daywindow+1]
            # disbars.plot(ax=ax1,lw=2)
            """  --------------  range found -------------------- """
            rangeFound = False
            for eDate in reversed(barsMAranges['eDate']):
                # print 'eDate',eDate,bars.Date[day],type(eDate) , type(bars.index[day])
                # neweDate = datetime.strptime(eDate, '%Y-%m-%d %H:%M:%S')
                # barsDate = datetime.strptime(bars.Date[day], '%Y-%m-%d %H:%M:%S')
               
                # print neweDate,barsDate
                # print eDate,bars.index[day]

                if eDate<bars.index[day]:
                    # print 'range found!!',bars.index[day]
                    selrange = barsMAranges[barsMAranges['eDate'] == eDate]
                    # print 'selrange.index:',selrange.index[0]
                    selindex = selrange.index[0]
                    # selrange = selrange.reset_index()
                    # print 'selrange.index ',selrange.index
                   
                    rangeFound = True

                    break

           

            for nday in range(len(bars)):
                if eDate == bars.index[nday]:
                    lasteDateDay = nday
                    break       
           

            if rangeFound == True:
                               
                scut = selrange['scut'][selindex]
                ecut = selrange['ecut'][selindex]
                esbench = selrange['eBench'][selindex] - selrange['sBench'][selindex]
                # print 'esbench',esbench
               
                if scut == -1 and esbench > 0.0 :
                    subtstate = 'low_rising'
                    barsMAranges['subtstate'][selindex] = subtstate
                    
                if scut == 1 and esbench > 0.0 :
                    subtstate = 'high_rising'
                    barsMAranges['subtstate'][selindex] = subtstate
                    
                if scut == -1 and esbench < 0.0:   
                    subtstate = 'low_falling'
                    barsMAranges['subtstate'][selindex] = subtstate
                    
                if scut == 1 and esbench < 0.0 :   
                    subtstate = 'high_falling'   
                    barsMAranges['subtstate'][selindex] = subtstate   
                        
            # disbars = selrange[['sDate','eDate','subtstate','days']]
            # display(HTML(disbars.to_html()))       
            """  --------------  range found -------------------- """

            """  --------------  hmm total  -------------------- """
            stkdata = bars
            # disbars = stkdata[:daywindow+1]
            # disbars = disbars.tail()
            # display(HTML(disbars.to_html()))               
            stkmchnlrndata = np.column_stack([stkdata['Close'][:daywindow+1], stkdata['Volume'][:daywindow+1]])
            ncomponents = 5
            lrnmodel, hiddenstates = stkHMM(stkmchnlrndata, ncomponents)
            nxtstateidx = lrnmodel.transmat_[hiddenstates[len(hiddenstates)-1], :]
            nxtstateprob = np.amax(nxtstateidx)
            nxtstate = np.argmax(nxtstateidx)

            # print 'nxtstate',nxtstate,'nxtstateprob',nxtstateprob,'nxtstateidx',nxtstateidx

            # print "means and vars of each hidden state"
            allmeans = []
            stdvarpct_mean = 0
            for i in xrange(ncomponents):
                # print "%dth hidden state" % i,'mean = ',lrnmodel.means_[i][0],
                covars = np.diag(lrnmodel.covars_[i])
                stdvar = np.sqrt(covars[0])
                stdvarpct = stdvar/lrnmodel.means_[i][0]
                stdvarpct_mean += stdvarpct
                # print 'var = ',stdvar, ' pct = ',stdvarpct,'high price:',stdvar+lrnmodel.means_[i][0],'low price:',-stdvar+lrnmodel.means_[i][0]
                # print ""
                allmeans.append(lrnmodel.means_[i][0])
        
            # print 'cur price',bars['Close'][Code],'benchdiff',bars['benchdiff'][day] 
            stdvarpct_mean = stdvarpct_mean/5.0
            # print 'stdvarpct_mean',stdvarpct_mean
            covars = np.diag(lrnmodel.covars_[nxtstate])
            stdvar = np.sqrt(covars[0])
            stdvarpct = stdvar/lrnmodel.means_[nxtstate][0]
            # print 'nxtstate',nxtstate,'var = ',stdvar, ' pct = ',stdvarpct
            # print 'nxtstate',nxtstate,'high price:',stdvar+lrnmodel.means_[nxtstate][0],'low price:',-stdvar+lrnmodel.means_[nxtstate][0]

            idx = (hiddenstates[:daywindow+1] == nxtstate)
            idx = idx[-20:]
            # print 'idx',idx
            windowbars = bars[:daywindow+1]
            # disbars = windowbars[-20:]
            # display(HTML(windowbars.to_html()))
            # print 'nxtstate average',windowbars[-20:]['Close'][idx].mean(),'nxtstate min',windowbars[-20:]['Close'][idx].min()\
            #         ,'nxtstate max',windowbars[-20:]['Close'][idx].max()
            # print len(windowbars)               
            # for nday in range(len(windowbars[-20:])):
                # print nday,windowbars['Close'][len(windowbars)-20+nday]
                # if windowbars[-20:]['Close'][idx].max() == windowbars['Close'][len(windowbars)-20+nday]:
                #     print 'nxtstate max day ',windowbars.index[len(windowbars)-20+nday],windowbars['Close'][len(windowbars)-20+nday]
                #     print 'nxtstate max day ',len(windowbars)-20+nday
                #     break
            # print windowbars['Close'][-20:]
            # print 'nxtstate max last day',day - (len(windowbars)-20+nday)
            """  --------------  hmm total  -------------------- """

            """  --------------  range last days, range min  -------------------- """
            rangegain = bars['Close'][lasteDateDay:day+1].pct_change().cumsum()[-1:].values
            # print 'range last days:',day-lasteDateDay,'range pct gain:',rangegain
            # print 'filter_hmm_real:',gfilterbars['filter_hmm_real'][day]

            searchday = selrange['days'][selindex]
            if searchday < 10 and selindex > 0:
                tdays = 0
                searchidx = selindex
                for ndays in reversed(barsMAranges['days'][:selindex]):
                    tdays += ndays
                    if tdays > 10:
                        break
                    searchidx -= 1

                searchday = tdays

            rangeMinbench = bars['bench'][day - searchday:day+1].min()    
            rangeMinday = 0
            for tmpval in bars['bench'][day-searchday:day+1]:
                if tmpval == rangeMinbench:
                    break
                rangeMinday += 1
            rangeMinday = rangeMinday + (day-searchday)

            # print 'gain from min day :',bars['bench'][day] -  bars['bench'][day - searchday:day+1].min(),'min Close,',bars['Close'][day - searchday:day+1].min()
            # print 'min date', bars['bench'][day - searchday:day+1][bars.bench == bars['bench'][day - searchday:day+1].min()].index            
            if bars['benchdiff'][day] > stdvarpct:
                print 'stdpct over benchdiff:',bars['benchdiff'][day],'stdvarpct:',stdvarpct,bars.index[day]
            else:
                print 'stdpct over benchdiff: not'
            """  --------------  range last days, range min  -------------------- """

            """  --------------  hmm state sort  -------------------- """
            allmeans.sort()
            # print 'allmeans',allmeans
            # bars['A'][Code] = allmeans[4]
            # bars['B'][Code] = allmeans[3]
            # bars['C'][Code] = allmeans[2]
            # bars['D'][Code] = allmeans[1]
            # bars['E'][Code] = allmeans[0]

            for i in range(0,ncomponents):
                # print allmeans[i]
                if int(lrnmodel.means_[nxtstate][0]) == int(allmeans[i]):
                    # print 'allmeans[i]',allmeans[i],i
                    break
            # if i == 0:
            #     print 'current state 0'
            # elif i == 1:
            #     print 'current state 1'
            # elif i == 2:
            #     print 'current state 2'
            # elif i == 3:
            #     print 'current state 3'
            # elif i == 4:
            #     print 'current state 4'
            # elif i == 5:
            #     print 'current state 5'   

            for i in range(0,ncomponents):
                # print allmeans[i]
                if int(lrnmodel.means_[hiddenstates[Code-1]][0]) == int(allmeans[i]):
                    # print 'previous allmeans[i]',allmeans[i],i
                    break   
            # if i == 0:
            #     print 'previous state 0'
            # elif i == 1:
            #     print 'previous state 1'
            # elif i == 2:
            #     print 'previous state 2'
            # elif i == 3:
            #     print 'previous state 3'
            # elif i == 4:
            #     print 'previous state 4'
            # elif i == 5:
            #     print 'previous state 5'   

            sameStatecnt = 0
            # print 'hiddenstates',hiddenstates[-5:],'nxtstate',nxtstate,'hiddenstates last',hiddenstates[-1]
            for rstate in reversed(hiddenstates):
                # if rstate == nxtstate:
                if rstate == hiddenstates[-1]:
                    sameStatecnt +=1
                else:
                    break
            if sameStatecnt == 0 :
                sameStatecnt = 1
            # print 'sameStatecnt:',sameStatecnt,

            # print 'range sameStatecnt gain:',bars['Close'][day-sameStatecnt:day+1].pct_change().cumsum()[-1:].values,'todayBench',bars['bench'][day]
            """  --------------  hmm state sort  -------------------- """

            """  --------------  hmm plot  , vertex polyfit , sameState polyfit-------------------- """        
            fig1 = plt.figure(figsize=(15, 2))
            ax1 = fig1.add_subplot(121,  ylabel='hmm')
            disbars = bars[:daywindow+1]
            disbars['Close'].plot(ax=ax1,lw=2)
            
            for i in xrange(ncomponents):
                idx = (hiddenstates == i)
                ax1.plot(disbars.index[idx], disbars['Close'][idx], 'o')
                
            # vertex = day - sameStatecnt 
            vertex = lasteDateDay

            fitx12 = np.arange(vertex,day+1,1.0)
            paramfit = np.polyfit(fitx12,bars['bench'][vertex:day+1],1)
            fity12 = np.polyval(paramfit,fitx12)
            

            fitx_samestate = np.arange(day-sameStatecnt+1,day+1,1.0)
            # print 'fitx_samestate',fitx_samestate
            paramfit_samestate = np.polyfit(fitx_samestate,bars['bench'][day-sameStatecnt+1:day+1],1)
            fity_samestate = np.polyval(paramfit_samestate,fitx_samestate)
            # print 'sameStatecnt min',bars['bench'][day-sameStatecnt+1:day+1].min(),'max',bars['bench'][day-sameStatecnt+1:day+1].max(),'bench today',bars['bench'][day]

            """  --------------  hmm plot  , vertex polyfit , sameState polyfit-------------------- """        

            """  --------------  coef, coeflowlim-------------------- """        
            if day <= 30 or vertex <=30:
                tall = np.arange(day-30, day+1, 1.0)      
                t1 = np.arange(day-30, day+10, 1.0)  
            else:
                tall = np.arange(vertex-30, day+1, 1.0)      
                t1 = np.arange(vertex-30, day+10, 1.0)  
            
            
            # if stdvarpct_mean <= 0.02:
            #     coef = 0.0002
            # if 0.02 < stdvarpct_mean <= 0.03:
            #     coef = 0.00025
            # if 0.03 <= stdvarpct_mean :    
            if paramfit[0] <= 0.0:
                coef = 0.0002
            if 0.0 < paramfit[0] <= 0.003:
                coef = 0.0002
            if 0.003 <= paramfit[0] <= 0.004:    
                coef = 0.0001
            if 0.004 < paramfit[0] :    
                coef = 0.00007
            interpx = [0.0,0.001,0.002,0.003,0.004,0.005,0.006,0.007]
            interpy = [0.0002,0.0002,0.0002,0.00018,0.00017,0.00016,0.00015,0.00014]

            interlowlimpx = [-0.002,-0.001,0.0,0.001,0.002,0.003,0.004,0.005,0.006,0.007]
            interlowlimpy = [0.0003,0.00035,0.0004,0.0005,0.00055,0.0006,0.00065,0.0007,0.00075,0.0008]
            coeflowlim = 0.0

            if paramfit[0] >= 0.007:
                coef = 0.00015
                coeflowlim = 0.0008
            if paramfit[0] <= 0.0:
                coef = 0.0002
                if  -0.002 < paramfit[0] <= 0.0:
                    interlowlimfl = sp.interp1d(interlowlimpx, interlowlimpy,kind='linear',bounds_error=False)
                    coeflowlim = interlowlimfl(paramfit[0])    
                else:
                    coeflowlim = 0.0003
            if 0.0< paramfit[0] < 0.007:
                # interpolation
                interfl = sp.interp1d(interpx, interpy,kind='linear',bounds_error=False)
                coef = interfl(paramfit[0])
                
                interlowlimfl = sp.interp1d(interlowlimpx, interlowlimpy,kind='linear',bounds_error=False)
                coeflowlim = interlowlimfl(paramfit[0])    

            # print 'coef',coef,'coeflowlim',coeflowlim

            y1= coef*(t1-(vertex))**2+disbars['bench'][vertex]
            """  --------------  coef, coeflowlim-------------------- """        

            """  --------------  plot ,right side of vertex over y2 , maxlim_y2 -------------------- """                
            if vertex <= 50:
                t2 = np.arange(0, vertex+1, 1.0)      
                y2 = coef*(t2-(vertex))**2+disbars['bench'][vertex]        
                maxlim_y2_idx = bars['bench'][:vertex+1] > y2
                maxlim_y2 = bars['bench'][:vertex+1][maxlim_y2_idx == True].max()

                # maxlim_y2_second = bars['bench'][:vertex+1].max()
                # if maxlim_y2_second > maxlim_y2:
                #     maxlim_y2 = maxlim_y2_second
                    
                if np.isnan(maxlim_y2) == True:
                    maxlim_y2 = bars['bench'][:vertex+1].max()
                    maxlim_y2_prev = maxlim_y2
                maxlim_y2_day = 0
                for tmpbench in bars['bench'][:vertex+1]:
                    if tmpbench == maxlim_y2:
                        break
                    maxlim_y2_day+=1    
                t_right = np.arange(vertex, day+1, 1.0)          
                y2 = coef*(t_right-(vertex))**2+disbars['bench'][vertex]        
                y_right_idx = bars['bench'][vertex:day+1] > y2
                """ ----------------hmm over y1--------------------"""
                # ax1 = fig1.add_subplot(122,  ylabel='hmm')
                # disbars = bars[day-30:daywindow+2]
                # disbars['bench'].plot(ax=ax1,lw=2)
                # for tmpvval,tmpvindex in zip(y_right_idx,y_right_idx.index):
                #     if tmpvval == True:
                #         ax1.axvline(bars.index[vertex], ymin=0.0, ymax = 0.915, linewidth=3, color='k')
                #         ax1.axvline(tmpvindex, ymin=0.0, ymax = 0.915, linewidth=3, color='r')
                """ ----------------hmm over y1--------------------"""
                showSelectSignal(fig1,bars,gfilterbars,day,daywindow)
                # y_right_vidx = []
                # for tmpidx in y_right_idx.index:
                #     for cnt in range(len(bars[:day+1])):
                #         if tmpidx == bars.index[cnt]:
                #             y_right_vidx.append(cnt)
                # print 'y_right_vidx',y_right_vidx
                # print 'y_right_idx',y_right_idx,y_right_idx.index
                # print 'maxlim_y2',maxlim_y2,'date',bars.index[maxlim_y2_day],bars['bench'][bars.index[maxlim_y2_day]]
            else:
                t2 = np.arange(vertex-50, vertex+1, 1.0)      
                y2 = coef*(t2-(vertex))**2+disbars['bench'][vertex]        
                maxlim_y2_idx = bars['bench'][vertex-50:vertex+1] > y2
                maxlim_y2 = bars['bench'][vertex-50:vertex+1][maxlim_y2_idx == True].max()

                # maxlim_y2_second = bars['bench'][vertex-50:vertex+1].max()
                # if maxlim_y2_second > maxlim_y2:
                #     maxlim_y2 = maxlim_y2_second

                if len(bars[:vertex-50]) < 100:
                    maxlim_y2_prev = bars['bench'][vertex-100:vertex-50].max()
                else:
                    maxlim_y2_prev = bars['bench'][:vertex-50].max()

                if np.isnan(maxlim_y2) == True:
                    maxlim_y2 = bars['bench'][vertex-50:vertex+1].max()
                
                maxlim_y2_day = 0
                for tmpbench in bars['bench'][:vertex+1]:
                    if tmpbench == maxlim_y2:
                        break
                    maxlim_y2_day+=1

                t_right = np.arange(vertex, day+1, 1.0)          
                y2 = coef*(t_right-(vertex))**2+disbars['bench'][vertex]        
                y_right_idx = bars['bench'][vertex:day+1] > y2
                
                """ ----------------hmm over y1--------------------"""
                # ax1 = fig1.add_subplot(122,  ylabel='hmm over y1')
                # disbars = bars[day-30:daywindow+2]
                # ax1.plot(disbars.index,disbars['bench'],lw=2)
                # for tmpvval,tmpvindex in zip(y_right_idx,y_right_idx.index):
                #     if tmpvval == True:
                #         ax1.axvline(bars.index[vertex], ymin=0.0, ymax = 0.915, linewidth=3, color='k')
                #         ax1.axvline(tmpvindex, ymin=0.0, ymax = 0.915, linewidth=3, color='r')
                """ ----------------hmm over y1--------------------"""
                showSelectSignal(fig1,bars,gfilterbars,day,daywindow)

                # y_right_vidx = []
                # for tmpidx in y_right_idx.index:
                #     for cnt in range(len(bars[:day+1])):
                #         if tmpidx == bars.index[cnt]:
                #             y_right_vidx.append(cnt)

                # print 'y_right_idx',y_right_idx,y_right_idx.index
                
                # print 'maxlim_y2_day',maxlim_y2_day,'vertex',vertex    
                # maxlim_y2_day = maxlim_y2_day + vertex-50     
                # print 'maxlim_y2',maxlim_y2,'date',bars.index[maxlim_y2_day],bars['bench'][bars.index[maxlim_y2_day]]
            
            """  --------------  plot ,right side of vertex over y2 , maxlim_y2 -------------------- """                
            
            """  --------------- hmm ploy  ----------------------"""
            fig1 = plt.figure(figsize=(15, 2))
            gs = gridspec.GridSpec(1, 2,width_ratios=[2,1])
            ax1 = fig1.add_subplot(gs[0],  ylabel='hmm poly')

            tall2 = np.arange(0,len(bars[:day+1]),1)
            tally2 = bars['bench'][:day+1]
            tally2 = tally2.reset_index()
            tally2 = tally2.drop('Date',1)
            for i in xrange(ncomponents):
                idx = (hiddenstates == i)
                ax1.plot(tall2[idx], tally2[idx], 'o')

            # if day <= 30 or vertex <=30:
                # ax1.plot(tall,bars['bench'][day-30:day+1],'o',t1,y1,'r--')
                # ax1.plot(t1,y1,'r--')
            # else:
                # ax1.plot(tall,bars['bench'][vertex-30:day+1],'o',t1,y1,'r--')
                # ax1.plot(t1,y1,'r--')
            ax1.axvline(vertex, ymin=0.0, ymax = 0.615, linewidth=2, color='k')
            if np.isnan(maxlim_y2 ) == False:
                ax1.axvline(maxlim_y2_day, ymin=0.0, ymax = 0.615, linewidth=2, color='b')
                ax1.axhline(y = maxlim_y2, xmin=0.4, xmax = 0.915, linewidth=2, color='b')
                if bars['bench'][day] > maxlim_y2 and np.isnan(maxlim_y2_prev) == False:
                    ax1.axhline(y = maxlim_y2_prev, xmin=0.4, xmax = 0.915, linewidth=2, color='b',linestyle='--')

            # ax1.scatter(day-5,disbars['bench'][day-5], color="g", marker="*")
            # if len(y_right_idx) >0:
            #     ax1.axvline(y_right_idx.index, ymin=0.0, ymax = 0.315, linewidth=2, color='r-')

            # ax1.plot(fitx12,fity12,'g-',linewidth=2.5)
            # ax1.plot(fitx_samestate,fity_samestate,'c-',linewidth=2.5)
            
            if vertex > 50:
                ax1.set_ylim(ymin = bars['bench'][vertex-50:day].min()-0.1,ymax = bars['bench'][vertex-50:day].max()+0.1)
            else:
                ax1.set_ylim(ymin = bars['bench'][vertex-20:day].min()-0.1,ymax = bars['bench'][vertex-20:day].max()+0.1)

            ax1.set_xlim(xmin = day-50,xmax = day+10)    
            
            fitx_min = np.arange(day-10,day+10,1.0)
            paramfit_min = np.polyfit(fitx_min,bars['bench'][day-10:day+10],1)
            fity_min = np.polyval(paramfit_min,fitx_min)
            # ax1.plot(fitx_min,fity_min,color= 'm',linewidth=2)  

            ' find solve with recent min , intersection with y1'
            solvey1 = lambda x : paramfit_min[0]*x+paramfit_min[1]
            solvey2 = lambda x : coef*(x-(vertex))**2+bars['bench'][vertex]
            # print 'paramfit_min[0]',paramfit_min[0],'paramfit_min[1]',paramfit_min[1],'vertex',vertex,'bench',bars['bench'][vertex]
            try:
                solveresult = findIntersection(solvey1,solvey2,0.0)
                solv_intsec_Y1 = solveresult
                solv_intsec_Y1_Val = solvey1(solveresult)
                # print 'intersection with y1 solveresult',solveresult,'value',solvey1(solveresult)
            except Exception,e:
                print 'intersection with y1 solve error',e
            # print 'polyfit range slope param[0]',paramfit[0],'paramfit_samestate',paramfit_samestate[0],'min slope',paramfit_min[0]
            
            if day > 40:
                tmpmin = bars['bench'][day-40:day+1].min()
                tmpdaycnt = 0
                for val in bars['bench'][day-40:day+1]:
                    if val == tmpmin:
                        break
                    tmpdaycnt += 1
                tmpminday = tmpdaycnt+day-40    
                minvertexy = coef*(day-(vertex))**2+bars['bench'][vertex]
                ax1.axhline(y = bars['bench'][tmpminday], xmin=0.0, xmax = 1.0, linewidth=2, color='k')
                if minvertexy > bars['bench'][day]:
                    tmpdaycnt = 0.0
                    tmpdaysum = 0.0
                    for tmpday,tmpval in reversed(zip(np.arange(vertex,day+1,1),bars['bench'][vertex:day+1])):    
                        if tmpval < coef*(tmpday-(vertex))**2+bars['bench'][vertex]:
                            tmpdaysum += tmpday    
                            tmpdaycnt += 1
                    tmpdayvertex = tmpdaysum/tmpdaycnt
                    # print 'tmpdayvertex',tmpdayvertex
                    # print 'minvertexy',minvertexy , bars['bench'][day]
                    t1_minvertex = np.arange(day-10, day+10, 1.0)  
                    y1_minvertex = coeflowlim*(t1_minvertex-tmpdayvertex)**2+\
                                    bars['bench'][vertex] - (bars['bench'][vertex] - bars['bench'][tmpminday])*0.4

                    # ax1.plot(t1_minvertex,y1_minvertex,'r--',linewidth=2)                  
                    # if  bars['bench'][day] >= coeflowlim*(day-tmpdayvertex)**2+\
                    #                 bars['bench'][vertex] - (bars['bench'][vertex] - bars['bench'][tmpminday])*0.4:
                    #     ax1.plot(t1_minvertex,y1_minvertex,'m--',linewidth=2)  
                    #     ax1.plot(tmpdayvertex,bars['bench'][vertex] - (bars['bench'][vertex] - bars['bench'][tmpminday])*0.4,'*')  
            """  --------------- hmm ploy  ----------------------"""                    



            """  --------------- hmm residual  ----------------------"""                    
            ax1 = fig1.add_subplot(gs[1],  ylabel='hmm setup buy vertex')
            # resiy =  bars['bench'][vertex:day+1] - fity12
            # ax1.plot(fitx12, resiy, 'ro')
            # ax1.axhline(y = 0, xmin=0.0, xmax = 1.0, linewidth=2, color='k')
            # ax1.axhline(y = -1.0*stdvarpct_mean, xmin=0.0, xmax = 1.0, linewidth=2, color='r')

            """  --------------- hmm residual  ----------------------"""                    

            """ ----------------showtotalStdVolandVol -------------------"""
            # showtotalStdVolandVol(gfilterbars,day,daywindow)
            """ ----------------showtotalStdVolandVol -------------------"""
            
            """  --------------- plot extrapolation_y  ----------------------"""                                             

            fig1 = plt.figure(figsize=(15, 2))
            gs = gridspec.GridSpec(1, 2,width_ratios=[2,1])
            ax1 = fig1.add_subplot(gs[0],  ylabel='extrapolation')

            if day > 40:
                'finding min low list '
                minlowlist = []
                backupminlowlist = []
                prevminval = 0
                for tmpday in range(len(bars[:day+1])-40,len(bars[:day+1]),10):
                    minval = bars['bench'][tmpday:tmpday+10].min()
                    # print 'minval',minval
                    if prevminval == 0:
                        prevminval = minval
                        minlowlist.append(minval)
                    else:
                        if prevminval < minval:
                            minlowlist.append(minval)    
                            
                        else:
                            minlowlist.pop(-1)
                            minlowlist.append(minval)    
                    prevminval = minval        
                    backupminlowlist.append(minval)
                # print 'minlowlist',minlowlist,min(minlowlist)
                if len(minlowlist) == 1:
                    minlowlist = backupminlowlist
                minlowdaylist = []
                for tmpmin in minlowlist:
                    tmpdaycnt = 0
                    for val in bars['bench'][day-40:day+1]:
                        if val == tmpmin:
                            break
                        tmpdaycnt += 1
                    minlowdaylist.append(tmpdaycnt+day-40)
                # print 'minlowdaylist',bars['bench'][minlowdaylist]
                'finding max up list '
                maxuplist = []
                backupmaxuplist = []
                prevmaxval = 0
                for tmpday in range(len(bars[:day+1])-40,len(bars[:day+1]),10):
                    maxval = bars['bench'][tmpday:tmpday+10].max()
                    # print 'maxval',maxval
                    if prevmaxval == 0:
                        prevmaxval = maxval
                        maxuplist.append(maxval)
                    else:
                        if prevmaxval > maxval:
                            maxuplist.append(maxval)    
                            
                        else:
                            maxuplist.pop(-1)
                            maxuplist.append(maxval)    
                    prevmaxval = maxval 
                    backupmaxuplist.append(maxval)
                if len(maxuplist) == 1:
                    maxuplist = backupmaxuplist

                maxupdaylist = []
                for tmpmax in maxuplist:
                    tmpdaycnt = 0
                    for val in bars['bench'][day-40:day+1]:
                        if val == tmpmax:
                            break
                        tmpdaycnt += 1
                    maxupdaylist.append(tmpdaycnt+day-40)
                # print 'maxupdaylist',bars['bench'][maxupdaylist]    

                'plot min list '
                fitx_dir = minlowdaylist
                paramfit_dir = np.polyfit(fitx_dir,bars['Close'][minlowdaylist],1)
                fity_dir = np.polyval(paramfit_dir,fitx_dir)
                # ax1.plot(fitx_dir,fity_dir,color='m',linewidth=3.5)        
                predictx_lowlim = np.arange(day+1,day+10,1)
                predicty_lowlim = paramfit_dir[0]*predictx_lowlim+paramfit_dir[1]
                # ax1.plot(predictx_lowlim,predicty_lowlim,'m--',linewidth=3.5)        
                ylowlim = paramfit_dir[0]*day+paramfit_dir[1]
                'min list with bench data'
                paramfit_bench_dir = np.polyfit(fitx_dir,bars['bench'][minlowdaylist],1)
                fity_bench_dir = np.polyval(paramfit_bench_dir,fitx_dir)
                predictx_bench_lowlim = np.arange(day+1,day+10,1)
                predicty_bench_lowlim = paramfit_bench_dir[0]*predictx_bench_lowlim+paramfit_bench_dir[1]
                y_bench_lowlim_prev = paramfit_bench_dir[0]*(day)+paramfit_bench_dir[1]
                y_bench_lowlim_next = paramfit_bench_dir[0]*(day+10)+paramfit_bench_dir[1]
                ax1.plot(fitx_dir,fity_bench_dir,color='m',linewidth=3.5)        
                ax1.plot(predictx_bench_lowlim,predicty_bench_lowlim,'m--',linewidth=3.5)        
                # resiy_mindir =  bars['bench'][day] - ylowlim

                'plot max list '
                fitx_max_dir = maxupdaylist
                paramfit_max_dir = np.polyfit(fitx_max_dir,bars['Close'][maxupdaylist],1)
                fity_max_dir = np.polyval(paramfit_max_dir,fitx_max_dir)
                # ax1.plot(fitx_max_dir,fity_max_dir,color='#088A08',linewidth=3.5)        
                predictx_uplim = np.arange(day+1,day+10,1)
                predicty_uplim = paramfit_max_dir[0]*predictx_uplim+paramfit_max_dir[1]
                # ax1.plot(predictx_uplim,predicty_uplim,color='#088A08',linestyle ='--',linewidth=3.5)        
                yuplim = paramfit_max_dir[0]*day+paramfit_max_dir[1]
                # resiy_maxdir =  yuplim - bars['Close'][day] 
                'max list with bench data'
                paramfit_max_bench_dir = np.polyfit(fitx_max_dir,bars['bench'][maxupdaylist],1)
                fity_max_bench_dir = np.polyval(paramfit_max_bench_dir,fitx_max_dir)
                predictx_bench_uplim = np.arange(day+1,day+10,1)
                predicty_bench_uplim = paramfit_max_bench_dir[0]*predictx_bench_uplim+paramfit_max_bench_dir[1]
                yup_bench_lim = paramfit_max_bench_dir[0]*day+paramfit_max_bench_dir[1]
                yup_bench_lim_prev = paramfit_max_bench_dir[0]*(day)+paramfit_max_bench_dir[1]
                yup_bench_lim_next = paramfit_max_bench_dir[0]*(day+10)+paramfit_max_bench_dir[1]
                ax1.plot(fitx_max_dir,fity_max_bench_dir,color='#088A08',linewidth=3.5)        
                ax1.plot(predictx_bench_uplim,predicty_bench_uplim,color='#088A08',linestyle ='--',linewidth=3.5)        

                ' find solve '
                solvey1 = lambda x : paramfit_bench_dir[0]*x+paramfit_bench_dir[1]
                solvey2 = lambda x : paramfit_max_bench_dir[0]*x+paramfit_max_bench_dir[1]
                solveresult = findIntersection(solvey1,solvey2,0.0)
                solv_intsec_MinMaxDir = solveresult
                solv_intsec_MinMaxDir_Val = solvey1(solveresult)
                # print 'MaxDir-MinDir',paramfit_max_bench_dir[0]*day+paramfit_max_bench_dir[1] - (paramfit_bench_dir[0]*day+paramfit_bench_dir[1])
                # print '5 Day Max-Min',bars['bench'][day-5:day+1].max() - bars['bench'][day-5:day+1].min()
                mintmpday = min(minlowdaylist)
                mintmpday2 = min(maxupdaylist)
                minstartday = 0
                if mintmpday2 < mintmpday:
                    minstartday = mintmpday2
                else:
                    minstartday = mintmpday
                startdiff_maxmin = (paramfit_max_bench_dir[0]*minstartday+paramfit_max_bench_dir[1]) \
                                     - (paramfit_bench_dir[0]*minstartday+paramfit_bench_dir[1])

                enddiff_maxmin = (paramfit_max_bench_dir[0]*day+paramfit_max_bench_dir[1]) \
                                     - (paramfit_bench_dir[0]*day+paramfit_bench_dir[1])
                print '20 days min',paramfit_bench_dir[0]*(day+20)+paramfit_bench_dir[1],'20days max',paramfit_max_bench_dir[0]*(day+20)+paramfit_max_bench_dir[1]
                # print 'minlowdaylist',minlowdaylist,'startdiff_maxmin',startdiff_maxmin,'enddiff_maxmin',enddiff_maxmin,'minstartday',minstartday\
                #         ,'startMaxbench',(paramfit_max_bench_dir[0]*minstartday+paramfit_max_bench_dir[1])\
                #         ,'startMinbench',(paramfit_bench_dir[0]*minstartday+paramfit_bench_dir[1])
                # print 'solveresult',solveresult,'value',solvey1(solveresult)
                # print 'paramfit_bench_dir[0]',paramfit_bench_dir[0],paramfit_bench_dir[1],paramfit_max_bench_dir[0],paramfit_max_bench_dir[1]
                # figaxecnt = 0
                # disfig = 0
                # for i in plt.get_fignums():
                #     if figaxecnt == 1:
                #         disfig = plt.figure(i)
                #         break
                #     figaxecnt += 1
                # disfig.axes[0].axhline(y = solvey1(solveresult), xmin=0.4, xmax = 0.915, linewidth=2, color='m',linestyle='--')    
                    # plt.figure(i)
                    # print 'plt.figure(i)',plt.figure(i)
                    # print 'axes',plt.figure(i).axes
            #     print 'today min predict',bars['Close'][day] - ylowlim,'resiy_mindir',resiy_mindir,'resiy_maxdir',resiy_maxdir,'max up slope',paramfit_max_dir[0]\
            #             ,'min low slope',paramfit_dir[0]
            # print 'extra paramfit[0]',paramfit_predict[0]
                
                global extrapolation_Startday
                for tmpdaycnt in range(day-1,0,-1):
                    minSlope = findMinSlope(bars,tmpdaycnt)
                    # print 'minSlope',minSlope,'paramfit_dir[0]',paramfit_dir[0]
                    if (minSlope > 0 and paramfit_dir[0] <= 0) or (minSlope <= 0 and paramfit_dir[0] > 0):
                        break

                # if f_extrapolation_prev != f_extrapolation:    
                if tmpdaycnt < 10:
                    tmpdaycnt = 10
                extrapolation_Startday = tmpdaycnt - 10
                # print 'extrapolation_Startday',extrapolation_Startday

                extrapolation_y = bars['Close'][extrapolation_Startday:day+1].values
                extrapolation_bench_y = bars['bench'][extrapolation_Startday:day+1].values
                # extra_x = np.arange(0, extrapolation_y.size)
                extra_x = np.arange(extrapolation_Startday,day+1,1)
                n_predict = 15
                extrapolation_fft = fourierExtrapolation(extrapolation_y, n_predict)
                extrapolation_bench_fft = fourierExtrapolation(extrapolation_bench_y, n_predict)
                # print 'extrapolation_Startday',extrapolation_Startday
                # print 'extrapolation_fft',extrapolation_fft
                
                # fig1 = plt.figure(figsize=(15, 2))
                # ax1.plot(extra_x,extrapolation_y,'r-')
                ax1.plot(extra_x,extrapolation_bench_y,'r-')
                extra_x = np.arange(extrapolation_Startday, day+1+n_predict,1)
                # print 'len(extra_x)',len(extra_x),'len(extrapolation_fft.size)',extrapolation_fft.size
                # ax1.plot(extra_x,extrapolation_fft,'g--')   
                ax1.plot(extra_x,extrapolation_bench_fft,'g--')   
                extra_x = np.arange(0, day+1+n_predict,1)
                extrapolation_y = bars['bench'][:day+1+15].values     
                try:
                    ax1.plot(extra_x[day:],extrapolation_y[day:],'b-')        
                    
                except Exception,e:
                    print 'plot extrapolation_y error ',e
                       
                fitx_predict = np.arange(vertex,day+1+15,1.0)
                if extrapolation_Startday == 0:
                    extrapolation_Startday = 1
                print 'extrapolation_Startday',extrapolation_Startday,'vertex', vertex
                if extrapolation_Startday >= vertex:
                    # print bars['Close'][vertex:extrapolation_Startday].values.tolist()
                    extrapolation_fft = bars['Close'][vertex:extrapolation_Startday].values.tolist() + extrapolation_fft.tolist()
                    extrapolation_bench_fft = bars['bench'][vertex:extrapolation_Startday].values.tolist() + extrapolation_bench_fft.tolist()
                else:
                    extrapolation_fft = extrapolation_fft[(vertex-extrapolation_Startday):]
                    extrapolation_bench_fft = extrapolation_bench_fft[(vertex-extrapolation_Startday):]
                # print 'extrapolation_fft',extrapolation_fft,'vertex:day+1',bars['Close'][vertex:day+1]
                # print len(fitx_predict),len(extrapolation_fft)
                
                # paramfit_bench_predict = np.polyfit(fitx_predict,extrapolation_bench_fft,1)
                # paramfit_predict = np.polyfit(fitx_predict,extrapolation_fft,1)
                # fity12_predict = np.polyval(paramfit_predict,fitx_predict)
                # ax1.plot(fitx_predict,fity12_predict,'-',linewidth=2.5)
                fitx_predict = np.arange(extrapolation_Startday,day+1,1.0)
                paramfit_bench_predict = np.polyfit(fitx_predict,bars['bench'][extrapolation_Startday:day+1],1)
                fity12_predict = np.polyval(paramfit_bench_predict,fitx_predict)
                # ax1.plot(fitx_predict,fity12_predict,'-',linewidth=2.5)
                fity12_predict_min = paramfit_bench_predict[0]*fitx_predict+ paramfit_bench_predict[1]-(0.05)
                # ax1.plot(fitx_predict,fity12_predict_min,'-',linewidth=2.5,color='r')


                if day > 70:
                    ax1.set_xlim(xmin = day-70,xmax = day+15)
                else:
                    ax1.set_xlim(xmin = 0,xmax = day+15)

                figaxecnt = 0
                disfig = 0
                for i in plt.get_fignums():
                    if figaxecnt == 1:
                        disfig = plt.figure(i)
                        break
                    figaxecnt += 1
                

                # disfig.axes[0].plot(predictx_bench_lowlim,predicty_bench_lowlim,'m--',linewidth=3.5)        
                # disfig.axes[0].plot(predictx_bench_uplim,predicty_bench_uplim,color='#088A08',linestyle ='--',linewidth=3.5)            
                disfig.axes[0].plot(fitx_predict,fity12_predict,'k-',linewidth=2.5)
                """  ---------------- hmm min, max dir setup vertex fix ----------"""                
                
                """  --------------- plot setup vertex residual  ----------------------"""                                                 
                ax1 = fig1.add_subplot(gs[1],  ylabel='min regresion residual')
                """  --------------- plot setup vertex residual  ----------------------"""                                                     

                global f_setupvertex
                global g_setup_mindir
                global g_setup_maxdir
                                
                if f_setupvertex == True:
                    g_setup_mindir = paramfit_bench_dir
                    g_setup_maxdir = paramfit_max_bench_dir
                    f_setupvertex = False
                if type(g_setup_mindir) is np.ndarray:
                    print 'g_setup_mindir[0]',g_setup_mindir[0],'g_setup_maxdir[0]',g_setup_maxdir[0],'setupvertex',setupvertex
                    buytestrange = np.arange(setupvertex,setupvertex+30,1)
                    buypredicty_lowlim = g_setup_mindir[0]*buytestrange+g_setup_mindir[1]
                    buypredicty_uplim = g_setup_maxdir[0]*buytestrange+g_setup_maxdir[1]
                    disfig.axes[1].plot(buytestrange,buypredicty_lowlim,'m--',linewidth=3.5)        
                    disfig.axes[1].plot(buytestrange,buypredicty_uplim,color='#088A08',linestyle ='--',linewidth=3.5)                                
                    if len(bars[:setupvertex+30])< len(bars):
                        disfig.axes[1].plot(buytestrange,bars['bench'][setupvertex:setupvertex+30])
                    elif len(bars[:setupvertex+30])>= len(bars):
                        disfig.axes[1].plot(buytestrange,bars['bench'][setupvertex:len(bars)])

                    """  --------------- plot setup vertex residual  ----------------------"""                                                 
                    extrabuypredict = paramfit_bench_predict[0]*buytestrange+paramfit_bench_predict[1]
                    resiy = bars['bench'][setupvertex:setupvertex+30] - extrabuypredict
                    ax1.plot(buytestrange,resiy,'ro')
                    ax1.axhline(y = 0, xmin=0.0, xmax = 1.0, linewidth=2, color='k')
                    """  --------------- plot setup vertex residual  ----------------------"""                                                     
                """  ---------------- hmm min, max dir fix ----------"""


                'find second vertex'
                tmpoffset = 0
                if day - (vertex+ tmpoffset) >= 1:
                    tmpsolveSecondVertex = []
                    tmpsolveSecondVertexY = []
                    solvedic = {}
                    for tmpdaycnt in range(vertex+tmpoffset,day+1,1):
                        minparamfit = findMinBenchSlope(bars,tmpdaycnt)
                        maxparamfit = findMaxBenchSlope(bars,tmpdaycnt)
                        # print 'minparamfit',minparamfit[0],minparamfit[1],'maxparamfit',maxparamfit[0],maxparamfit[1]
                        # if minparamfit[0] > 0.0 :
                        ' find solve '
                        solvey1 = lambda x : minparamfit[0]*x+minparamfit[1]
                        solvey2 = lambda x : maxparamfit[0]*x+maxparamfit[1]
                        solveresult = findIntersection(solvey1,solvey2,0.0)
                        tmpsolv_intsec_MinMaxDir = solveresult[0]
                        tmpsolv_intsec_MinMaxDir_Val = solvey1(solveresult[0])
                        # print 'tmpsolv_intsec_MinMaxDir',tmpsolv_intsec_MinMaxDir,tmpsolv_intsec_MinMaxDir_Val
                        secondVlimx =  [0,0.06,0.14]
                        secondVlimy = [100,70,50]
                        secondlimfl = sp.interp1d(secondVlimx, secondVlimy,kind='linear',bounds_error=False)
                        if stdvarpct_mean <= 0.0:
                            secondlim = 80
                        elif stdvarpct_mean >= 0.14:    
                            secondlim = 30
                        else:
                            secondlim = secondlimfl(stdvarpct_mean)           
                        
                        if solveresult > tmpdaycnt-30 and solveresult < tmpdaycnt + secondlim:
                            tmpsolveSecondVertex.append(tmpsolv_intsec_MinMaxDir)
                            tmpsolveSecondVertexY.append(tmpsolv_intsec_MinMaxDir_Val)
                            solvedic[tmpsolv_intsec_MinMaxDir] = tmpsolv_intsec_MinMaxDir_Val
                    if len(tmpsolveSecondVertex) > 0:
                        tmpsortkey = sorted(solvedic)
                        if tmpsortkey[0] < day - 10:
                            del solvedic[tmpsortkey[0]]
                    if len(tmpsolveSecondVertex) > 0:
                        # print 'solve second vertex,',tmpsolveSecondVertex,tmpsolveSecondVertexY
                        tmpsortkey = sorted(solvedic)
                        min_maxlim_y2 = 0
                        min_maxlim_day = 0
                        for cnt in range(0,len(solvedic)):
                            if maxlim_y2 < solvedic[tmpsortkey[cnt]]:
                                if min_maxlim_y2 != 0 and min_maxlim_y2 > solvedic[tmpsortkey[cnt]]:
                                    min_maxlim_y2 = solvedic[tmpsortkey[cnt]]
                                    min_maxlim_day = tmpsortkey[cnt]
                                if min_maxlim_y2 == 0:
                                    min_maxlim_y2 = solvedic[tmpsortkey[cnt]]
                                    min_maxlim_day = tmpsortkey[cnt]

                                # print 'solve over maxlim_y2 ',tmpsortkey[cnt],solvedic[tmpsortkey[cnt]],        
                        if min_maxlim_y2 != 0:
                            secondvertextmp = min_maxlim_day
                            secondvertexval = min_maxlim_y2
                        else:
                            secondvertextmp = tmpsortkey[0]
                            secondvertexval = solvedic[tmpsortkey[0]]
                        
                        secondMeanday = secondvertextmp
                        if day < secondvertextmp:
                            meansum = 0
                            meancnt = 0
                            for tmpday in minlowdaylist:
                                if tmpday > vertex:
                                    meansum += tmpday
                                    meancnt += 1
                            if meancnt >= 1:
                                secondMeanday = meansum/meancnt
                                secondvertextmp = secondMeanday         
                            # print 'secondMeanday',secondMeanday,
                                
                        # print 'solve key',secondvertextmp,'value',secondvertexval
                        figaxecnt = 0
                        disfig = 0
                        for i in plt.get_fignums():
                            if figaxecnt == 1:
                                disfig = plt.figure(i)
                                break
                            figaxecnt += 1
                        t1_secondvertex = np.arange(day-10, day+10, 1.0) 
                        y1_secondvertex = coef*(t1_secondvertex-secondvertextmp)**2+\
                                        secondvertexval
                        # disfig.axes[0].plot(t1_secondvertex,y1_secondvertex,'b',linewidth=2)  
                        # disfig.axes[0].plot(secondvertextmp,secondvertexval,'o')  

                        
                        # if bars['bench'][day] < coef*(day-secondvertextmp)**2+ secondvertexval:
                        #     disfig.axes[0].axvline(day, ymin=0.0, ymax = 0.315, linewidth=2, color='b',linestyle='--')

                # 'find extrapolation minslope vertex'
                # ' find solve '
                # solvey1 = lambda x : paramfit_bench_dir[0]*x+paramfit_bench_dir[1]
                # solvey2 = lambda x : paramfit_bench_predict[0]*x+paramfit_bench_predict[1]
                # solveresult = findIntersection(solvey1,solvey2,0.0)
                # intersect_extraMinslope = solveresult[0]
                # intersect_extraMinslope_Val = solvey1(solveresult[0])             
                # print 'intersect_extraMinslope',intersect_extraMinslope,'intersect_extraMinslope_Val',intersect_extraMinslope_Val

            curstd = bars['bench'][lasteDateDay:day+1].std()    
            # print 'curren range std',curstd,'stdvarpct_mean',stdvarpct_mean,curstd <= stdvarpct_mean ,'pct ',curstd /stdvarpct_mean       
            
            

            ''' bench select slope cal '''
            if 40 < day:

                

                setbenchSelectDay = 0
                # for tmpday in range(vertex,day+1,1):
                #     if bars['benchselect'][tmpday] == 0:
                #         if (setbenchSelectDay != 0 and tmpday - setbenchSelectDay > 4) or setbenchSelectDay == 0:
                #             setbenchSelectDay = tmpday 
                            # print 'setbenchSelectDay',setbenchSelectDay,'tmpday',tmpday,'benchselect',bars['benchselect'][tmpday]
                tmprisingSelectdays = []                                        
                tmpSelectdays = []                            
                for tmpday in range(day-40,day+1,1):
                    if bars['benchselect'][tmpday] == 0:
                        tmpSelectdays.append(tmpday)
                    if bars['benchselect'][tmpday] == 1:
                        tmprisingSelectdays.append(tmpday)
                # print 'tmpSelectdays',tmpSelectdays,np.mean(tmpSelectdays)
                # print 'mean bars',bars['bench'][tmpSelectdays].mean()
                tmpSelectMean = 0
                if len(tmpSelectdays) > 0:
                    setbenchSelectDay = int(np.mean(tmpSelectdays))
                    # tmpSelectMean = bars['bench'][tmpSelectdays].mean()

                else:
                    setbenchSelectDay = tmpminday
                # print 'solv_intsec_MinMaxDir',solv_intsec_MinMaxDir[0]
                # if day-100 < solv_intsec_MinMaxDir[0] < day-40:
                #     setbenchSelectDay = int(solv_intsec_MinMaxDir[0])
                # else:    
                #     setbenchSelectDay = tmpminday
                    # tmpSelectMean = bars['bench'][day]
                # print 'setbenchSelectDay',setbenchSelectDay,'solv_intsec_MinMaxDir',solv_intsec_MinMaxDir[0]
                # if len(tmprisingSelectdays) > 0:
                #     tmpSelectMean = bars['bench'][tmprisingSelectdays].mean()                    
                # else:
                #     tmpSelectMean = bars['bench'][day]
                setBenchSlopeX = np.arange(setbenchSelectDay,day+1,1.0)
                param_setBenchSlope = np.polyfit(setBenchSlopeX,bars['bench'][setbenchSelectDay:day+1],1)
                setBenchSlopeY = np.polyval(param_setBenchSlope,setBenchSlopeX)
                figaxecnt = 0
                disfig = 0
                for i in plt.get_fignums():
                    if figaxecnt == 2:
                        disfig = plt.figure(i)
                        break
                    figaxecnt += 1
                
                disfig.axes[0].plot(setBenchSlopeX,setBenchSlopeY,'r-',linewidth=2.5)  
                
                # ax1.plot(setBenchSlopeX,fity12_predict,'-',linewidth=2.5)

                # print 'min mean',bars['bench'][minlowdaylist].mean(),'max mean',bars['bench'][maxupdaylist].mean(), bars['bench'][day] >bars['bench'][maxupdaylist].mean()
                # ''' extrapolationSlope residual calc  '''
                # extraResiBelow = []
                # extraResiAbove = []
                # for tmpday in range(setbenchSelectDay,day+1,1):
                #     tmpextra = param_setBenchSlope[0]*tmpday + param_setBenchSlope[1]
                #     if bars['bench'][tmpday] > tmpextra:
                #         extraResiAbove.append(bars['bench'][tmpday])
                #     else:
                #         extraResiBelow.append(bars['bench'][tmpday])
                # extraAboveMean = np.mean(extraResiAbove)
                # extraBelowMean = np.mean(extraResiBelow)
                # # extraStd = bars['bench'][extrapolation_Startday:day+1].std()
                # print 'extraAboveMean',extraAboveMean,'extraBelowMean',extraBelowMean
                # # ,bars['bench'][day] - (paramfit_bench_predict[0]*day + paramfit_bench_predict[1])\
                # #     ,'extraStd',extraStd,( extraAboveMean > extraStd or extraBelowMean > extraStd)
                # ''' extrapolationSlope residual calc  '''        

                findflag = 'none'
                if day - setbenchSelectDay <= 3:
                    findflag = 'skip'

                if findflag == 'none' and paramfit_bench_predict[0] > 0.0 and paramfit_bench_dir[0] > 0.0\
                    and (paramfit_bench_predict[0]*extrapolation_Startday+paramfit_bench_predict[1] < param_setBenchSlope[0]*setbenchSelectDay +param_setBenchSlope[1]):
                    
                    if bars['bench'][day] < paramfit_bench_predict[0]*day+paramfit_bench_predict[1] \
                        and ( day-200 < solv_intsec_MinMaxDir[0] < day-40 and solv_intsec_MinMaxDir[0] > 0) :
                        # print 'param_setBenchSlope below Stop'
                        print 'paramfit_bench_predict[0]*day+paramfit_bench_predict[1]',paramfit_bench_predict[0]*day+paramfit_bench_predict[1],bars['bench'][day]
                        print 'no signal'
                    else:
                        if paramfit_bench_predict[0] < paramfit_bench_dir[0] or param_setBenchSlope[0] >paramfit_bench_predict[0] \
                            or (paramfit_max_bench_dir[0] > 0.0  and paramfit_max_bench_dir[0] < paramfit_bench_predict[0]):
                            print 'findflag',findflag,'setbenchSelectDay',setbenchSelectDay,'param_setBenchSlope',param_setBenchSlope[0],\
                            'extrapolationSlope',paramfit_bench_predict[0],'True'
                        else:
                            print 'findflag',findflag,'setbenchSelectDay',setbenchSelectDay,'param_setBenchSlope',param_setBenchSlope[0],\
                            'extrapolationSlope',paramfit_bench_predict[0],'No Signal'
                    
                    # print 'tmpSelectMean',tmpSelectMean,\
                    #  'today Bench',bars['bench'][day],tmpSelectMean < bars['bench'][day]
                elif findflag == 'skip' or ( paramfit_bench_predict[0] < 0.0) or paramfit_bench_dir[0] < 0.0:
                    print 'findflag',findflag,'setbenchSelectDay',setbenchSelectDay,'param_setBenchSlope',param_setBenchSlope[0],'extrapolationSlope',paramfit_bench_predict[0],'False'
                else:
                    print 'No Signal'



                ''' bench select slope cal '''

                ''' min slope from extrapolation_Startday '''
            if 40 < day:
                extraMinflag = 'none'
                extraMinDayarrs = []
                extraMinVararrs = []
                extraMinDay = 0
                extraMinVar = 0

                extraMaxflag = 'none'
                extraMaxDayarrs = []
                extraMaxVararrs = []
                extraMaxDay = 0
                extraMaxVar = 0

                for tmpday in range(extrapolation_Startday,day+1,1):
                    slopeY = paramfit_bench_predict[0]*tmpday + paramfit_bench_predict[1]
                    # print 'slopeY',slopeY,'bench',bars['bench'][tmpday]
                    if bars['bench'][tmpday] < slopeY:
                        if extraMinflag == 'progress':
                            if slopeY - bars['bench'][tmpday] > extraMinVar:
                                extraMinDay = tmpday
                                extraMinVar = slopeY - bars['bench'][tmpday]    
                                extraMinDayarrs.pop(-1)
                                extraMinDayarrs.append(tmpday)
                                extraMinVararrs.append(extraMinVar)
                                # print 'progress extraMinVar',extraMinVar,tmpday
                        if extraMinflag == 'none':
                            extraMinflag = 'progress'
                            extraMinDay = tmpday
                            extraMinVar = slopeY - bars['bench'][tmpday]
                            extraMinDayarrs.append(tmpday)
                            extraMinVararrs.append(extraMinVar)
                            # print 'none extraMinVar',extraMinVar,tmpday
                    elif bars['bench'][tmpday] > slopeY:
                        if extraMinflag == 'progress':
                            extraMinflag = 'none'
                            extraMinDay = 0
                            extraMinVar = 0

                    if bars['bench'][tmpday] > slopeY:
                        if extraMaxflag == 'progress':
                            if  bars['bench'][tmpday] - slopeY > extraMaxVar:
                                extraMaxDay = tmpday
                                extraMaxVar = bars['bench'][tmpday]-  slopeY 
                                extraMaxDayarrs.pop(-1)
                                extraMaxDayarrs.append(tmpday)
                                extraMaxVararrs.append(extraMaxVar)
                                # print 'progress extraMaxVar',extraMaxVar,tmpday
                        if extraMaxflag == 'none':
                            extraMaxflag = 'progress'
                            extraMaxDay = tmpday
                            extraMaxVar = bars['bench'][tmpday] - slopeY
                            extraMaxDayarrs.append(tmpday)
                            extraMaxVararrs.append(extraMaxVar)
                            
                            # print 'none extraMaxVar',extraMaxVar,tmpday
                    elif bars['bench'][tmpday] < slopeY:
                        if extraMaxflag == 'progress':
                            extraMaxflag = 'none'
                            extraMaxDay = 0
                            extraMaxVar = 0                
                # print  'extraMinDayarrs',extraMinDayarrs ,'extraMinVararrs',extraMinVararrs
                MinSlopeX = extraMinDayarrs
                param_MinSlope = np.polyfit(MinSlopeX,bars['bench'][MinSlopeX],1)
                MinSlopeY = np.polyval(param_MinSlope,MinSlopeX)

                MaxSlopeX = extraMaxDayarrs
                param_MaxSlope = np.polyfit(MaxSlopeX,bars['bench'][MaxSlopeX],1)
                MaxSlopeY = np.polyval(param_MaxSlope,MaxSlopeX)
                figaxecnt = 0
                disfig = 0
                for i in plt.get_fignums():
                    if figaxecnt == 1:
                        disfig = plt.figure(i)
                        break
                    figaxecnt += 1
                
                disfig.axes[0].plot(MinSlopeX,MinSlopeY,'m-',linewidth=2.5)  
                disfig.axes[0].plot(MaxSlopeX,MaxSlopeY,'m-',linewidth=2.5)  
                if param_MinSlope[0] > 0.0 :
                    print 'Min Slope from extrapolation_Startday',bars['bench'][day] < param_MinSlope[0]*day+param_MinSlope[1],'(False is Buy signal)'
                else:
                    print 'Min Slope from extrapolation_Startday True'
                ''' min slope from extrapolation_Startday '''

                ''' simul min slope with bench select signal '''
                tradeSimul(extrapolation_Startday,day,bars,tmpminday)
                

                ''' simul min slope with bench select signal '''
            """  --------------- plot extrapolation_y  ----------------------"""                                                 
            
            


            """  --------------- plot min regresion residual  ----------------------"""                                                 
            
            # ax1 = fig1.add_subplot(gs[1],  ylabel='min regresion residual')
            # resiy =  bars['bench'][day-10:day+10] - fity_min
            # ax1.plot(fitx_min, resiy, 'ro')
            # ax1.axhline(y = 0, xmin=0.0, xmax = 1.0, linewidth=2, color='k')
            # ax1.axhline(y = -1.0*stdvarpct_mean, xmin=0.0, xmax = 1.0, linewidth=2, color='r')

            """  --------------- plot min regresion residual  ----------------------"""                                                 
            
            

            """  --------------- plot predict min, sameStateMin ,sameStateMax  ----------------------"""                                                 
            
            # """
            """ predict min, sameStateMin ,sameStateMax   """
            barsClosetmp = bars['Close'][:day].values.tolist()
            # print type(extrapolation_fft)
            # print 'barsClosetmp',barsClosetmp,'extrapolation_fft[-15:]',extrapolation_fft[-15:]
            # print 'type extrapolation_fft',type(extrapolation_fft)
            if type(extrapolation_fft) is list:
                barsClosetmp = barsClosetmp+extrapolation_fft[-15:]
            else:
                barsClosetmp = barsClosetmp+extrapolation_fft[-15:].tolist()

            # barsClosetmp = barsClosetmp+extrapolation_fft[day:day+1+15].tolist()
            tmpdf = pd.DataFrame({'Close':barsClosetmp})
            tmpdf['bench'] = tmpdf['Close'].pct_change().cumsum()
            maxbench = tmpdf['bench'][day:day+1+15].max()
            minbench = tmpdf['bench'][day:day+1+15].min()
            # print tmpdf['bench'][day:day+1+15]
            sameStateMin = bars['bench'][day-sameStatecnt+1:day+1].min()
            sameStateMax = bars['bench'][day-sameStatecnt+1:day+1].max()
            # print 'sameStatecnt min',sameStateMin,'max',sameStateMax,'bench today',bars['bench'][day],'maxlim_y2',maxlim_y2
            # print 'predict min',minbench,'max',maxbench
            tmppredictdf = pd.DataFrame({'Date':[bars.index[day]],'predict Min':[minbench],'predict Max':[maxbench],'same Min':[sameStateMin],'same Max':[sameStateMax]})
            predictdf = pd.concat([predictdf,tmppredictdf])
            predictdf = predictdf.sort(['Date'],ascending=True)
            predictdf = predictdf.drop_duplicates(cols='Date', take_last=True)
            # predictdf = predictdf.set_index('Date')
            # print predictdf
            daycnt = 0
            for tmpDate in predictdf['Date']:
                if tmpDate == bars.index[day]:
                    # print 'tmpDate stop',tmpDate
                    break
                daycnt+=1
            
            # if minbench > sameStateMin:
            #     ax1.axhline(y = -1.0*(abs(bars['bench'][day] - minbench)), xmin=0.0, xmax = 1.0, linewidth=2, color='r')
            # else:
            #     ax1.axhline(y = -1.0*(abs(bars['bench'][day] - sameStateMin)), xmin=0.0, xmax = 1.0, linewidth=2, color='r')
            ax1.axhline(y = (minbench+ sameStateMin)*0.5, xmin=0.0, xmax = 1.0, linewidth=2, color='r')
            

            
            showPredictMinMaxSameStateMinMax_hmmrealSignal(predictdf,day,daycnt)
            plt.clf()

            plt.show()
            """  --------------- plot predict min, sameStateMin ,sameStateMax  ----------------------"""
            
            """  --------------- data result summary ------------- """
            """ 'sDate','eDate','subtstate','days' """
            tmpdf = selrange[['sDate','eDate','subtstate','days']]
            """  disbars = bars[['bench']][day-5:day+1] """
            # disbars = bars[['bench']][day-5:day+1]
            # bars[['bench']][day-5:day+1]
            """  stdvarpct_mean """        
            """  coef,coeflowlim """
            """  'polyfit range slope param[0]',paramfit[0],'paramfit_samestate',paramfit_samestate[0],'min slope',paramfit_min[0] """
            """  'extra paramfit[0]',paramfit_predict[0] """
            """  'today min predict',bars['Close'][day] - ylowlim,'resiy_mindir',resiy_mindir,'resiy_maxdir',resiy_maxdir
                    ,'max up slope',paramfit_max_dir[0],'min low slope',paramfit_dir[0]"""
            """  'sameStatecnt min',sameStateMin,'max',sameStateMax,'bench today',bars['bench'][day],'maxlim_y2',maxlim_y2
                 'predict min',minbench,'max',maxbench """
                 
            try: 
                columnsix = ['day','subtstate','days','coef','coeflowlim','rangeSlope','samStateSlope','minSlope','extrapolationSlope'\
                            ,'miDirSlope','maxDirSlope','y_bench_lowlim_prev','y_bench_lowlim_next','yup_bench_lim_prev','yup_bench_lim_next'\
                            ,'benchToday','maxlim_y2'\
                            ,'sameStateMin','sameStateMax','predictMin','predictMax','stdvarpct_mean'\
                            ,'rangeLastDays','rangeGain','solveKey','solveVal']
                summarydf = pd.DataFrame({'day':[day],'subtstate':tmpdf['subtstate'],'days':tmpdf['days'],'coef':[coef],'coeflowlim':[coeflowlim]\
                            ,'rangeSlope':[paramfit[0]],'samStateSlope':[paramfit_samestate[0]],'minSlope':[paramfit_min[0]]\
                            ,'extrapolationSlope':[paramfit_bench_predict[0]],'miDirSlope':[paramfit_bench_dir[0]],'maxDirSlope':[paramfit_max_bench_dir[0]]\
                            ,'y_bench_lowlim_prev':[y_bench_lowlim_prev],'y_bench_lowlim_next':[y_bench_lowlim_next]\
                            ,'yup_bench_lim_prev':[yup_bench_lim_prev],'yup_bench_lim_next':[yup_bench_lim_next]\
                            ,'sameStateMin':[sameStateMin]\
                            ,'sameStateMax':[sameStateMax],'benchToday':bars['bench'][day],'maxlim_y2':[maxlim_y2]\
                            ,'predictMin':[minbench],'predictMax':[maxbench],'stdvarpct_mean':[stdvarpct_mean]\
                            ,'rangeLastDays':[day-lasteDateDay],'rangeGain':[rangegain]\
                            ,'solveKey':[secondvertextmp],'solveVal':[secondvertexval]}\
                            ,columns = columnsix)     
                display(HTML(summarydf[['day','subtstate','days','coef','coeflowlim','rangeSlope','samStateSlope','minSlope','extrapolationSlope'\
                            ,'miDirSlope','maxDirSlope','y_bench_lowlim_prev','y_bench_lowlim_next','yup_bench_lim_prev','yup_bench_lim_next']].to_html()))
                display(HTML(summarydf[['benchToday','maxlim_y2','sameStateMin','sameStateMax','predictMin','predictMax','stdvarpct_mean'\
                                        ,'rangeLastDays','rangeGain','solveKey','solveVal']].to_html()))
            except Exception,e:

                columnsix = ['day','subtstate','days','coef','coeflowlim','rangeSlope','samStateSlope','minSlope','extrapolationSlope'\
                            ,'sameStateMin','sameStateMax'\
                            ,'benchToday','maxlim_y2','predictMin','predictMax','stdvarpct_mean'\
                            ,'rangeLastDays','rangeGain']
                summarydf = pd.DataFrame({'day':[day],'subtstate':tmpdf['subtstate'],'days':tmpdf['days'],'coef':[coef],'coeflowlim':[coeflowlim]\
                            ,'rangeSlope':[paramfit[0]],'samStateSlope':[paramfit_samestate[0]],'minSlope':[paramfit_min[0]]\
                            ,'extrapolationSlope':[paramfit_bench_predict[0]]\
                            ,'sameStateMin':[sameStateMin]\
                            ,'sameStateMax':[sameStateMax],'benchToday':bars['bench'][day],'maxlim_y2':[maxlim_y2]\
                            ,'predictMin':[minbench],'predictMax':[maxbench],'stdvarpct_mean':[stdvarpct_mean]\
                            ,'rangeLastDays':[day-lasteDateDay],'rangeGain':[rangegain]}\
                            ,columns = columnsix)     
                display(HTML(summarydf[['day','subtstate','days','coef','coeflowlim','rangeSlope','samStateSlope','minSlope','extrapolationSlope'\
                                ,'benchToday','maxlim_y2']].to_html()))
                display(HTML(summarydf[['sameStateMin','sameStateMax','predictMin','predictMax','stdvarpct_mean'\
                                        ,'rangeLastDays','rangeGain']].to_html()))
            try:
                print 'solv_intsec_Y1',solv_intsec_Y1,'solv_intsec_Y1_Val',solv_intsec_Y1_Val,'solv_intsec_MinMaxDir',solv_intsec_MinMaxDir\
                        ,'solv_intsec_MinMaxDir_Val',solv_intsec_MinMaxDir_Val
            except Exception,e:
                print 'data summary error',e
            """  --------------- data result summary ------------- """
            print 'today ',bars.index[day]
            
            
        except Exception,e:
            PrintException()

def handle_export(widget):
    global daywindow
    global bars
    global barsMAranges
    global gfilterbars
    global predictdf    
    global gCode

    for tmpcodeday in range(gCode,len(bars)):

        daywindow = tmpcodeday
        day = daywindow   
        # disbars = bars['bench'][daywindow-20:daywindow+1]
        # disbars.plot(ax=ax1,lw=2)
        """  --------------  range found -------------------- """
        rangeFound = False
        for eDate in barsMAranges['eDate']:
            # print 'eDate',eDate,bars.Date[day],type(eDate) , type(bars.index[day])
            # neweDate = datetime.strptime(eDate, '%Y-%m-%d %H:%M:%S')
            # barsDate = datetime.strptime(bars.Date[day], '%Y-%m-%d %H:%M:%S')
           
            # print neweDate,barsDate
            # print eDate,bars.index[day]

            if eDate > bars.index[day]:
                # print 'range found!!',bars.index[day]
                selrange = barsMAranges[barsMAranges['eDate'] == eDate]
                # print 'selrange.index:',selrange.index[0]
                selindex = selrange.index[0]
                # selrange = selrange.reset_index()
                # print 'selrange.index ',selrange.index
               
                rangeFound = True

                break

       

        for nday in range(len(bars)):
            if eDate == bars.index[nday]:
                lasteDateDay = nday
                break       
       

        if rangeFound == True:
                           
            scut = selrange['scut'][selindex]
            ecut = selrange['ecut'][selindex]
            esbench = selrange['eBench'][selindex] - selrange['sBench'][selindex]
            # print 'esbench',esbench
           
            if scut == -1 and esbench > 0.0 :
                subtstate = 'low_rising'
                barsMAranges['subtstate'][selindex] = subtstate
                
                for tmpday in range(len(bars)):
                    if barsMAranges['eDate'][selindex] == bars.index[tmpday]:
                        break
                gCode = tmpday
                print 'low_rising',barsMAranges['eDate'][selindex],gCode
                break
            if scut == 1 and esbench > 0.0 :
                subtstate = 'high_rising'
                barsMAranges['subtstate'][selindex] = subtstate
                for tmpday in range(len(bars)):
                    if barsMAranges['eDate'][selindex] == bars.index[tmpday]:
                        break
                gCode = tmpday
                print 'high_rising',barsMAranges['eDate'][selindex],gCode
                break
            if scut == -1 and esbench < 0.0:   
                subtstate = 'low_falling'
                barsMAranges['subtstate'][selindex] = subtstate
                continue
            if scut == 1 and esbench < 0.0 :   
                subtstate = 'high_falling'   
                barsMAranges['subtstate'][selindex] = subtstate   
                continue

export_button = widgets.ButtonWidget(description="NextRange")
export_button.on_click(handle_export)
display(export_button)

setup()

def setup_vertex(vertex=1):
    # print vertex
    global setupvertex 
    setupvertex = vertex
    show_args(daywindow)    

i2 = interact(setup_vertex,vertex=(1,daywindow-1))        
display(i2)

def handle_setupVertex(widgets):
    global setupvertex 
    global f_setupvertex
    if f_setupvertex == False:
        f_setupvertex = True
    else:
        f_setupvertex = False
    clear_output()
    # print 'f_setupvertex',f_setupvertex
    setup_vertex(setupvertex)
        
export_button = widgets.ButtonWidget(description="SetupVertex")
export_button.on_click(handle_setupVertex)
display(export_button)





# for day in range(len(bars)):
#     show_args(day)
i = interact(show_args,Code=(1,daywindow-1))

# if __name__ == "__main__":
#     setup()
