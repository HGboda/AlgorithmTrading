# %matplotlib inline
# from matplotlib import interactive
# interactive(True)

# from guiqwt.plot import CurveDialog
# from guiqwt.builder import make

# pd.set_option('display.width',500)
# import sys
# sys.path.append('../../lib/')
from stockcore import *
from tradingalgo import *
from data_mani import *
import stockcore as stcore
import tradingalgo as talgo 
import data_mani as dmani

from IPython.display import clear_output, display, HTML

global plot_bars
global plot_returns
global plot_returns2
global plot_maxstddf
global plot_minstddf
global plot_mmstddf
plot_bars =0
plot_returns = 0
plot_returns2 = 0
plot_maxstddf = 0
plot_minstddf = 0
plot_mmstddf = 0

def saveplotlyvar(bars,returns,returns2):

    global plot_bars
    global plot_returns
    global plot_returns2
    plot_bars = bars
    plot_returns = returns
    plot_returns2 = returns2
        

'''
pattern functions
'''
def caseAdjust(goodcasedf,fixcasedf,barsdf,patternAr):
    print 'case adjust'
    start = time.clock()
    targetcasedf = pd.concat([goodcasedf,fixcasedf],axis = 1)
    # print goodcasedf.tail()
    # print fixcasedf.tail()
    
    targetcasedf['targetCol'] = targetcasedf['GoodMMSignals'] != targetcasedf['MMSignals'] 
    # global gbarsdf
    targetcasedf['Gain'] = (barsdf['Close'].pct_change().cumsum()).fillna(0.0)
    
    targetcasedf['targetCol2'] = targetcasedf['targetCol']

    targetcasedf['targetCol3'] = 0
    # print targetcasedf[-150:-100]
    # global targetlen
    targetlen1 = len(targetcasedf[targetcasedf['targetCol'] == True])
    # targetlen = targetlen1
    targetlen2 = len(targetcasedf[targetcasedf['targetCol'] == False])
    totallen = len(targetcasedf)
    # print totallen,targetlen1,targetlen2,targetlen1+targetlen2
    # len(targetcasedf[targetcasedf['targetCol2'] == True]),len(targetcasedf[targetcasedf['targetCol2'] == False])
    '''
    performance upgrade 
    '''
    # lpatternAr = patternAr
    
    # for numcnt in range(len(targetcasedf)):
    #     if targetcasedf['targetCol'][numcnt] == True:
    #         gcurday = numcnt

    #         curpat = barsdf.reset_index()
    #         '''
    #         make pattern of current last 10 days 
    #         '''
    #         corrclosex1 = curpat['Close'][gcurday-9:gcurday+1].pct_change().cumsum()
    #         closex1 = pd.DataFrame(corrclosex1,columns=['ClosePc']).fillna(0.0)
    #         closex1 = closex1.reset_index()
            
    #         corropenx1 = curpat['Open'][gcurday-9:gcurday+1].pct_change().cumsum()
    #         openx1 = pd.DataFrame(corropenx1,columns=['OpenPc']).fillna(0.0)
    #         openx1 = openx1.reset_index()
            
    #         corrhighx1 = curpat['High'][gcurday-9:gcurday+1].pct_change().cumsum()
    #         highx1 = pd.DataFrame(corrhighx1,columns=['HighPc']).fillna(0.0)
    #         highx1 = highx1.reset_index()

    #         corrlowx1 = curpat['Low'][gcurday-9:gcurday+1].pct_change().cumsum()
    #         lowx1 = pd.DataFrame(corrlowx1,columns=['LowPc']).fillna(0.0)
    #         lowx1 = lowx1.reset_index()

    #         corrvolx1 = curpat['Volume'][gcurday-9:gcurday+1].pct_change().cumsum()
    #         volx1 = pd.DataFrame(corrvolx1,columns=['VolPc']).fillna(0.0)
    #         volx1 = volx1.reset_index()

    #         '''
    #         compare patterns with past pattern data
    #         '''
    #         for patternnum in range(len(patternAr)):

    #             pattern0 = lpatternAr[patternnum].patterndf.reset_index()
                
    #             pattern0_closepc = pattern0['Close'].pct_change().cumsum()   
    #             closepc0 = pd.DataFrame(pattern0_closepc,index = pattern0.index,columns=['ClosePc']).fillna(0.0)

    #             closecorr = closex1['ClosePc'].corr(closepc0['ClosePc'])
    #             if closecorr <= 0.8:
    #                 targetcasedf['targetCol2'][numcnt] = False
    #                 continue

    #             pattern0_openpc = pattern0['Open'].pct_change().cumsum()   
    #             openpc0 = pd.DataFrame(pattern0_openpc,index = pattern0.index,columns=['OpenPc']).fillna(0.0)

                
    #             pattern0_highpc = pattern0['High'].pct_change().cumsum()   
    #             highpc0 = pd.DataFrame(pattern0_highpc,index = pattern0.index,columns=['HighPc']).fillna(0.0)

    #             pattern0_lowpc = pattern0['Low'].pct_change().cumsum()   
    #             lowpc0 = pd.DataFrame(pattern0_lowpc,index = pattern0.index,columns=['LowPc']).fillna(0.0)
                
    #             pattern0_volpc = pattern0['Volume'].pct_change().cumsum()   
    #             volpc0 = pd.DataFrame(pattern0_volpc,index = pattern0.index,columns=['VolPc']).fillna(0.0)
                
    #             # patternall = pd.concat([pattern0,openpc0,closepc0,highpc0,lowpc0,volpc0],axis=1)
                
                
    #             opencorr = openx1['OpenPc'].corr(openpc0['OpenPc'])
    #             highcorr = highx1['HighPc'].corr(highpc0['HighPc'])
    #             lowcorr = lowx1['LowPc'].corr(lowpc0['LowPc'])
    #             volcorr = volx1['VolPc'].corr(volpc0['VolPc'])

    #             corrsum = (closecorr+opencorr+highcorr+lowcorr+volcorr)/5
    #             if corrsum > 0.8 :
    #                 targetcasedf['targetCol2'][numcnt] = True
    #                 targetcasedf['targetCol3'][numcnt] = patternnum
    #                 break
    #             else:
    #                 targetcasedf['targetCol2'][numcnt] = False

                    
    
    targetlen1 = len(targetcasedf[targetcasedf['targetCol2'] == True])
    targetlen = targetlen1
    print 'targetlen:',targetlen
    elapsed = (time.clock() - start)
    print 'caseAdjust elapsed time:',elapsed
    return targetcasedf,targetlen





def findExtraMinSlope(argdf,day):
    'finding min low list '
    minlowlist = []
    backupminlowlist = []
    prevminval = 0
    for tmpday in range(len(argdf[:day+1])-40,len(argdf[:day+1]),10):
        minval = argdf[tmpday:tmpday+10].min()
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
        for val in argdf[day-40:day+1]:
            if val == tmpmin:
                break
            tmpdaycnt += 1
        minlowdaylist.append(tmpdaycnt+day-40)

    fitx_dir = minlowdaylist
    paramfit_dir = np.polyfit(fitx_dir,argdf[minlowdaylist],1)
    fity_dir = np.polyval(paramfit_dir,fitx_dir)

    return paramfit_dir[0]

def findExtrapolation(argdf,day):
    'finding min low list '
    minlowlist = []
    backupminlowlist = []
    prevminval = 0
    for inday in range(len(argdf[:day+1])-40,len(argdf[:day+1]),10):
        minval = argdf[inday:inday+10].min()
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
        for val in argdf[day-40:day+1]:
            if val == tmpmin:
                break
            tmpdaycnt += 1
        minlowdaylist.append(tmpdaycnt+day-40)
    # print 'minlowdaylist',bars['bench'][minlowdaylist]
    
    'plot min list '
    fitx_dir = minlowdaylist
    paramfit_dir = np.polyfit(fitx_dir,argdf[minlowdaylist],1)

    for tmpdaycnt in range(day-1,0,-1):
        minSlope = findExtraMinSlope(argdf,tmpdaycnt)
        if (minSlope > 0 and paramfit_dir[0] <= 0) or (minSlope <= 0 and paramfit_dir[0] > 0):
            break
    
    if tmpdaycnt < 10:
        tmpdaycnt = 10
    extrapolation_Startday = tmpdaycnt - 10

    return extrapolation_Startday

def talibtechsignal(bars):
    print 'talib tech signal'

    MA1 = 5
    MA2 = 20
    MA3 = 60

    macd, macdsignal, macdhist = talib.MACD(bars['Close'].values, 12, 26, 9)
    slowk,slowd =talib.STOCH(bars['High'].values,bars['Low'].values,bars['Close'].values,10,6,6)
    upperband,middleband,lowerband = talib.BBANDS(bars['Close'].values,10,2,2)
    obvout = talib.OBV(bars['Close'].values,bars['Volume'].values)
    rsiout = talib.RSI(bars['Close'].values,14)
    # wmaout = talib.WMA(bars['Close'],30)
    # mfiout = talib.MFI(bars['High'].values,bars['Low'].values,bars['Close'].values,bars['Volume'].values,14)
    # dojiout = talib.CDLDOJI(bars['Open'].values,bars['High'].values,bars['Low'].values,bars['Close'].values)
    # marubozuout = talib.CDLMARUBOZU(bars['Open'],bars['High'],bars['Low'],bars['Close'])
    # hammerout = talib.CDLHAMMER(bars['Open'],bars['High'],bars['Low'],bars['Close'])
    # engulfingout = talib.CDLENGULFING(bars['Open'],bars['High'],bars['Low'],bars['Close'])
    # varout = talib.VAR(bars['Close'].values,5,1)
    
    rsidf = pd.DataFrame(rsiout,index = bars.index,columns=['RSI'])
    # slowkdf = pd.DataFrame(slowk,index = bars.index,columns=['Slowk'])
    # slowddf = pd.DataFrame(slowd,index = bars.index,columns=['Slowd'])
    # upperbanddf = pd.DataFrame(upperband,index = bars.index,columns=['upperband'])
    # middlebanddf = pd.DataFrame(middleband,index = bars.index,columns=['middleband'])
    # lowerbanddf = pd.DataFrame(lowerband,index = bars.index,columns=['lowerband'])
    obvdf = pd.DataFrame(obvout,index = bars.index,columns=['OBV'])
    # mfidf = pd.DataFrame(mfiout,index = bars.index,columns=['MFI'])
    # dojidf = pd.DataFrame(dojiout,index = bars.index,columns=['DOJI'])
    # marubozudf = pd.DataFrame(marubozuout,index = bars.index,columns=['MARUBOZU'])
    # hammerdf = pd.DataFrame(hammerout,index = bars.index,columns=['HAMMER'])
    # engulfingdf = pd.DataFrame(engulfingout,index = bars.index,columns=['ENGULFING'])
    # obvMA1 = pd.rolling_mean(obvdf, MA1, min_periods=1).fillna(0.0)
    # obvMA1np = obvMA1.values

    # mfiMA1 = pd.rolling_mean(mfidf, MA1, min_periods=1).fillna(0.0)
    # mfiMA1np = mfiMA1.values
    
    # vardf = pd.DataFrame(varout,index = bars.index,columns=['VAR'])
    
    
    macddf = pd.DataFrame({'macd':macd,'signal':macdsignal},index = bars.index).fillna(0.0)
    bars['MACDSig'] = 0
    bars['MACDSig'][(macddf['macd'] >= macddf['signal']) & (macddf['macd'] != 0)] = 1

    slowdf = pd.DataFrame({'slowk':slowk,'slowd':slowd},index = bars.index).fillna(0.0)
    bars['SLOWSig'] = 0
    bars['SLOWSig'][(slowdf['slowk'] >= slowdf['slowd']) & (slowdf['slowk'] != 0)] = 1


    disbars = bars[['MACDSig','SLOWSig','benchsig','bench']]
    # display(HTML(disbars[-30:].to_html()))
    argdf = obvdf['OBV']
    argextrapolationday = findExtrapolation(argdf,len(bars)-1)
    fitx_dir = np.arange(argextrapolationday,len(bars))
    paramfit_dir = np.polyfit(fitx_dir,argdf[argextrapolationday:],1)
    print 'argextrapolationday',bars.index[argextrapolationday],'paramfit_dir',paramfit_dir[0]


    argdf = rsidf['RSI']
    argextrapolationday = findExtrapolation(argdf,len(bars)-1)
    fitx_dir = np.arange(argextrapolationday,len(bars))
    paramfit_dir = np.polyfit(fitx_dir,argdf[argextrapolationday:],1)
    print 'argextrapolationday',bars.index[argextrapolationday],'paramfit_dir',paramfit_dir[0]

    """
    fig1 = plt.figure(figsize=(20, 3))
    ax1 = fig1.add_subplot(111,  ylabel='MACDSig')
    bars['bench'].plot(ax=ax1,lw=2)
    ax1.plot(bars['bench'].ix[bars['MACDSig'] == 1].index,
             bars['bench'][bars['MACDSig'] == 1],
             '.', markersize=10, color='r')        

    fig1 = plt.figure(figsize=(20, 3))
    ax1 = fig1.add_subplot(111,  ylabel='SLOWSig')
    bars['bench'].plot(ax=ax1,lw=2)
    ax1.plot(bars['bench'].ix[bars['SLOWSig'] == 1].index,
             bars['bench'][bars['SLOWSig'] == 1],
             '.', markersize=10, color='r')        

    fig1 = plt.figure(figsize=(20, 3))
    ax1 = fig1.add_subplot(111,  ylabel='OBV')
    obvdf['OBV'].plot(ax=ax1,lw=2)

    fig1 = plt.figure(figsize=(20, 3))
    ax1 = fig1.add_subplot(111,  ylabel='RSI')
    rsidf['RSI'].plot(ax=ax1,lw=2)

    plt.show()
    """


global goodcasedf
goodcasedf = 0
global fixcasedf
fixcasedf = 0
global gbarsdf
gbarsdf = 0
global gpatternAr
gpatternAr = 0
from numpy import arange
from numpy import sin,linspace,power
from scipy import interpolate
from pylab import plot,show
def draw_tangent(x,y,a):

    
    # interpolate the data with a spline
    spl = interpolate.splrep(x,y)
    # small_t = arange(a-5,a+5)
    fa = interpolate.splev(a,spl,der=0)     # f(a)
    fprime = interpolate.splev(a,spl,der=1) # f'(a)
    # tan = fa+fprime*(small_t-a) # tangent
    # ax.plot(a,fa,'om',small_t,tan,'--r')
    return fprime

def tran_func(x,w):
    return x + np.sin(w)



def RunSimul(codearg,typearg,namearg,mode,dbmode,histmode,runcount,srcsite,writedblog,updbpattern,appenddb,startdatemode 
    ,dbtradinghist,plotly,*args):

    

    if typearg == 4:# DOW,NASDAQ,S&P500
        startdate= '2014-01-01'
        arg_index = 0
        stdarg = args[arg_index]
        arg_index += 1
        smallarg = args[arg_index]
        arg_index += 1
        dayselect = args[arg_index]
        arg_index += 1
        tangentmode = args[arg_index]
        print 'stdarg',stdarg,'tangentmode',tangentmode
        tangentmode = 'tan_gen'
        if namearg == 'dow':
            symbol = 'GOOG/INDEXDJX_DJI'
            bars =  Quandl.get(symbol, collapse='Daily', trim_start=startdate, trim_end=datetime.today(),authtoken="")
        elif namearg == 'nasdaq':
            symbol = '^IXIC'
            import pandas.io.data as web
            bars = web.get_data_yahoo(symbol,startdate)
        elif namearg == 'sandp':
            symbol = '^GSPC'
            import pandas.io.data as web
            bars = web.get_data_yahoo(symbol,startdate)

        if mode =='dbpattern'  or dbmode == 'dbpattern':        
            if updbpattern == 'none':
                print 'read DB patterns'
                patternAr, extractid= stcore.ReadPatternsFromDB(codearg,typearg,namearg,mode)
                patternAppendAr = stcore.ReadPatternsAppendFromDB(codearg,namearg)
            elif updbpattern == 'updbpattern':
                print 'read UP DB patterns'
                patternAr, extractid= stcore.ReadUpPatternsFromDB(codearg,typearg,namearg,mode)

            if patternAr == -1:
                print 'real time gen db pattern'
                startdate = '2011-01-01'
                dbmode  = 'none'


    else:    
        bars,patternAr, extractid,patternAppendAr,bars_25p,bars_50p\
        ,bars_75p,bars_90p,tangent_25p,tangent_50p,tangentmode\
        ,startdate,dbmode,stdarg,smallarg,dayselect,tangentmode =\
            stcore._inRunSimul_FetchData(codearg,typearg,namearg,mode,dbmode,histmode,runcount,srcsite,writedblog,updbpattern,appenddb,startdatemode \
                            ,dbtradinghist,plotly,*args)
        # print 'stdarg',stdarg,'smallarg',smallarg,'dayselect',dayselect,'tangentmode',tangentmode
        
    
    # bars = bars.drop(bars.index[-1])    
    
    today  = datetime.today()
    todayweek = today.weekday()

    bars['week'] = bars.index.weekday
    tailweekdays = bars['week'][-5:]
    # print tailweekdays

    # if 0 <= todayweek <=4 :
    #     for cnt in range(0,len(tailweekdays)):
    #         day = cnt +1 
    #         # print gbars2['week'][-1*day]
    #         checkday = bars['week'][-1*day]
            
    #         # print todayweek,checkday,bars.index[-1*day]
    #         if todayweek != checkday:
    #             raise Exception("week check error")
    #         if todayweek == 0:
    #             todayweek = 4
    #         else:
    #             todayweek = todayweek - 1

    # bars = bars[:len(bars)-100]  
    # bars = bars[:'2015-07-07']
    # print bars.tail()  
    '''
    test code for inflection point
    '''
    # bars = bars.drop(bars.tail(1).index)
    # print '----------final test bars---------'
    # print bars.tail()    
    '''
    end test code
    '''

    mac1 = stcore.MovingAverageCrossStrategy(
    'test', bars, short_window=5, long_window=20)
    signals = mac1.generate_signals()
    # # print signals.tail()
    # portfolio = MarketOnClosePortfolio(
    #     'test', bars, signals, initial_capital=3000000.0)
     
    # returns = portfolio.backtest_portfolio()


    MA1 = 5
    MA2 = 30
    MA3 = 60

    
    # """
    # print pdvol1.ix[pdvol1.longsignal ==1]
    
    
    
    bars['bench'] = bars['Close'].pct_change().cumsum()
    bars['benchdiff'] = bars['bench'].diff()

    # stance = 'none'

    newtradesig = np.zeros(len(bars['Close']))

    dmani._genStdsigs(bars)    
    # print 'std bars gen end'        
    
    try:
        bars2 = deepcopy(bars)
        
        # bars2 = bars2.astype(np.float64,copy = False)
        # barsStdnp = (bars2['Std'].values + bars2['Std'].values) /(bars2['Avg'].values-bars2['Std'].values)

        bars2['Std'] = bars2['Std'].astype(np.float64)
        bars2['Avg'] = bars2['Avg'].astype(np.float64)
        barsStddf = bars2['Stdsig'] = (bars2['Std'] + bars2['Std']) /(bars2['Avg']-bars2['Std'])
        
        # bars_25p = bars2['Stdsig'].describe()['25%']
        # bars_50p = bars2['Stdsig'].describe()['50%']
        # bars_75p = bars2['Stdsig'].describe()['75%']
        # bars_90p = bars2['Stdsig'].quantile(0.9)
        
        if stdarg == 'generate':    
            bars_25p = bars2['Stdsig'][:day].describe()['25%']
            # bars_50p = bars2['Stdsig'][:day].describe()['50%']
            bars_50p = bars2['Stdsig'][:day].quantile(0.5)
            bars_75p = bars2['Stdsig'][:day].describe()['75%']
            # bars_75p = bars2['Stdsig'][:day].quantile(0.9)
            bars_90p = bars2['Stdsig'][:day].quantile(0.9)
            print 'std bars real time not from DB'
    except Exception,e:
        print 'bars2 stdarg gen error',e
    # print 'bars 25%:',bars_25p,'bars_50p:',bars_50p,'bars 75%:',bars_75p,'bars_90p:',bars_90p
    # print '----bars std sig----'
    # print barsStddf
    # print 'obv gen start2'      
    
    try:
        global plot_maxstddf
        global plot_minstddf
        global plot_mmstddf
        # plot_maxstddf = maxsigdf_std
        # plot_minstddf = minsigdf_std
        # plot_mmstddf = mmsigdf_std
        # plot_mmstddf= plot_mmstddf.diff()
        # mmsigdf_std['Stdsig'] = bars2['Stdsig']
        # display(HTML(mmsigdf_std.to_html()))
        global gbars2  
        
        # bars2['OBV'] = bars['Volume']*bars['OpenCloseSig']
        # bars2['OBV'] = bars2['OBV'].cumsum()
        # # bars2['VolAvg'] =  [bars2['Volume'][day-20:day].mean() for day in range(len(bars['Close'])) ]
        # # bars2['VolStdSig'] =  (bars2['VolStd'] + bars2['VolStd']) /(bars2['VolAvg']-bars2['VolStd'])
        # bars2['StdCorr'] = [bars2['Stdsig'][:day].corr(bars2['OBV'][:day]) for day in range(len(bars['Close'])) ]
        # bars['OBV'] = bars2['OBV']
        # barsOBVMA1 = pd.rolling_mean(bars['OBV'], 5, min_periods=1).fillna(0.0)
        # barsobvma1np = barsOBVMA1.values 
        # barsOBVMA2 = pd.rolling_mean(bars['OBV'], 10, min_periods=1).fillna(0.0)
        # barsobvma2np = barsOBVMA1.values 
        # barsobvnp = bars['OBV'].values 
        # bars['Stdsig'] = bars2['Stdsig']
        # bars['OpenCloseSig'] = bars2['OpenCloseSig']
        gbars2 = bars

        # global gmmsignp4
        # global gmmsignp5
        # gmmsignp4 = mmsignp4
        # gmmsignp5 = mmsignp5
        # global gmaxsigdf_std
        # global gminsigdf_std
        # gmaxsigdf_std = maxsigdf_std
        # gminsigdf_std = minsigdf_std
        # print 'std bars obv gen end'
    except Exception,e:
        print 'error B',e    
     

    """
    kalman filter
    """
    #"""
    try:
      
        initial_state_mean_in = bars['Close'][0]
        observations = bars['Close'].values

        # ukf = UnscentedKalmanFilter(lambda x, w: x + np.sin(w), lambda x, v: x + v, transition_covariance=0.1,initial_state_mean=initial_state_mean_in)
        ukf = UnscentedKalmanFilter(tran_func, lambda x, v: x + np.sin(v), transition_covariance=0.1,initial_state_mean=initial_state_mean_in)
        (filtered_state_means, filtered_state_covariances) = ukf.filter(observations)
        (smoothed_state_means, smoothed_state_covariances) = ukf.smooth(observations)
        predictdf = pd.DataFrame({'predict':filtered_state_means[:,0],'smooth':smoothed_state_means[:,0]},index=bars.index)

        bars['predict'] = predictdf['predict']
        bars['smooth'] = predictdf['smooth']

        bars['benchsig'] =  0
        bars['benchsig'][bars['predict'] <= bars['smooth']] = 1
        bars['benchsigdiff'] = bars['benchsig'].diff()
        
        

    except Exception,e:
        print 'kalman error ',e
    #"""        
    """ kalman end"""
    

    barsMAranges = dmani._rangeDivide(bars)    
    dmani._genVolwithBenchdiff(bars,barsMAranges)
    dmani._genVolwithStdsig(bars)
    
    try:
        sp = pd.DataFrame(bars['Close'],index = bars.index,columns=['Price'])
        ms = pd.concat([sp,signals],axis=1).fillna(0.0)
        ms['newsignals'] = bars['benchsig']
        # display(HTML(ms[['newsignals']].to_html()))
        # initial_capital = bars['Close'][0] *110
        initial_capital = 10000000 
        portfolio_bench = MarketOnMixedPortfolio(
            'KDAS', ms, initial_capital=initial_capital)
        
         
        returns_bench = portfolio_bench.backtest_portfolio()
        # display(HTML(returns_bench.to_html()))
        print '--------------algo Result-------------------------'    
        
        barstmp = deepcopy(bars)
        barstmp['reversesig'] = (barstmp['benchsigdiff']*-1)
        barstmp['calbenchsig'] = barstmp['reversesig'] * barstmp['bench']
        totalreturngain_algo1 = barstmp['calbenchsig'].sum()
        tradingnum_algo1 = len(bars['benchsigdiff'][bars['benchsigdiff'] != 0]) /2
        print 'start closep:',bars['Close'][0],'current closep:',bars['Close'][-1]
        print 'total Account Gain:', (returns_bench.total[returns_bench.total.index[-1]] - initial_capital)/initial_capital,'initial_capital:',initial_capital,\
                'totalReturn:',returns_bench.total[returns_bench.total.index[-1]]
        print 'totalAccum Gain:',returns_bench.total.pct_change().cumsum()[returns_bench.total.pct_change().cumsum().index[-1]],\
               'tradingGain',totalreturngain_algo1,'tradingnum_algo1:',tradingnum_algo1
        print 'benchmark:',bars['bench'][-1]

        disbars = barstmp[['bench','benchsigdiff','reversesig']]
        disbars = disbars[disbars['benchsigdiff'] != 0]
        disbars = disbars[-5:]
        # display(HTML(disbars.to_html()))
        print '--------------algo Result End-------------------------'    
        # display(HTML(returns_bench.to_html()))
        # display(HTML(returns3.to_html()))
        # print returns_bench.head(10)
        # print returns3.head(10)

        # newtradesig = np.zeros(len(bars['Close']))
        talgodf = talgo._HGtradingalgo_verEJ(bars,barsMAranges)
        bars['tsig'] = talgodf['tsig']
        ms['newsignals'] = 0
        ms['newsignals'] = bars['tsig']
        ms['bench'] = bars['bench']
        # display(HTML(ms[['newsignals']].to_html()))
        # initial_capital = bars['Close'][0] *110 
        initial_capital = 10000000
        portfolio_talgo = MarketOnMixedPortfolio(
            'KDAS', ms, initial_capital=initial_capital)
        
         
        returns_talgo = portfolio_talgo.backtest_portfolio()
        # display(HTML(returns_talgo.to_html()))
        bars['tsigdiff'] = bars['tsig'].diff()
        print '--------------algo Result-------------------------'    
        
        # barstmp = deepcopy(bars)
        # barstmp['reversesig'] = (barstmp['tsigdiff']*-1)

        talibtechsignal(bars)


        buybench = 0.0
        totalgain = 0.0
        stance = 'none'
        lastbuydate = 'NA'
        for day in range(len(bars)):
            if bars['tsigdiff'][day] == 1:
                buybench = bars['bench'][day]
                stance = 'hold'
                lastbuydate = str(bars.index[day])
            elif bars['tsigdiff'][day] == -1 and buybench != 0.0:
                totalgain += (bars['bench'][day] - buybench)
                stance = 'none'

        totalreturngain_algo1 = totalgain
        tradingnum_algo1 = len(bars['tsigdiff'][bars['tsigdiff'] != 0]) /2
        print 'start closep:',bars['Close'][0],'current closep:',bars['Close'][-1]
        print 'total Account Gain:', (returns_talgo.total[returns_talgo.total.index[-1]] - initial_capital)/initial_capital,'initial_capital:',initial_capital,\
                'totalReturn:',returns_talgo.total[returns_talgo.total.index[-1]]
        print 'totalAccum Gain:',returns_talgo.total.pct_change().cumsum()[returns_talgo.total.pct_change().cumsum().index[-1]],\
               'tradingGain',totalreturngain_algo1,'tradingnum_algo1:',tradingnum_algo1
        print 'benchmark:',bars['bench'][-1]
        print 'stance',stance
        print 'lastbuydate',lastbuydate
        print 'MACDSig',bars['MACDSig'][-1],'SLOWSig',bars['SLOWSig'][-1]

        # disbars = barstmp[['bench','tsig','calbenchsig']]
        # disbars = disbars[barstmp['tempreversesig'] != 0]
        # display(HTML(disbars.to_html()))
        print '--------------algo Result End-------------------------'    
    except Exception,e:
        print 'algo result error ',e

    global gtix
    global gobv
    global gobv2
    global gmmsignp_obv
    # gtix = tix
    # gobv = barsOBVMA1
    # gobv2 = barsOBVMA2
    # gmmsignp_obv = mmsignp_obv
    
    
                   
    '''
    draw matplotlib chart
    '''
    #''' plot draw
    # """
    if writedblog == 'none':
        fig = plt.figure(figsize=(20, 30))

        fig.patch.set_facecolor('white')     # Set the outer colour to white
        ax1 = fig.add_subplot(811,  ylabel='Price in $')

        # # Plot the AAPL closing price overlaid with the moving averages
        bars['Close'].plot(ax=ax1, color='r', lw=2.)
        signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)

        # # # Plot the "buy" trades against AAPL
        # ax1.plot(signals.ix[signals.positions == 1.0].index,
        #           signals.short_mavg[signals.positions == 1.0],
        #           '^', markersize=10, color='m')

        # # # Plot the "sell" trades against AAPL
        # ax1.plot(signals.ix[signals.positions == -1.0].index,
        #           signals.short_mavg[signals.positions == -1.0],
        #           'v', markersize=10, color='k')


        from matplotlib.finance import candlestick
        from itertools import izip
        ax3 = fig.add_subplot(812,ylabel='Candlestick')
        ax3dates = bars.index.to_pydatetime() 
        ax3times = date2num(ax3dates)
        quotes = izip(ax3times,bars['Open'],bars['Close'],bars['High'],bars['Low'])
        candlestick(ax3,quotes,width=1.5, colorup='g', colordown='r', alpha=1.0)
        ax3.xaxis.set_major_locator(mticker.MaxNLocator(10))
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax3.set_xlim(ax3times[0], ax3times[-1])

        ax4 = fig.add_subplot(813,  ylabel='Volume')
        bars['Volume'].plot(ax=ax4, color='#EF15C3', lw=2.)

        # ax4.plot(date,volume,'#EF15C3',label='VolAV1',linewidth=1.5)
        # ax4.xaxis.set_major_locator(mticker.MaxNLocator(10))
        # ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        # ax4.set_xlim(date[0], date[-1])

        # datestr1 = datetime.fromtimestamp(time.mktime(mdates.num2date(date[0]).timetuple())).strftime('%Y %m %d')
        # print datestr1,len(bars['Close']),len(date),len(volume)
        # print bars.head()
        # plt.show()
        # '''             
    # """
    if writedblog == 'none':
        #''' plot draw
        # ax5 = fig.add_subplot(814,  ylabel='Return')
        # print returns2[-10:]
        # print len(returns2),len(returns2['total'])
        # print returns2['total'][-10:]
        # print returns2['total'].index
        
        ax6 = fig.add_subplot(814,  ylabel='Return Percentage')
        ax6.plot(returns_bench.index,
                  returns_bench['total'].pct_change().cumsum(),
                   color='r', lw=2.)
        ax6.plot(returns_bench.ix[returns_bench['buysignal'] == 100].index,
          returns_bench['total'].pct_change().cumsum()[returns_bench['buysignal'] == 100],
          '^', markersize=10, color='r')
        ax6.plot(returns_bench.ix[returns_bench['sellsignal'] == -100].index,
          returns_bench['total'].pct_change().cumsum()[returns_bench['sellsignal'] == -100],
          'v', markersize=10, color='b')

        ax7 = fig.add_subplot(815,  ylabel='Benchmark')
        ax7.plot(bars.index,
                  bars['Close'].pct_change().cumsum(),
                   color='r', lw=2.)
        ax7.plot(bars['bench'].ix[returns_bench['buysignal'] == 100].index,
          bars['bench'][returns_bench['buysignal'] == 100],
          '^', markersize=10, color='r')
        ax7.plot(bars['bench'].ix[returns_bench['sellsignal'] == -100].index,
          bars['bench'][returns_bench['sellsignal'] == -100],
          'v', markersize=10, color='b')

        ax8 = fig.add_subplot(816,  ylabel='Benchmark_algo')
        ax8.plot(bars.index,
                  bars['Close'].pct_change().cumsum(),
                   color='r', lw=2.)
        ax8.plot(bars['bench'].ix[returns_talgo['buysignal'] == 100].index,
          bars['bench'][returns_talgo['buysignal'] == 100],
          '^', markersize=10, color='r')
        ax8.plot(bars['bench'].ix[returns_talgo['sellsignal'] == -100].index,
          bars['bench'][returns_talgo['sellsignal'] == -100],
          'v', markersize=10, color='b')

        ax9 = fig.add_subplot(817,  ylabel='Return Percentage algo')
        ax9.plot(returns_talgo.index,
                  returns_talgo['total'].pct_change().cumsum(),
                   color='r', lw=2.)
        ax9.plot(returns_talgo.ix[returns_talgo['buysignal'] == 100].index,
          returns_talgo['total'].pct_change().cumsum()[returns_talgo['buysignal'] == 100],
          '^', markersize=10, color='r')
        ax9.plot(returns_talgo.ix[returns_talgo['sellsignal'] == -100].index,
          returns_talgo['total'].pct_change().cumsum()[returns_talgo['sellsignal'] == -100],
          'v', markersize=10, color='b')
        
        # ax10 = fig.add_subplot(818,  ylabel='Std')
        # ax10.plot(bars2.index,
        #           bars2['Stdsig'],
        #            color='r', lw=2.)
        # ax10.axhline(y=bars_50p,color='r')        
        # ax10.axhline(y=bars_75p,color = 'b')        
        # ax10.axhline(y=bars_90p,color = 'green')        
        
        
        # ax10 = fig.add_subplot(919,  ylabel='StdCorr')
        # ax10.plot(bars2.index,bars2['StdCorr'],color = 'r',lw=2.)
        # try:
            

        #     fig1 = plt.figure(figsize=(20, 10))
        #     ax1 = fig1.add_subplot(311,  ylabel='OBV')
        #     ax1.plot(bars.index,bars['OBV'],color = 'r',lw=2.)
        #     # barsOBVMA1.plot(ax=ax1, lw=2.)
        #     # barsOBVMA2.plot(ax=ax1, lw=2.)
        #     ax2 = fig1.add_subplot(312,  ylabel='kalman')
        #     # closeMA1 = pd.rolling_mean(bars['Close'],5,min_periods=1).fillna(0.0)
        #     # ax2.plot(bars.index,closeMA1,color = 'c',lw=2)
        #     bars['Close'].plot(ax=ax2,lw=2)
        #     # predictdf[['predict','smooth']].plot(ax=ax2,lw=2)
        #     global gpredictdf
        #     gpredictdf = predictdf
        #     # ax3 = fig1.add_subplot(313,  ylabel='kalman')
        #     # bars[['techsigcnt','tradesigcnt','obvmmcnt']].plot(ax=ax3,lw=2)
        #     # barstmp[['obvpredict','obvsmooth']].plot(ax=ax3,lw=2)
        #     # ax2 = fig1.add_subplot(312,  ylabel='StdCorr')
        #     # ax2.plot(bars2.index,bars2['StdCorr'],color = 'r',lw=2.)

        #     # ax3 = fig1.add_subplot(212,  ylabel='tangent')
        #     # ax3.plot(bars.index,tangents,alpha=0.5)
        #     # ax3.axhline(y=0,color = 'red')        
        #     # ax3.axhline(y=tangent_25p,color = 'green',linewidth= 2)        
        #     # ax3.axhline(y=tangent_50p,color = 'purple',linewidth= 2)        


            

            
            
        # except Exception,e:
        #     print 'fig draw error ',e
        plt.show()
    # """
    # print 'bars_50p',bars_50p,'bars_75p',bars_75p,'bars_90p',bars_90p
    # fig.savefig("../../data/png/"+namearg+".png", dpi = 100)
    #''' 
    # fig.savefig("../../data/"+name+".png", dpi = 400)

    # print newsigstr_arr1,newsigstr_arr2,newsigstr_arr3,newsigstr_arr4,type(newsigstr_arr1)

    
    ''' display plotly '''
    if plotly == 'plotly':
        returns3 = 'none'
        saveplotlyvar(bars,returns_bench,returns3)
    print '----------------------------Run Session End-------------------------'    
    
    # bars['Bench'] = bars['Close'].pct_change().cumsum()
    barsgain5 = bars['Close'][-5:].pct_change().cumsum()[bars['Close'][-5:].pct_change().cumsum().index[-1]]
    barsgain15 = bars['Close'][-15:].pct_change().cumsum()[bars['Close'][-15:].pct_change().cumsum().index[-1]]
    barsgain30 = bars['Close'][-30:].pct_change().cumsum()[bars['Close'][-30:].pct_change().cumsum().index[-1]]
    barsgain50 = bars['Close'][-50:].pct_change().cumsum()[bars['Close'][-50:].pct_change().cumsum().index[-1]]

    """ check kalman result """
    
    kalresultdf = pd.DataFrame({'tday':newsigstr_arr1_algo2,'stance':newsigstr_arr2_algo2,'tcount':newsigstr_arr6_algo2},index = newsigstr_arr1_algo2)
    barskaldf = pd.concat([bars,kalresultdf], axis = 1)
    # display(HTML(barskaldf.to_html()))
    global gbarskaldf
    gbarskaldf = barskaldf
    
    # bars = bars.reset_index()
    # bars.to_csv('bars.csv')
    """ check kalman end """

    # return returns2.total.pct_change().cumsum()[returns2.total.pct_change().cumsum().index[-1]],totalreturngain_algo1,tradingnum_algo1 \
    #     ,returns3.total.pct_change().cumsum()[returns3.total.pct_change().cumsum().index[-1]],totalreturngain_algo2,tradingnum_algo2,closepgainnp[-1]\
    #     ,bars_25p,bars_50p,bars_75p,bars_90p,stance_algo1,stance_algo2,barsStdnp[-1],barsgain5,barsgain15,barsgain30,barsgain50


    return  codearg,namearg\
           ,returns_talgo.total.pct_change().cumsum()[returns_talgo.total.pct_change().cumsum().index[-1]]\
           ,totalreturngain_algo1\
           ,tradingnum_algo1\
           ,stance,bars['bench'][-1]\
           ,returns_talgo.total[returns_talgo.total.index[-1]]\
           ,lastbuydate\
           ,bars['MACDSig'][-1],bars['SLOWSig'][-1]


def RunSimul_world(symbol,startdate,*args):

    
    bars = Quandl.get(symbol,  trim_start=startdate, trim_end=datetime.today(),authtoken="")
    
    # bars = bars.drop(bars.index[-2])    
    today  = datetime.today()
    todayweek = today.weekday()

    bars['week'] = bars.index.weekday
    tailweekdays = bars['week'][-5:]
    # print tailweekdays

    # if 0 <= todayweek <=4 :
    #     for cnt in range(0,len(tailweekdays)):
    #         day = cnt +1 
    #         # print gbars2['week'][-1*day]
    #         checkday = bars['week'][-1*day]
            
    #         # print todayweek,checkday,bars.index[-1*day]
    #         if todayweek != checkday:
    #             raise Exception("week check error")
    #         if todayweek == 0:
    #             todayweek = 4
    #         else:
    #             todayweek = todayweek - 1

    # bars = bars[:len(bars)-100]  
    # bars = bars[:'2015-07-07']
    # print bars.tail()  
    '''
    test code for inflection point
    '''
    # bars = bars.drop(bars.tail(1).index)
    # print '----------final test bars---------'
    # print bars.tail()    
    '''
    end test code
    '''

    mac1 = stcore.MovingAverageCrossStrategy(
    'test', bars, short_window=5, long_window=20)
    signals = mac1.generate_signals()
    # # print signals.tail()
    # portfolio = MarketOnClosePortfolio(
    #     'test', bars, signals, initial_capital=3000000.0)
     
    # returns = portfolio.backtest_portfolio()


    MA1 = 5
    MA2 = 30
    MA3 = 60

    
    # """
    # print pdvol1.ix[pdvol1.longsignal ==1]
    
    
    
    bars['bench'] = bars['Close'].pct_change().cumsum()
    bars['benchdiff'] = bars['bench'].diff()

    # stance = 'none'

    newtradesig = np.zeros(len(bars['Close']))

    dmani._genStdsigs(bars)    
    # print 'std bars gen end'        
    
    try:
        bars2 = deepcopy(bars)
        
        # bars2 = bars2.astype(np.float64,copy = False)
        # barsStdnp = (bars2['Std'].values + bars2['Std'].values) /(bars2['Avg'].values-bars2['Std'].values)

        bars2['Std'] = bars2['Std'].astype(np.float64)
        bars2['Avg'] = bars2['Avg'].astype(np.float64)
        barsStddf = bars2['Stdsig'] = (bars2['Std'] + bars2['Std']) /(bars2['Avg']-bars2['Std'])
        
        # bars_25p = bars2['Stdsig'].describe()['25%']
        # bars_50p = bars2['Stdsig'].describe()['50%']
        # bars_75p = bars2['Stdsig'].describe()['75%']
        # bars_90p = bars2['Stdsig'].quantile(0.9)
        
        
    except Exception,e:
        print 'bars2 stdarg gen error',e
    # print 'bars 25%:',bars_25p,'bars_50p:',bars_50p,'bars 75%:',bars_75p,'bars_90p:',bars_90p
    # print '----bars std sig----'
    # print barsStddf
    # print 'obv gen start2'      
    
    try:
        global plot_maxstddf
        global plot_minstddf
        global plot_mmstddf
        # plot_maxstddf = maxsigdf_std
        # plot_minstddf = minsigdf_std
        # plot_mmstddf = mmsigdf_std
        # plot_mmstddf= plot_mmstddf.diff()
        # mmsigdf_std['Stdsig'] = bars2['Stdsig']
        # display(HTML(mmsigdf_std.to_html()))
        global gbars2  
        
        # bars2['OBV'] = bars['Volume']*bars['OpenCloseSig']
        # bars2['OBV'] = bars2['OBV'].cumsum()
        # # bars2['VolAvg'] =  [bars2['Volume'][day-20:day].mean() for day in range(len(bars['Close'])) ]
        # # bars2['VolStdSig'] =  (bars2['VolStd'] + bars2['VolStd']) /(bars2['VolAvg']-bars2['VolStd'])
        # bars2['StdCorr'] = [bars2['Stdsig'][:day].corr(bars2['OBV'][:day]) for day in range(len(bars['Close'])) ]
        # bars['OBV'] = bars2['OBV']
        # barsOBVMA1 = pd.rolling_mean(bars['OBV'], 5, min_periods=1).fillna(0.0)
        # barsobvma1np = barsOBVMA1.values 
        # barsOBVMA2 = pd.rolling_mean(bars['OBV'], 10, min_periods=1).fillna(0.0)
        # barsobvma2np = barsOBVMA1.values 
        # barsobvnp = bars['OBV'].values 
        # bars['Stdsig'] = bars2['Stdsig']
        # bars['OpenCloseSig'] = bars2['OpenCloseSig']
        gbars2 = bars

        # global gmmsignp4
        # global gmmsignp5
        # gmmsignp4 = mmsignp4
        # gmmsignp5 = mmsignp5
        # global gmaxsigdf_std
        # global gminsigdf_std
        # gmaxsigdf_std = maxsigdf_std
        # gminsigdf_std = minsigdf_std
        # print 'std bars obv gen end'
    except Exception,e:
        print 'error B',e    
     

    """
    kalman filter
    """
    #"""
    try:
      
        initial_state_mean_in = bars['Close'][0]
        observations = bars['Close'].values

        # ukf = UnscentedKalmanFilter(lambda x, w: x + np.sin(w), lambda x, v: x + v, transition_covariance=0.1,initial_state_mean=initial_state_mean_in)
        ukf = UnscentedKalmanFilter(tran_func, lambda x, v: x + np.sin(v), transition_covariance=0.1,initial_state_mean=initial_state_mean_in)
        (filtered_state_means, filtered_state_covariances) = ukf.filter(observations)
        (smoothed_state_means, smoothed_state_covariances) = ukf.smooth(observations)
        predictdf = pd.DataFrame({'predict':filtered_state_means[:,0],'smooth':smoothed_state_means[:,0]},index=bars.index)

        bars['predict'] = predictdf['predict']
        bars['smooth'] = predictdf['smooth']

        bars['benchsig'] =  0
        bars['benchsig'][bars['predict'] <= bars['smooth']] = 1
        bars['benchsigdiff'] = bars['benchsig'].diff()
        
        

    except Exception,e:
        print 'kalman error ',e
    #"""        
    """ kalman end"""
    

    barsMAranges = dmani._rangeDivide(bars)    
    dmani._genVolwithBenchdiff(bars,barsMAranges)
    dmani._genVolwithStdsig(bars)
    
    try:
        sp = pd.DataFrame(bars['Close'],index = bars.index,columns=['Price'])
        ms = pd.concat([sp,signals],axis=1).fillna(0.0)
        ms['newsignals'] = bars['benchsig']
        # display(HTML(ms[['newsignals']].to_html()))
        # initial_capital = bars['Close'][0] *110
        initial_capital = 10000000 
        portfolio_bench = MarketOnMixedPortfolio(
            'KDAS', ms, initial_capital=initial_capital)
        
         
        returns_bench = portfolio_bench.backtest_portfolio()
        # display(HTML(returns_bench.to_html()))
        print '--------------algo Result-------------------------'    
        
        barstmp = deepcopy(bars)
        barstmp['reversesig'] = (barstmp['benchsigdiff']*-1)
        barstmp['calbenchsig'] = barstmp['reversesig'] * barstmp['bench']
        totalreturngain_algo1 = barstmp['calbenchsig'].sum()
        tradingnum_algo1 = len(bars['benchsigdiff'][bars['benchsigdiff'] != 0]) /2
        print 'start closep:',bars['Close'][0],'current closep:',bars['Close'][-1]
        print 'total Account Gain:', (returns_bench.total[returns_bench.total.index[-1]] - initial_capital)/initial_capital,'initial_capital:',initial_capital,\
                'totalReturn:',returns_bench.total[returns_bench.total.index[-1]]
        print 'totalAccum Gain:',returns_bench.total.pct_change().cumsum()[returns_bench.total.pct_change().cumsum().index[-1]],\
               'tradingGain',totalreturngain_algo1,'tradingnum_algo1:',tradingnum_algo1
        print 'benchmark:',bars['bench'][-1]

        disbars = barstmp[['bench','benchsigdiff','reversesig']]
        disbars = disbars[disbars['benchsigdiff'] != 0]
        # display(HTML(disbars.to_html()))
        print '--------------algo Result End-------------------------'    
        # display(HTML(returns_bench.to_html()))
        # display(HTML(returns3.to_html()))
        # print returns_bench.head(10)
        # print returns3.head(10)

        # newtradesig = np.zeros(len(bars['Close']))
        talgodf = talgo._HGtradingalgo_verEJ(bars,barsMAranges)
        bars['tsig'] = talgodf['tsig']
        ms['newsignals'] = 0
        ms['newsignals'] = bars['tsig']
        ms['bench'] = bars['bench']
        # display(HTML(ms[['newsignals']].to_html()))
        # initial_capital = bars['Close'][0] *110 
        initial_capital = 10000000
        portfolio_talgo = MarketOnMixedPortfolio(
            'KDAS', ms, initial_capital=initial_capital)
        
         
        returns_talgo = portfolio_talgo.backtest_portfolio()
        # display(HTML(returns_talgo.to_html()))
        bars['tsigdiff'] = bars['tsig'].diff()
        print '--------------algo Result-------------------------'    
        
        # barstmp = deepcopy(bars)
        # barstmp['reversesig'] = (barstmp['tsigdiff']*-1)

        talibtechsignal(bars)


        buybench = 0.0
        totalgain = 0.0
        stance = 'none'
        lastbuydate = 'NA'
        for day in range(len(bars)):
            if bars['tsigdiff'][day] == 1:
                buybench = bars['bench'][day]
                stance = 'hold'
                lastbuydate = str(bars.index[day])
            elif bars['tsigdiff'][day] == -1 and buybench != 0.0:
                totalgain += (bars['bench'][day] - buybench)
                stance = 'none'

        totalreturngain_algo1 = totalgain
        tradingnum_algo1 = len(bars['tsigdiff'][bars['tsigdiff'] != 0]) /2
        print 'start closep:',bars['Close'][0],'current closep:',bars['Close'][-1]
        print 'total Account Gain:', (returns_talgo.total[returns_talgo.total.index[-1]] - initial_capital)/initial_capital,'initial_capital:',initial_capital,\
                'totalReturn:',returns_talgo.total[returns_talgo.total.index[-1]]
        print 'totalAccum Gain:',returns_talgo.total.pct_change().cumsum()[returns_talgo.total.pct_change().cumsum().index[-1]],\
               'tradingGain',totalreturngain_algo1,'tradingnum_algo1:',tradingnum_algo1
        print 'benchmark:',bars['bench'][-1]
        print 'stance',stance
        print 'lastbuydate',lastbuydate
        print 'MACDSig',bars['MACDSig'][-1],'SLOWSig',bars['SLOWSig'][-1]

        # disbars = barstmp[['bench','tsig','calbenchsig']]
        # disbars = disbars[barstmp['tempreversesig'] != 0]
        # display(HTML(disbars.to_html()))
        print '--------------algo Result End-------------------------'    
    except Exception,e:
        print 'algo result error ',e

    global gtix
    global gobv
    global gobv2
    global gmmsignp_obv
    # gtix = tix
    # gobv = barsOBVMA1
    # gobv2 = barsOBVMA2
    # gmmsignp_obv = mmsignp_obv
    
    
    
    
    print '----------------------------Run Session End-------------------------'    
    
    # bars['Bench'] = bars['Close'].pct_change().cumsum()
    barsgain5 = bars['Close'][-5:].pct_change().cumsum()[bars['Close'][-5:].pct_change().cumsum().index[-1]]
    barsgain15 = bars['Close'][-15:].pct_change().cumsum()[bars['Close'][-15:].pct_change().cumsum().index[-1]]
    barsgain30 = bars['Close'][-30:].pct_change().cumsum()[bars['Close'][-30:].pct_change().cumsum().index[-1]]
    barsgain50 = bars['Close'][-50:].pct_change().cumsum()[bars['Close'][-50:].pct_change().cumsum().index[-1]]

    """ check kalman result """
    
    kalresultdf = pd.DataFrame({'tday':newsigstr_arr1_algo2,'stance':newsigstr_arr2_algo2,'tcount':newsigstr_arr6_algo2},index = newsigstr_arr1_algo2)
    barskaldf = pd.concat([bars,kalresultdf], axis = 1)
    # display(HTML(barskaldf.to_html()))
    global gbarskaldf
    gbarskaldf = barskaldf
    
    # bars = bars.reset_index()
    # bars.to_csv('bars.csv')
    """ check kalman end """

    # return returns2.total.pct_change().cumsum()[returns2.total.pct_change().cumsum().index[-1]],totalreturngain_algo1,tradingnum_algo1 \
    #     ,returns3.total.pct_change().cumsum()[returns3.total.pct_change().cumsum().index[-1]],totalreturngain_algo2,tradingnum_algo2,closepgainnp[-1]\
    #     ,bars_25p,bars_50p,bars_75p,bars_90p,stance_algo1,stance_algo2,barsStdnp[-1],barsgain5,barsgain15,barsgain30,barsgain50


    return returns_talgo.total.pct_change().cumsum()[returns_talgo.total.pct_change().cumsum().index[-1]]\
           ,totalreturngain_algo1\
           ,tradingnum_algo1\
           ,stance,bars['bench'][-1]\
           ,returns_talgo.total[returns_talgo.total.index[-1]]\
           ,lastbuydate\
           ,bars['MACDSig'][-1],bars['SLOWSig'][-1]

    

    
global barstmp_ret
def RunSimul_Kalman(codearg,typearg,namearg,mode,dbmode,histmode,runcount,srcsite,writedblog,updbpattern,appenddb,startdatemode 
    ,dbtradinghist,plotly,*args):

    print 'RunSimul_Kalman inside'
    start = time.clock()
    if typearg == 4:# DOW,NASDAQ,S&P500
        startdate= '2014-01-01'
        arg_index = 0
        stdarg = args[arg_index]
        arg_index += 1
        smallarg = args[arg_index]
        arg_index += 1
        dayselect = args[arg_index]
        arg_index += 1
        tangentmode = args[arg_index]
        print 'stdarg',stdarg,'tangentmode',tangentmode
        tangentmode = 'tan_gen'
        if namearg == 'dow':
            symbol = 'GOOG/INDEXDJX_DJI'
            bars =  Quandl.get(symbol, collapse='Daily', trim_start=startdate, trim_end=datetime.today(),authtoken="")
        elif namearg == 'nasdaq':
            symbol = '^IXIC'
            import pandas.io.data as web
            bars = web.get_data_yahoo(symbol,startdate)
        elif namearg == 'sandp':
            symbol = '^GSPC'
            import pandas.io.data as web
            bars = web.get_data_yahoo(symbol,startdate)

        if mode =='dbpattern'  or dbmode == 'dbpattern':        
            if updbpattern == 'none':
                print 'read DB patterns'
                patternAr, extractid= stcore.ReadPatternsFromDB(codearg,typearg,namearg,mode)
                patternAppendAr = stcore.ReadPatternsAppendFromDB(codearg,namearg)
            elif updbpattern == 'updbpattern':
                print 'read UP DB patterns'
                patternAr, extractid= stcore.ReadUpPatternsFromDB(codearg,typearg,namearg,mode)

            if patternAr == -1:
                print 'real time gen db pattern'
                startdate = '2011-01-01'
                dbmode  = 'none'


    else:    
        bars,patternAr, extractid,patternAppendAr,bars_25p,bars_50p\
        ,bars_75p,bars_90p,tangent_25p,tangent_50p,tangentmode\
        ,startdate,dbmode,stdarg,smallarg,dayselect,tangentmode =\
            stcore._inRunSimul_FetchData(codearg,typearg,namearg,mode,dbmode,histmode,runcount,srcsite,writedblog,updbpattern,appenddb,startdatemode \
                            ,dbtradinghist,plotly,*args)
        # print 'stdarg',stdarg,'smallarg',smallarg,'dayselect',dayselect,'tangentmode',tangentmode
        
    
    # bars = bars.drop(bars.index[-1])    
    
    today  = datetime.today()
    todayweek = today.weekday()

    bars['week'] = bars.index.weekday
    tailweekdays = bars['week'][-5:]
    # print tailweekdays

    # if 0 <= todayweek <=4 :
    #     for cnt in range(0,len(tailweekdays)):
    #         day = cnt +1 
    #         # print gbars2['week'][-1*day]
    #         checkday = bars['week'][-1*day]
            
    #         # print todayweek,checkday,bars.index[-1*day]
    #         if todayweek != checkday:
    #             raise Exception("week check error")
    #         if todayweek == 0:
    #             todayweek = 4
    #         else:
    #             todayweek = todayweek - 1

    # bars = bars[:len(bars)-100]  
    # bars = bars[:'2015-07-07']
    # print bars.tail()  
    '''
    test code for inflection point
    '''
    # bars = bars.drop(bars.tail(240*6).index)
    # bars = bars.drop(bars.tail(20*2+5).index)
    # print '----------final test bars---------'
    # print bars.tail()    
    '''
    end test code
    '''

    mac1 = stcore.MovingAverageCrossStrategy(
    'test', bars, short_window=5, long_window=20)
    signals = mac1.generate_signals()
    # # print signals.tail()
    # portfolio = MarketOnClosePortfolio(
    #     'test', bars, signals, initial_capital=3000000.0)
     
    # returns = portfolio.backtest_portfolio()


    MA1 = 5
    MA2 = 30
    MA3 = 60

    
    # """
    # print pdvol1.ix[pdvol1.longsignal ==1]
    
    
    
    bars['bench'] = bars['Close'].pct_change().cumsum()
    bars['benchdiff'] = bars['bench'].diff()

    # stance = 'none'

    newtradesig = np.zeros(len(bars['Close']))

    dmani._genStdsigs(bars)    
    # print 'std bars gen end'        
    
    try:
        bars2 = deepcopy(bars)
        
        # bars2 = bars2.astype(np.float64,copy = False)
        # barsStdnp = (bars2['Std'].values + bars2['Std'].values) /(bars2['Avg'].values-bars2['Std'].values)

        bars2['Std'] = bars2['Std'].astype(np.float64)
        bars2['Avg'] = bars2['Avg'].astype(np.float64)
        barsStddf = bars2['Stdsig'] = (bars2['Std'] + bars2['Std']) /(bars2['Avg']-bars2['Std'])
        
        # bars_25p = bars2['Stdsig'].describe()['25%']
        # bars_50p = bars2['Stdsig'].describe()['50%']
        # bars_75p = bars2['Stdsig'].describe()['75%']
        # bars_90p = bars2['Stdsig'].quantile(0.9)
        
        if stdarg == 'generate':    
            bars_25p = bars2['Stdsig'][:day].describe()['25%']
            # bars_50p = bars2['Stdsig'][:day].describe()['50%']
            bars_50p = bars2['Stdsig'][:day].quantile(0.5)
            bars_75p = bars2['Stdsig'][:day].describe()['75%']
            # bars_75p = bars2['Stdsig'][:day].quantile(0.9)
            bars_90p = bars2['Stdsig'][:day].quantile(0.9)
            print 'std bars real time not from DB'
    except Exception,e:
        print 'bars2 stdarg gen error',e
    # print 'bars 25%:',bars_25p,'bars_50p:',bars_50p,'bars 75%:',bars_75p,'bars_90p:',bars_90p
    # print '----bars std sig----'
    # print barsStddf
    # print 'obv gen start2'      
    
    
    """
    kalman filter
    """
    #"""
    try:
        
        # barsMAranges = dmani._rangeDivide(bars)    
        # dmani._genVolwithBenchdiff(bars,barsMAranges)
        # dmani._genVolwithStdsig(bars)

        initial_state_mean_in = bars['Close'][0]
        observations = bars['Close'].values

        # ukf = UnscentedKalmanFilter(lambda x, w: x + np.sin(w), lambda x, v: x + v, transition_covariance=0.1,initial_state_mean=initial_state_mean_in)
        ukf = UnscentedKalmanFilter(tran_func, lambda x, v: x + np.sin(v), transition_covariance=0.1,initial_state_mean=initial_state_mean_in)
        (filtered_state_means, filtered_state_covariances) = ukf.filter(observations)
        (smoothed_state_means, smoothed_state_covariances) = ukf.smooth(observations)
        predictdf = pd.DataFrame({'predict':filtered_state_means[:,0],'smooth':smoothed_state_means[:,0]},index=bars.index)

        bars['predict'] = predictdf['predict']
        bars['smooth'] = predictdf['smooth']

        bars['benchsig'] =  0
        bars['benchsig'][bars['predict'] <= bars['smooth']] = 1
        bars['benchsigdiff'] = bars['benchsig'].diff()
        
        bars['Volume_pct'] = bars['Volume'].pct_change()
        # bars['Volume_pct_short'] = pd.rolling_mean(bars['Volume_pct'],window=5)
        # bars['Volume_pct_std'] = 0.0
        # bars['Volume_pct_offset'] = 0.0
        # for intday in range(0,len(bars)):
        #     if intday > 0:
        #         vol_std_mean = bars['Volume_pct'][:intday].std()
        #         bars['Volume_pct_std'][intday] = bars['Volume_pct'][:intday].std()
        #         bars['Volume_pct_offset'][intday] = bars['Volume_pct'][intday] - vol_std_mean
        
        signalnp = bars['Close'].values
        dayslim = 5
        barsdf = bars
        mmsigdf,mmsignp,maxsigdf,minsigdf,maxsignp,minsignp,maxqueue,minqueue = inflectionPoint(signalnp,dayslim,barsdf)
        bars = pd.concat([mmsigdf,bars],axis=1)
        # disbars = bars[['bench','MMSignals']]
        # display(HTML(disbars.to_html()))
        # talgodf = talgo._HGtradingalgo_verYE(bars,-1)
        algomode = args[7]
        print 'RunSimul_Kalman len(*args)',args,len(args)
        if algomode == 'Sunny':
            talgodf = talgo._HGtradingalgo_verSunny(bars,-1,*args)
        elif algomode == 'Hyanggu':
            talgodf = talgo._HGtradingalgo_verHyanggu(bars,-1,*args)
        # talgodf = talgo._HGtradingalgo_verSunny_MA(bars,-1)
        global barstmp_ret
        barstmp_ret = bars

        disbars = bars[['bench','benchsigdiff_kr']]
        disbars = disbars[disbars['benchsigdiff_kr'] != 0]
        # disbars = disbars[-5:]
        # display(HTML(disbars.to_html()))
    except Exception,e:
        print 'kalman error ',e
        pass
    #"""        
    """ kalman end"""
    
    try:
        sp = pd.DataFrame(bars['Close'],index = bars.index,columns=['Price'])
        ms = pd.concat([sp,signals],axis=1).fillna(0.0)
        ms['newsignals'] = bars['benchsig']
        # display(HTML(ms[['newsignals']].to_html()))
        # initial_capital = bars['Close'][0] *110
        initial_capital = 10000000 
        portfolio_bench = MarketOnMixedPortfolio(
            'KDAS', ms, initial_capital=initial_capital)
        
         
        returns_bench = portfolio_bench.backtest_portfolio()
        # display(HTML(returns_bench.to_html()))
        print '--------------algo Result-------------------------'    
        
        barstmp = deepcopy(bars)
        barstmp['reversesig'] = (barstmp['benchsigdiff']*-1)
        barstmp['calbenchsig'] = barstmp['reversesig'] * barstmp['bench']
        totalreturngain_algo1 = barstmp['calbenchsig'].sum()
        tradingnum_algo1 = len(bars['benchsigdiff'][bars['benchsigdiff'] != 0]) /2
        print 'start closep:',bars['Close'][0],'current closep:',bars['Close'][-1]
        print 'total Account Gain:', (returns_bench.total[returns_bench.total.index[-1]] - initial_capital)/initial_capital,'initial_capital:',initial_capital,\
                'totalReturn:',returns_bench.total[returns_bench.total.index[-1]]
        print 'totalAccum Gain:',returns_bench.total.pct_change().cumsum()[returns_bench.total.pct_change().cumsum().index[-1]],\
               'tradingGain',totalreturngain_algo1,'tradingnum_algo1:',tradingnum_algo1
        print 'benchmark:',bars['bench'][-1]

        # disbars = barstmp[['bench','benchsigdiff','reversesig']]
        # disbars = disbars[disbars['benchsigdiff'] != 0]
        # disbars = disbars[-5:]
        # display(HTML(disbars.to_html()))
        print '--------------algo Result End-------------------------'    
        
        sp = pd.DataFrame(bars['Close'],index = bars.index,columns=['Price'])
        ms = pd.concat([sp,signals],axis=1).fillna(0.0)
        ms['newsignals'] = bars['benchsig_kr']
        # display(HTML(ms[['newsignals']].to_html()))
        # initial_capital = bars['Close'][0] *110
        initial_capital = 10000000 
        portfolio_bench = MarketOnMixedPortfolio(
            'KDAS', ms, initial_capital=initial_capital)
        
         
        returns_talgo = portfolio_bench.backtest_portfolio()
        print '--------------algo Result-------------------------'    
        
        barstmp = deepcopy(bars)
        
        totalreturngain_algo1 = 0.0
        barstmp['reversesig'] = (barstmp['benchsigdiff_kr']*-1)
        barstmp['calbenchsig'] = barstmp['reversesig'] * barstmp['bench']
        for tsig,tbench in zip(barstmp['reversesig'],barstmp['bench']):
            if tsig == -1:
                buybench = tbench
                
            elif tsig == 1:
                totalreturngain_algo1 = totalreturngain_algo1 + (tbench - buybench)
                print 'buybench',buybench,' sellbench',tbench,' gain',tbench - buybench

        buybench = 0.0
        totalgain = 0.0
        stance = 'none'
        lastbuydate = 'NA'
        for day in range(len(bars)):
            if bars['benchsigdiff_kr'][day] == 1:
                buybench = bars['bench'][day]
                stance = 'hold'
                lastbuydate = str(bars.index[day])
            elif bars['benchsigdiff_kr'][day] == -1 and buybench != 0.0:
                totalgain += (bars['bench'][day] - buybench)
                stance = 'none'

        totalreturngain_algo1 = totalgain
        tradingnum_algo1 = len(bars['benchsigdiff_kr'][bars['benchsigdiff_kr'] != 0]) /2
        print 'start closep:',bars['Close'][0],'current closep:',bars['Close'][-1]
        print 'total Account Gain:', (returns_talgo.total[returns_talgo.total.index[-1]] - initial_capital)/initial_capital,'initial_capital:',initial_capital,\
                'totalReturn:',returns_talgo.total[returns_talgo.total.index[-1]]
        print 'totalAccum Gain:',returns_talgo.total.pct_change().cumsum()[returns_talgo.total.pct_change().cumsum().index[-1]],\
               'tradingGain',totalreturngain_algo1,'tradingnum_algo1:',tradingnum_algo1
        print 'benchmark:',bars['bench'][-1]
        print 'price',bars['Close'][-1]
        holding_num = int(initial_capital/bars['Close'][-1])
        print 'holding num',holding_num
        disbars = barstmp[['bench','benchsigdiff_kr','reversesig','calbenchsig']]
        disbars = disbars[disbars['benchsigdiff_kr'] != 0]
        # disbars = disbars[-5:]
        display(HTML(disbars.to_html()))
        traderesult = disbars
        print '--------------algo Result End-------------------------'      

        # global barstmp_ret
        # barstmp_ret = barstmp

        
        elapsed = (time.clock() - start)
        print 'caseAdjust elapsed time:',elapsed
    except Exception,e:
        print 'algo result error ',e
        stcore.PrintException()     
        pass           

    try:
        if writedblog == 'none':
            # bars = bars.fillna(0.0)

            fig = plt.figure(figsize=(20, 30))
            ax1 = fig.add_subplot(811,  ylabel='Return Percentage')
            ax1.plot(returns_bench.index,
                      returns_bench['total'].pct_change().cumsum(),
                       color='r', lw=2.)
            ax1.plot(returns_bench.ix[returns_bench['buysignal'] == 100].index,
              returns_bench['total'].pct_change().cumsum()[returns_bench['buysignal'] == 100],
              '^', markersize=10, color='r')
            ax1.plot(returns_bench.ix[returns_bench['sellsignal'] == -100].index,
              returns_bench['total'].pct_change().cumsum()[returns_bench['sellsignal'] == -100],
              'v', markersize=10, color='b')

            ax1 = fig.add_subplot(812,  ylabel='Benchmark')
            ax1.plot(bars.index,
                      bars['Close'].pct_change().cumsum(),
                       color='r', lw=2.)
            ax1.plot(bars['bench'].ix[returns_bench['buysignal'] == 100].index,
              bars['bench'][returns_bench['buysignal'] == 100],
              '^', markersize=10, color='r')
            ax1.plot(bars['bench'].ix[returns_bench['sellsignal'] == -100].index,
              bars['bench'][returns_bench['sellsignal'] == -100],
              'v', markersize=10, color='b')

            ax1 = fig.add_subplot(813,  ylabel='Benchmark_algo')
            ax1.plot(bars.index,
                      bars['Close'].pct_change().cumsum(),
                       color='r', lw=2.)
            ax1.plot(bars['bench'].ix[returns_talgo['buysignal'] == 100].index,
              bars['bench'][returns_talgo['buysignal'] == 100],
              '^', markersize=10, color='r')
            ax1.plot(bars['bench'].ix[returns_talgo['sellsignal'] == -100].index,
              bars['bench'][returns_talgo['sellsignal'] == -100],
              'v', markersize=10, color='b')

            ax1 = fig.add_subplot(814,  ylabel='Return Percentage algo')
            ax1.plot(returns_talgo.index,
                      returns_talgo['total'].pct_change().cumsum(),
                       color='r', lw=2.)
            ax1.plot(returns_talgo.ix[returns_talgo['buysignal'] == 100].index,
              returns_talgo['total'].pct_change().cumsum()[returns_talgo['buysignal'] == 100],
              '^', markersize=10, color='r')
            ax1.plot(returns_talgo.ix[returns_talgo['sellsignal'] == -100].index,
              returns_talgo['total'].pct_change().cumsum()[returns_talgo['sellsignal'] == -100],
              'v', markersize=10, color='b')

            bars['bench_short'] = 0
            bars['bench_long'] = 0
            bars['bench_short'] =  pd.ewma(bars['bench'],span=5)
            bars['bench_long'] =  pd.ewma(bars['bench'],span=7)
            
            # disbars = bars[['bench','bench_long','bench_short']]

            # display(HTML(disbars.to_html()))
            ax1 = fig.add_subplot(815,  ylabel='moving average')
            if algomode == 'Sunny':
                # bars[['bench_short','bench_long']].plot(ax=ax1, lw=2.)
                ax1.plot(bars.index,bars[['bench_short','bench_long']])
            elif algomode == 'Hyanggu':
                # bars[['bench_short','bench_long','lpf','hpf']].plot(ax=ax1, lw=2.)
                ax1.plot(bars.index,bars[['bench_short','bench_long','lpf','hpf']])

            ax1 = fig.add_subplot(816,  ylabel='volume pct')
            # bars[['Volume_pct']].plot(ax=ax1, lw=2.)
            ax1.plot(bars.index,bars[['Volume_pct']])

            ax1 = fig.add_subplot(817,  ylabel='MMSignals')
            ax1.plot(bars.index,
                      bars['bench'],
                       color='black', lw=2.)
            ax1.plot(bars['bench'].ix[bars['MMSignals'] == 1].index,
              bars['bench'][bars['MMSignals'] == 1],
              '^', markersize=10, color='r')
            ax1.plot(bars['bench'].ix[bars['MMSignals'] == 0].index,
              bars['bench'][bars['MMSignals'] == 0],
              '^', markersize=10, color='b')

            # ax1 = fig.add_subplot(818,  ylabel='')
            # bars[['bench_sig_sl_short_std','bench_sig_sl_long_std']].plot(ax=ax1, lw=2.)



    except Exception,e:
        print 'plot error:',e                
        stcore.PrintException()                
        pass

    
    return  codearg,namearg\
           ,returns_talgo.total.pct_change().cumsum()[returns_talgo.total.pct_change().cumsum().index[-1]]\
           ,totalreturngain_algo1\
           ,tradingnum_algo1\
           ,stance,bars['bench'][-1]\
           ,returns_talgo.total[returns_talgo.total.index[-1]]\
           ,lastbuydate\
           ,bars['Close'][-1],holding_num\
           

def RunSimul_Kalman_gain(codearg,typearg,namearg,mode,dbmode,histmode,runcount,srcsite,writedblog,updbpattern,appenddb,startdatemode 
    ,dbtradinghist,plotly,*args):

    print 'RunSimul_Kalman_gain inside'
    start = time.clock()
    if typearg == 4:# DOW,NASDAQ,S&P500
        startdate= '2014-01-01'
        arg_index = 0
        stdarg = args[arg_index]
        arg_index += 1
        smallarg = args[arg_index]
        arg_index += 1
        dayselect = args[arg_index]
        arg_index += 1
        tangentmode = args[arg_index]
        print 'stdarg',stdarg,'tangentmode',tangentmode
        tangentmode = 'tan_gen'
        if namearg == 'dow':
            symbol = 'GOOG/INDEXDJX_DJI'
            bars =  Quandl.get(symbol, collapse='Daily', trim_start=startdate, trim_end=datetime.today(),authtoken="")
        elif namearg == 'nasdaq':
            symbol = '^IXIC'
            import pandas.io.data as web
            bars = web.get_data_yahoo(symbol,startdate)
        elif namearg == 'sandp':
            symbol = '^GSPC'
            import pandas.io.data as web
            bars = web.get_data_yahoo(symbol,startdate)

        if mode =='dbpattern'  or dbmode == 'dbpattern':        
            if updbpattern == 'none':
                print 'read DB patterns'
                patternAr, extractid= stcore.ReadPatternsFromDB(codearg,typearg,namearg,mode)
                patternAppendAr = stcore.ReadPatternsAppendFromDB(codearg,namearg)
            elif updbpattern == 'updbpattern':
                print 'read UP DB patterns'
                patternAr, extractid= stcore.ReadUpPatternsFromDB(codearg,typearg,namearg,mode)

            if patternAr == -1:
                print 'real time gen db pattern'
                startdate = '2011-01-01'
                dbmode  = 'none'


    else:    
        bars,patternAr, extractid,patternAppendAr,bars_25p,bars_50p\
        ,bars_75p,bars_90p,tangent_25p,tangent_50p,tangentmode\
        ,startdate,dbmode,stdarg,smallarg,dayselect,tangentmode =\
            stcore._inRunSimul_FetchData(codearg,typearg,namearg,mode,dbmode,histmode,runcount,srcsite,writedblog,updbpattern,appenddb,startdatemode \
                            ,dbtradinghist,plotly,*args)
        # print 'stdarg',stdarg,'smallarg',smallarg,'dayselect',dayselect,'tangentmode',tangentmode
        
    
    # bars = bars.drop(bars.index[-1])    
    
    today  = datetime.today()
    todayweek = today.weekday()

    bars['week'] = bars.index.weekday
    tailweekdays = bars['week'][-5:]
    # print tailweekdays

    # if 0 <= todayweek <=4 :
    #     for cnt in range(0,len(tailweekdays)):
    #         day = cnt +1 
    #         # print gbars2['week'][-1*day]
    #         checkday = bars['week'][-1*day]
            
    #         # print todayweek,checkday,bars.index[-1*day]
    #         if todayweek != checkday:
    #             raise Exception("week check error")
    #         if todayweek == 0:
    #             todayweek = 4
    #         else:
    #             todayweek = todayweek - 1

    # bars = bars[:len(bars)-100]  
    # bars = bars[:'2015-07-07']
    print bars.tail()  
    '''
    test code for inflection point
    '''
    # bars = bars.drop(bars.tail(240*6).index)
    # bars = bars.drop(bars.tail(20*2+5).index)
    # print '----------final test bars---------'
    # print bars.tail()    
    '''
    end test code
    '''

    mac1 = stcore.MovingAverageCrossStrategy(
    'test', bars, short_window=5, long_window=20)
    signals = mac1.generate_signals()
    # # print signals.tail()
    # portfolio = MarketOnClosePortfolio(
    #     'test', bars, signals, initial_capital=3000000.0)
     
    # returns = portfolio.backtest_portfolio()


    MA1 = 5
    MA2 = 30
    MA3 = 60

    
    # """
    # print pdvol1.ix[pdvol1.longsignal ==1]
    
    
    
    bars['bench'] = bars['Close'].pct_change().cumsum()
    bars['benchdiff'] = bars['bench'].diff()

    # stance = 'none'

    newtradesig = np.zeros(len(bars['Close']))

    dmani._genStdsigs(bars)    
    # print 'std bars gen end'        
    
    try:
        bars2 = deepcopy(bars)
        
        # bars2 = bars2.astype(np.float64,copy = False)
        # barsStdnp = (bars2['Std'].values + bars2['Std'].values) /(bars2['Avg'].values-bars2['Std'].values)

        bars2['Std'] = bars2['Std'].astype(np.float64)
        bars2['Avg'] = bars2['Avg'].astype(np.float64)
        barsStddf = bars2['Stdsig'] = (bars2['Std'] + bars2['Std']) /(bars2['Avg']-bars2['Std'])
        
        # bars_25p = bars2['Stdsig'].describe()['25%']
        # bars_50p = bars2['Stdsig'].describe()['50%']
        # bars_75p = bars2['Stdsig'].describe()['75%']
        # bars_90p = bars2['Stdsig'].quantile(0.9)
        
        if stdarg == 'generate':    
            bars_25p = bars2['Stdsig'][:day].describe()['25%']
            # bars_50p = bars2['Stdsig'][:day].describe()['50%']
            bars_50p = bars2['Stdsig'][:day].quantile(0.5)
            bars_75p = bars2['Stdsig'][:day].describe()['75%']
            # bars_75p = bars2['Stdsig'][:day].quantile(0.9)
            bars_90p = bars2['Stdsig'][:day].quantile(0.9)
            print 'std bars real time not from DB'
    except Exception,e:
        print 'bars2 stdarg gen error',e
    # print 'bars 25%:',bars_25p,'bars_50p:',bars_50p,'bars 75%:',bars_75p,'bars_90p:',bars_90p
    # print '----bars std sig----'
    # print barsStddf
    # print 'obv gen start2'      
    
     

    """
    kalman filter
    """
    #"""
    try:
        
        # barsMAranges = dmani._rangeDivide(bars)    
        # dmani._genVolwithBenchdiff(bars,barsMAranges)
        # dmani._genVolwithStdsig(bars)

        initial_state_mean_in = bars['Close'][0]
        observations = bars['Close'].values

        # ukf = UnscentedKalmanFilter(lambda x, w: x + np.sin(w), lambda x, v: x + v, transition_covariance=0.1,initial_state_mean=initial_state_mean_in)
        ukf = UnscentedKalmanFilter(tran_func, lambda x, v: x + np.sin(v), transition_covariance=0.1,initial_state_mean=initial_state_mean_in)
        (filtered_state_means, filtered_state_covariances) = ukf.filter(observations)
        (smoothed_state_means, smoothed_state_covariances) = ukf.smooth(observations)
        predictdf = pd.DataFrame({'predict':filtered_state_means[:,0],'smooth':smoothed_state_means[:,0]},index=bars.index)

        bars['predict'] = predictdf['predict']
        bars['smooth'] = predictdf['smooth']

        bars['benchsig'] =  0
        bars['benchsig'][bars['predict'] <= bars['smooth']] = 1
        bars['benchsigdiff'] = bars['benchsig'].diff()
        
        bars['Volume_pct'] = bars['Volume'].pct_change()
        # bars['Volume_pct_short'] = pd.rolling_mean(bars['Volume_pct'],window=5)
        # bars['Volume_pct_std'] = 0.0
        # bars['Volume_pct_offset'] = 0.0
        # for intday in range(0,len(bars)):
        #     if intday > 0:
        #         vol_std_mean = bars['Volume_pct'][:intday].std()
        #         bars['Volume_pct_std'][intday] = bars['Volume_pct'][:intday].std()
        #         bars['Volume_pct_offset'][intday] = bars['Volume_pct'][intday] - vol_std_mean
        
        signalnp = bars['Close'].values
        dayslim = 5
        barsdf = bars
        mmsigdf,mmsignp,maxsigdf,minsigdf,maxsignp,minsignp,maxqueue,minqueue = inflectionPoint(signalnp,dayslim,barsdf)
        bars = pd.concat([mmsigdf,bars],axis=1)
        # disbars = bars[['bench','MMSignals']]
        # display(HTML(disbars.to_html()))
        # talgodf = talgo._HGtradingalgo_verYE(bars,-1)
        algomode = args[7]
        print 'RunSimul_Kalman len(*args)',args,len(args)
        if algomode == 'Sunny':
            talgodf = talgo._HGtradingalgo_verSunny(bars,-1,*args)
        elif algomode == 'Hyanggu':
            talgodf = talgo._HGtradingalgo_verHyanggu(bars,-1,*args)
        # talgodf = talgo._HGtradingalgo_verSunny_MA(bars,-1)
        global barstmp_ret
        barstmp_ret = bars

        disbars = bars[['bench','benchsigdiff_kr']]
        disbars = disbars[disbars['benchsigdiff_kr'] != 0]
        # disbars = disbars[-5:]
        # display(HTML(disbars.to_html()))
    except Exception,e:
        print 'kalman error ',e
        pass
    #"""        
    """ kalman end"""
    
    try:
        sp = pd.DataFrame(bars['Close'],index = bars.index,columns=['Price'])
        ms = pd.concat([sp,signals],axis=1).fillna(0.0)
        ms['newsignals'] = bars['benchsig']
        # display(HTML(ms[['newsignals']].to_html()))
        # initial_capital = bars['Close'][0] *110
        initial_capital = 10000000 
        portfolio_bench = MarketOnMixedPortfolio(
            'KDAS', ms, initial_capital=initial_capital)
        
         
        returns_bench = portfolio_bench.backtest_portfolio()
        # display(HTML(returns_bench.to_html()))
        print '--------------algo Result-------------------------'    
        
        barstmp = deepcopy(bars)
        barstmp['reversesig'] = (barstmp['benchsigdiff']*-1)
        barstmp['calbenchsig'] = barstmp['reversesig'] * barstmp['bench']
        totalreturngain_algo1 = barstmp['calbenchsig'].sum()
        tradingnum_algo1 = len(bars['benchsigdiff'][bars['benchsigdiff'] != 0]) /2
        print 'start closep:',bars['Close'][0],'current closep:',bars['Close'][-1]
        print 'total Account Gain:', (returns_bench.total[returns_bench.total.index[-1]] - initial_capital)/initial_capital,'initial_capital:',initial_capital,\
                'totalReturn:',returns_bench.total[returns_bench.total.index[-1]]
        print 'totalAccum Gain:',returns_bench.total.pct_change().cumsum()[returns_bench.total.pct_change().cumsum().index[-1]],\
               'tradingGain',totalreturngain_algo1,'tradingnum_algo1:',tradingnum_algo1
        print 'benchmark:',bars['bench'][-1]

        # disbars = barstmp[['bench','benchsigdiff','reversesig']]
        # disbars = disbars[disbars['benchsigdiff'] != 0]
        # disbars = disbars[-5:]
        # display(HTML(disbars.to_html()))
        print '--------------algo Result End-------------------------'    
        
        sp = pd.DataFrame(bars['Close'],index = bars.index,columns=['Price'])
        ms = pd.concat([sp,signals],axis=1).fillna(0.0)
        ms['newsignals'] = bars['benchsig_kr']
        # display(HTML(ms[['newsignals']].to_html()))
        # initial_capital = bars['Close'][0] *110
        initial_capital = 10000000 
        portfolio_bench = MarketOnMixedPortfolio(
            'KDAS', ms, initial_capital=initial_capital)
        
         
        returns_talgo = portfolio_bench.backtest_portfolio()
        print '--------------algo Result-------------------------'    
        
        barstmp = deepcopy(bars)
        
        totalreturngain_algo1 = 0.0
        barstmp['reversesig'] = (barstmp['benchsigdiff_kr']*-1)
        barstmp['calbenchsig'] = barstmp['reversesig'] * barstmp['bench']
        for tsig,tbench in zip(barstmp['reversesig'],barstmp['bench']):
            if tsig == -1:
                buybench = tbench
                
            elif tsig == 1:
                totalreturngain_algo1 = totalreturngain_algo1 + (tbench - buybench)
                print 'buybench',buybench,' sellbench',tbench,' gain',tbench - buybench

        buybench = 0.0
        totalgain = 0.0
        stance = 'none'
        lastbuydate = 'NA'
        for day in range(len(bars)):
            if bars['benchsigdiff_kr'][day] == 1:
                buybench = bars['bench'][day]
                stance = 'hold'
                lastbuydate = str(bars.index[day])
            elif bars['benchsigdiff_kr'][day] == -1 and buybench != 0.0:
                totalgain += (bars['bench'][day] - buybench)
                stance = 'none'

        totalreturngain_algo1 = totalgain
        tradingnum_algo1 = len(bars['benchsigdiff_kr'][bars['benchsigdiff_kr'] != 0]) /2
        print 'start closep:',bars['Close'][0],'current closep:',bars['Close'][-1]
        print 'total Account Gain:', (returns_talgo.total[returns_talgo.total.index[-1]] - initial_capital)/initial_capital,'initial_capital:',initial_capital,\
                'totalReturn:',returns_talgo.total[returns_talgo.total.index[-1]]
        print 'totalAccum Gain:',returns_talgo.total.pct_change().cumsum()[returns_talgo.total.pct_change().cumsum().index[-1]],\
               'tradingGain',totalreturngain_algo1,'tradingnum_algo1:',tradingnum_algo1
        print 'benchmark:',bars['bench'][-1]
        print 'price',bars['Close'][-1]
        holding_num = int(initial_capital/bars['Close'][-1])
        print 'holding num',holding_num
        disbars = barstmp[['bench','benchsigdiff_kr','reversesig','calbenchsig']]
        disbars = disbars[disbars['benchsigdiff_kr'] != 0]
        # disbars = disbars[-5:]
        display(HTML(disbars.to_html()))
        traderesult = disbars
        print '--------------algo Result End-------------------------'      

        # global barstmp_ret
        # barstmp_ret = barstmp

        
        elapsed = (time.clock() - start)
        print 'caseAdjust elapsed time:',elapsed
    except Exception,e:
        print 'algo result error ',e
        stcore.PrintException()     
        pass           

    
    
    return  codearg,namearg\
           ,returns_talgo.total.pct_change().cumsum()[returns_talgo.total.pct_change().cumsum().index[-1]]\
           ,totalreturngain_algo1\
           ,tradingnum_algo1\
           ,stance,bars['bench'][-1]\
           ,returns_talgo.total[returns_talgo.total.index[-1]]\
           ,lastbuydate\
           ,bars['Close'][-1],holding_num\
           ,traderesult


def RunSimul_Kalman_FromXing(bars,codearg,namearg,*args):

    print 'RunSimul_Kalman_FromXing'
    arg_index = 0
    stdarg = args[arg_index]
    
    start = time.clock()
    
    
    today  = datetime.today()
    todayweek = today.weekday()

    bars['week'] = bars.index.weekday
    tailweekdays = bars['week'][-5:]
    # print tailweekdays

    # if 0 <= todayweek <=4 :
    #     for cnt in range(0,len(tailweekdays)):
    #         day = cnt +1 
    #         # print gbars2['week'][-1*day]
    #         checkday = bars['week'][-1*day]
            
    #         # print todayweek,checkday,bars.index[-1*day]
    #         if todayweek != checkday:
    #             raise Exception("week check error")
    #         if todayweek == 0:
    #             todayweek = 4
    #         else:
    #             todayweek = todayweek - 1

    # bars = bars[:len(bars)-100]  
    # bars = bars[:'2015-07-07']
    # print bars.tail()  
    '''
    test code for inflection point
    '''
    # bars = bars.drop(bars.tail(240*6).index)
    # bars = bars.drop(bars.tail(20*2+5).index)
    # print '----------final test bars---------'
    # print bars.tail()    
    '''
    end test code
    '''
    # display(HTML(bars.tail(7).to_html()))

    mac1 = stcore.MovingAverageCrossStrategy(
    'test', bars, short_window=5, long_window=20)
    signals = mac1.generate_signals()
    # # print signals.tail()
    # portfolio = MarketOnClosePortfolio(
    #     'test', bars, signals, initial_capital=3000000.0)
     
    # returns = portfolio.backtest_portfolio()


    MA1 = 5
    MA2 = 30
    MA3 = 60

    
    # """
    # print pdvol1.ix[pdvol1.longsignal ==1]
    
    
    
    bars['bench'] = bars['Close'].pct_change().cumsum()
    bars['benchdiff'] = bars['bench'].diff()

    # stance = 'none'

    newtradesig = np.zeros(len(bars['Close']))

    dmani._genStdsigs(bars)    
    # print 'std bars gen end'        
    
    try:
        bars2 = deepcopy(bars)
        
        # bars2 = bars2.astype(np.float64,copy = False)
        # barsStdnp = (bars2['Std'].values + bars2['Std'].values) /(bars2['Avg'].values-bars2['Std'].values)

        bars2['Std'] = bars2['Std'].astype(np.float64)
        bars2['Avg'] = bars2['Avg'].astype(np.float64)
        barsStddf = bars2['Stdsig'] = (bars2['Std'] + bars2['Std']) /(bars2['Avg']-bars2['Std'])
        
        # bars_25p = bars2['Stdsig'].describe()['25%']
        # bars_50p = bars2['Stdsig'].describe()['50%']
        # bars_75p = bars2['Stdsig'].describe()['75%']
        # bars_90p = bars2['Stdsig'].quantile(0.9)
        
        if stdarg == 'generate':    
            bars_25p = bars2['Stdsig'][:day].describe()['25%']
            # bars_50p = bars2['Stdsig'][:day].describe()['50%']
            bars_50p = bars2['Stdsig'][:day].quantile(0.5)
            bars_75p = bars2['Stdsig'][:day].describe()['75%']
            # bars_75p = bars2['Stdsig'][:day].quantile(0.9)
            bars_90p = bars2['Stdsig'][:day].quantile(0.9)
            print 'std bars real time not from DB'
    except Exception,e:
        print 'bars2 stdarg gen error',e
    # print 'bars 25%:',bars_25p,'bars_50p:',bars_50p,'bars 75%:',bars_75p,'bars_90p:',bars_90p
    # print '----bars std sig----'
    # print barsStddf
    # print 'obv gen start2'      
    
     

    """
    kalman filter
    """
    #"""
    try:
        
        # barsMAranges = dmani._rangeDivide(bars)    
        # dmani._genVolwithBenchdiff(bars,barsMAranges)
        # dmani._genVolwithStdsig(bars)

        initial_state_mean_in = bars['Close'][0]
        observations = bars['Close'].values

        # ukf = UnscentedKalmanFilter(lambda x, w: x + np.sin(w), lambda x, v: x + v, transition_covariance=0.1,initial_state_mean=initial_state_mean_in)
        ukf = UnscentedKalmanFilter(tran_func, lambda x, v: x + np.sin(v), transition_covariance=0.1,initial_state_mean=initial_state_mean_in)
        (filtered_state_means, filtered_state_covariances) = ukf.filter(observations)
        (smoothed_state_means, smoothed_state_covariances) = ukf.smooth(observations)
        predictdf = pd.DataFrame({'predict':filtered_state_means[:,0],'smooth':smoothed_state_means[:,0]},index=bars.index)

        bars['predict'] = predictdf['predict']
        bars['smooth'] = predictdf['smooth']

        bars['benchsig'] =  0
        bars['benchsig'][bars['predict'] <= bars['smooth']] = 1
        bars['benchsigdiff'] = bars['benchsig'].diff()
        
        bars['Volume_pct'] = bars['Volume'].pct_change()
        # bars['Volume_pct_short'] = pd.rolling_mean(bars['Volume_pct'],window=5)
        # bars['Volume_pct_std'] = 0.0
        # bars['Volume_pct_offset'] = 0.0
        # for intday in range(0,len(bars)):
        #     if intday > 0:
        #         vol_std_mean = bars['Volume_pct'][:intday].std()
        #         bars['Volume_pct_std'][intday] = bars['Volume_pct'][:intday].std()
        #         bars['Volume_pct_offset'][intday] = bars['Volume_pct'][intday] - vol_std_mean
        
        signalnp = bars['Close'].values
        dayslim = 5
        barsdf = bars
        mmsigdf,mmsignp,maxsigdf,minsigdf,maxsignp,minsignp,maxqueue,minqueue = inflectionPoint(signalnp,dayslim,barsdf)
        bars = pd.concat([mmsigdf,bars],axis=1)
        # disbars = bars[['bench','MMSignals']]
        # display(HTML(disbars.to_html()))
        # talgodf = talgo._HGtradingalgo_verYE(bars,-1)
        talgodf = talgo._HGtradingalgo_verSunny(bars,-1,*args)
        # talgodf = talgo._HGtradingalgo_verSunny_MA(bars,-1)
        global barstmp_ret
        barstmp_ret = bars

        disbars = bars[['bench','benchsigdiff_kr']]
        disbars = disbars[disbars['benchsigdiff_kr'] != 0]
        # disbars = disbars[-5:]
        # display(HTML(bars.to_html()))
    except Exception,e:
        print 'kalman error ',e
        pass
    #"""        
    """ kalman end"""
    
    try:
        sp = pd.DataFrame(bars['Close'],index = bars.index,columns=['Price'])
        ms = pd.concat([sp,signals],axis=1).fillna(0.0)
        ms['newsignals'] = bars['benchsig']
        # display(HTML(ms[['newsignals']].to_html()))
        # initial_capital = bars['Close'][0] *110
        initial_capital = 10000000 
        portfolio_bench = MarketOnMixedPortfolio(
            'KDAS', ms, initial_capital=initial_capital)
        
         
        returns_bench = portfolio_bench.backtest_portfolio()
        # display(HTML(returns_bench.to_html()))
        print '--------------algo Result-------------------------'    
        
        barstmp = deepcopy(bars)
        barstmp['reversesig'] = (barstmp['benchsigdiff']*-1)
        barstmp['calbenchsig'] = barstmp['reversesig'] * barstmp['bench']
        totalreturngain_algo1 = barstmp['calbenchsig'].sum()
        tradingnum_algo1 = len(bars['benchsigdiff'][bars['benchsigdiff'] != 0]) /2
        print 'start closep:',bars['Close'][0],'current closep:',bars['Close'][-1]
        print 'total Account Gain:', (returns_bench.total[returns_bench.total.index[-1]] - initial_capital)/initial_capital,'initial_capital:',initial_capital,\
                'totalReturn:',returns_bench.total[returns_bench.total.index[-1]]
        print 'totalAccum Gain:',returns_bench.total.pct_change().cumsum()[returns_bench.total.pct_change().cumsum().index[-1]],\
               'tradingGain',totalreturngain_algo1,'tradingnum_algo1:',tradingnum_algo1
        print 'benchmark:',bars['bench'][-1]

        # disbars = barstmp[['bench','benchsigdiff','reversesig']]
        # disbars = disbars[disbars['benchsigdiff'] != 0]
        # disbars = disbars[-5:]
        # display(HTML(disbars.to_html()))
        print '--------------algo Result End-------------------------'    
        
        sp = pd.DataFrame(bars['Close'],index = bars.index,columns=['Price'])
        ms = pd.concat([sp,signals],axis=1).fillna(0.0)
        ms['newsignals'] = bars['benchsig_kr']
        # display(HTML(ms[['newsignals']].to_html()))
        # initial_capital = bars['Close'][0] *110
        initial_capital = 10000000 
        portfolio_bench = MarketOnMixedPortfolio(
            'KDAS', ms, initial_capital=initial_capital)
        
         
        returns_talgo = portfolio_bench.backtest_portfolio()
        print '--------------algo Result-------------------------'    
        
        barstmp = deepcopy(bars)
        
        totalreturngain_algo1 = 0.0
        barstmp['reversesig'] = (barstmp['benchsigdiff_kr']*-1)
        barstmp['calbenchsig'] = barstmp['reversesig'] * barstmp['bench']
        for tsig,tbench in zip(barstmp['reversesig'],barstmp['bench']):
            if tsig == -1:
                buybench = tbench
                
            elif tsig == 1:
                totalreturngain_algo1 = totalreturngain_algo1 + (tbench - buybench)
                print 'buybench',buybench,' sellbench',tbench,' gain',tbench - buybench

        buybench = 0.0
        totalgain = 0.0
        stance = 'none'
        lastbuydate = 'NA'
        for day in range(len(bars)):
            if bars['benchsigdiff_kr'][day] == 1:
                buybench = bars['bench'][day]
                stance = 'hold'
                lastbuydate = str(bars.index[day])
            elif bars['benchsigdiff_kr'][day] == -1 and buybench != 0.0:
                totalgain += (bars['bench'][day] - buybench)
                stance = 'none'

        totalreturngain_algo1 = totalgain
        tradingnum_algo1 = len(bars['benchsigdiff_kr'][bars['benchsigdiff_kr'] != 0]) /2
        print 'start closep:',bars['Close'][0],'current closep:',bars['Close'][-1]
        print 'total Account Gain:', (returns_talgo.total[returns_talgo.total.index[-1]] - initial_capital)/initial_capital,'initial_capital:',initial_capital,\
                'totalReturn:',returns_talgo.total[returns_talgo.total.index[-1]]
        print 'totalAccum Gain:',returns_talgo.total.pct_change().cumsum()[returns_talgo.total.pct_change().cumsum().index[-1]],\
               'tradingGain',totalreturngain_algo1,'tradingnum_algo1:',tradingnum_algo1
        print 'benchmark:',bars['bench'][-1]
        print 'price',bars['Close'][-1]
        holding_num = int(initial_capital/bars['Close'][-1])
        print 'holding num',holding_num
        disbars = barstmp[['bench','benchsigdiff_kr','reversesig','calbenchsig']]
        disbars = disbars[disbars['benchsigdiff_kr'] != 0]
        # disbars = disbars[-5:]
        display(HTML(disbars.to_html()))
        traderesult = disbars
        print '--------------algo Result End-------------------------'      

        # global barstmp_ret
        # barstmp_ret = barstmp

        
        elapsed = (time.clock() - start)
        print 'caseAdjust elapsed time:',elapsed
    except Exception,e:
        print 'algo result error ',e
        stcore.PrintException()     
        pass           

    
    
    return  codearg,namearg\
           ,returns_talgo.total.pct_change().cumsum()[returns_talgo.total.pct_change().cumsum().index[-1]]\
           ,totalreturngain_algo1\
           ,tradingnum_algo1\
           ,stance,bars['bench'][-1]\
           ,returns_talgo.total[returns_talgo.total.index[-1]]\
           ,lastbuydate\
           ,bars['Close'][-1],holding_num\
           ,traderesult



def Kospi_RunSimul(fetch_date,current_date,tradestart_day,display_date,algo_mode,seltype):
   
    startdatemode = 3
    dbtradinghist = 'none'
    histmode = 'histdb'
    plotly = 'plotly'
    stdmode = 'stddb'
    tangentmode = 'tangentdb'        
    daych  =0
    runcount = 0
    srcsite = 1#google
    # srcsite = 2#yahoo
    writedblog = 'none'
    updbpattern = 'none'
    appenddb = 'none'

    # code = '000000'
    # name = 'kospi'
    # markettype = 3

    code = '069500'
    name = 'KODEX200'
    markettype = 1
    try:
        codevar,namevar\
        ,accumGain\
        ,tradingGain\
        ,tradingNum\
        ,tradingStance\
        ,lastbench\
        ,totalmoney\
        ,lastbuydate\
        ,price,holding_num\
        ,traderesult\
         = RunSimul_Kalman_gain(str(code),markettype,name,'realtime','dbpattern',histmode,runcount,srcsite,writedblog,updbpattern\
                    ,appenddb,startdatemode,\
                     dbtradinghist,plotly,stdmode,'none',daych,tangentmode\
                     ,fetch_date,current_date,tradestart_day,algo_mode,seltype)
        # clear_output()
        return traderesult
    except:
        stcore.PrintException()
        pass   


def Dow_RunSimul(fetch_date,current_date,tradestart_day,display_date,algo_mode):
   
    startdatemode = 3
    dbtradinghist = 'none'
    histmode = 'histdb'
    plotly = 'plotly'
    stdmode = 'stddb'
    tangentmode = 'tangentdb'        
    daych  =0
    runcount = 0
    srcsite = 1#google
    # srcsite = 2#yahoo
    writedblog = 'none'
    updbpattern = 'none'
    appenddb = 'none'

    # code = '000000'
    # name = 'kospi'
    # markettype = 3

    # code = '069500'
    # name = 'KODEX200'
    code = 'GOOG/INDEXDJX_DJI'
    code = code.split('/')[1]
    name = 'dow'

    markettype = 1
    try:
        codevar,namevar\
        ,accumGain\
        ,tradingGain\
        ,tradingNum\
        ,tradingStance\
        ,lastbench\
        ,totalmoney\
        ,lastbuydate\
        ,price,holding_num\
        ,traderesult\
         = RunSimul_Kalman_gain(code,markettype,name,'realtime','dbpattern',histmode,runcount,srcsite,writedblog,updbpattern\
                    ,appenddb,startdatemode,\
                     dbtradinghist,plotly,stdmode,'none',daych,tangentmode\
                     ,fetch_date,current_date,tradestart_day,algo_mode,'nas_index')
        # clear_output()
        return traderesult
    except:
        stcore.PrintException()
        pass   



