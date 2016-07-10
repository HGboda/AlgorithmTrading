# -*- coding: utf-8 -*-
from stockcore import *
from tradingalgo import *
from data_mani import *
import stockcore as stcore
import tradingalgo as talgo 
import data_mani as dmani

from IPython.html.widgets import interact, interactive, fixed
from IPython.html import widgets
from IPython.display import clear_output, display, HTML

global gbarMAranges
gbarMAranges = 0
def _genStdsigs(bars):
    print '_genStdsigs'

    bars['OpenCloseSig'] = 0
    bars['OpenCloseSig'] = bars['Close']- bars['Open']
    # bars2['OpenCloseSig'] = np.where(bars2['OpenCloseSig'] > 0.0, 1.0, -1.0)  
    bars['OpenCloseSig'][bars['OpenCloseSig'] > 0.0] = 1.0
    bars['OpenCloseSig'][bars['OpenCloseSig'] <= 0.0] = -1.0

    bars['OBV'] = bars['Volume']*bars['OpenCloseSig']
    bars['OBV'] = bars['OBV'].cumsum()
    bars['OBVdiff'] = bars['OBV'].diff()

    bars['Std'] = 0
    bars['Avg'] = 0
    bars['CumStd'] = 0
    bars['VolPrice'] = 0
    
    bars['HLSig'] = 0
    bars['HLSig2'] = 0
    for day in range(len(bars)):
    #     print bars['Close'][day]
        try:
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
                volmean = bars['Volume'][1:day].std()
                if len(bars['Volume'][1:day][bars['Volume'][1:day] > volmean].index) > 0 :
                    bars['VolPrice'][day] = bars['Close'][bars['Volume'][1:day][bars['Volume'][1:day] > volmean].index].mean()
                else:
                    bars['VolPrice'][day] = bars['Close'][day]
                
                
                # print volmean,bars['Volume'][day]
                # print bars.index[day]
                # print bars['Close'][bars['Volume'][:day][bars['Volume'][:day] > volmean].index]
            elif day >20:
                volmean = bars['Volume'][day-20:day].std()
                bars['VolPrice'][day] = bars['Close'][bars['Volume'][day-20:day][bars['Volume'][day-20:day] > volmean].index].mean()    

                
                
        except Exception,e:
            print 'std error1 ',e

    bars['StdLowP'] = bars['Avg'] - bars['Std']
    bars['StdHighP'] = bars['Avg'] + bars['Std']
    bars['StdLowBench'] =  bars['bench'] - (bars['Close'] - bars['StdLowP'])/bars['StdLowP'] 
    bars['StdHighBench'] = (bars['StdHighP'] - bars['Close'])/bars['Close'] + bars['bench']
    bars['StdHighBenchDiff'] = 0.0
    bars['StdHighBenchDiff'] = bars['StdHighBench'] - bars['bench']
    bars['StdLowBenchDiff'] = 0.0
    bars['StdLowBenchDiff'] = bars['bench'] - bars['StdLowBench']

    Stdtmp = bars['Std'].astype(np.float64)
    Avgtmp = bars['Avg'].astype(np.float64)
    bars['Stdsig'] = (Stdtmp + Stdtmp) /(Avgtmp-Stdtmp)
    
    for day in range(len(bars)):
        if day > 1:
            
            if bars['OBVdiff'][day] > 0.0 :
                bars['HLSig'][day]  = -1.0
            else:
                bars['HLSig'][day] = 1.0                

            
            if bars['benchdiff'][day] > 0.0:    
                bars['HLSig2'][day]  = -1.0
            else:
                bars['HLSig2'][day] = 1.0                    
        

    bars['VolPriceLow'] = 0
    bars['VolPriceHigh'] = 0
    try:
        for day in range(len(bars)):
            if day >= 20:

                if len(bars['HLSig'][day-20:day][bars['HLSig'][day-20:day] == -1]) > 1:
                    bars['VolPriceLow'][day] = bars['Close'][bars['HLSig'][day-20:day][bars['HLSig'][day-20:day] == -1].index].min()    \

                else:
                    if bars['VolPriceLow'][day-1] != 0:
                        bars['VolPriceLow'][day] = (bars['VolPriceLow'][day-1]+ bars['Close'][day-1])/2.0
                    else:
                        bars['VolPriceLow'][day] = bars['Close'][day-1]
                    
                if len(bars['HLSig'][day-20:day][bars['HLSig'][day-20:day] == 1]) > 1:
                    bars['VolPriceHigh'][day] = bars['Close'][bars['HLSig'][day-20:day][bars['HLSig'][day-20:day] == 1].index].max()    
                else:
                    if bars['VolPriceHigh'][day-1] != 0:
                        bars['VolPriceHigh'][day] = (bars['VolPriceHigh'][day-1] + bars['Close'][day-1])/2.0
                    else:
                        bars['VolPriceHigh'][day] = bars['Close'][day-1]
            else:
                bars['VolPriceLow'][day] = bars['Close'][day]
                bars['VolPriceHigh'][day] = bars['Close'][day]
    except Exception,e:
        print 'std error2 ',e        
    
    bars['VolPriceHigh'] = (bars['VolPriceHigh'] + bars['VolPriceLow'])/2.0
    
    """    
    fig1 = plt.figure(figsize=(20, 10))
    ax1 = fig1.add_subplot(211,  ylabel='HLSig')
    xvollow = bars.ix[bars['HLSig'] == -1].index
    xvolhigh = bars.ix[bars['HLSig'] == 1].index
    yvollowprice = bars['Close'][bars['HLSig'] == -1]
    yvolhighprice = bars['Close'][bars['HLSig'] == 1]
    bars['Close'].plot(ax=ax1,lw=2)
    ax1.plot(bars.index,bars['Close'],'o', markersize=10, color='black')
    ax1.plot(xvollow,yvollowprice,'v', markersize=10, color='b')
    ax1.plot(xvolhigh,yvolhighprice,'^', markersize=10, color='r')

    ax2 = fig1.add_subplot(212,  ylabel='HLSig2')
    xvollow = bars.index
    xvolhigh = bars.index
    yvollowprice = bars['VolPriceLow']
    yvolhighprice = bars['VolPriceHigh']
    bars['Close'].plot(ax=ax2,lw=2)
    ax2.plot(bars.index,bars['Close'],'o', markersize=10, color='black')
    # ax2.plot(xvollow,yvollowprice,'v', markersize=10, color='b')
    ax2.plot(xvolhigh,yvolhighprice,'^', markersize=10, color='r')
    bars['StdHighP'].plot(ax=ax2,lw=2)
    """
    # display(HTML(bars[['HLSig','VolPriceLow','VolPriceHigh']].to_html()))    
def _rangeDivide(bars):
    print '_rangeDivide'
    
    ma1close = pd.rolling_mean(bars['Close'],5,min_periods=1).fillna(0.0)
    ma2close = pd.rolling_mean(bars['Close'],20,min_periods=1).fillna(0.0)
    ma3close = pd.rolling_mean(bars['Close'],60,min_periods=1).fillna(0.0)
    ma1bench = pd.rolling_mean(bars['bench'],5,min_periods=1).fillna(0.0)
    ma2bench = pd.rolling_mean(bars['bench'],20,min_periods=1).fillna(0.0)
    ma3bench = pd.rolling_mean(bars['bench'],60,min_periods=1).fillna(0.0)

    bars['CloseMAsig'] = 0
    bars['CloseMAsig'][ma1close >= ma2close] = 1
    bars['CloseMAdiff'] = bars['CloseMAsig'].diff()

    bars['ma3close'] = ma3close
    bars['ma1bench'] = ma1bench
    bars['ma2bench'] = ma2bench
    bars['ma3bench'] = ma3bench
    bars['ma3closePct'] = ma3close.pct_change().cumsum()
    bars['CloseMAsig2'] = 0
    bars['CloseMAsig2'][bars['Close'] >= ma3close] = 1
    bars['CloseMAdiff2'] = bars['CloseMAsig2'].diff()

    cutranges = bars['CloseMAdiff'][bars['CloseMAdiff'] != 0].dropna()
    cutindex = bars['CloseMAdiff'][bars['CloseMAdiff'] != 0].dropna().index


    # print cutranges[0]
    # print 'cutindex:',cutindex[:len(cutindex)]
    print len(cutranges)-1
    dfcols= ['sDate','eDate','scut','ecut','days'\
                ,'benchMin','benchMax','benchMean','benchMedian','sBench','eBench'\
                ,'vMin','vMax','vMean','vMedian','vStart','vEnd']
    try:
        MArangesdf = []
        datecnt = 0
        for cnt in range(len(cutranges)-1):

            if datecnt == len(cutranges)-1 and datecnt % 2 ==0:
                tmprangedf = bars[cutindex[datecnt]:bars.index[-1]]

                tmpdf = pd.DataFrame({'sDate':[cutindex[datecnt]],'eDate':[bars.index[-1]]\
                                       ,'scut':[cutranges[datecnt]],'ecut':[cutranges[datecnt]]\
                                       ,'days':[len(bars[:cutindex[-1]]) - len(bars[:cutindex[datecnt]]) +1]\
                                       ,'benchMin':[tmprangedf['bench'].min()]\
                                       ,'benchMax':[tmprangedf['bench'].max()]\
                                       ,'benchMean':[tmprangedf['bench'].mean()]\
                                       ,'benchMedian':[tmprangedf['bench'].median()]\
                                       ,'sBench':[tmprangedf['bench'][0]]\
                                       ,'eBench':[tmprangedf['bench'][-1]]\
                                       ,'vMin':[tmprangedf['Volume'].min()]\
                                       ,'vMax':[tmprangedf['Volume'].max()]\
                                       ,'vMean':[tmprangedf['Volume'].mean()]\
                                       ,'vMedian':[tmprangedf['Volume'].median()]\
                                       ,'vStart':[tmprangedf['Volume'][0]]\
                                       ,'vEnd':[tmprangedf['Volume'][-1]]\
                                       },columns= dfcols)
            else:
                tmprangedf = bars[cutindex[datecnt]:cutindex[datecnt+1]]
                tmpdf = pd.DataFrame({'sDate':[cutindex[datecnt]],'eDate':[cutindex[datecnt+1]]\
                                        ,'scut':[cutranges[datecnt]],'ecut':[cutranges[datecnt+1]]\
                                        ,'days':[len(bars[:cutindex[datecnt+1]]) - len(bars[:cutindex[datecnt]])+1]\
                                        ,'benchMin':[tmprangedf['bench'].min()]\
                                        ,'benchMax':[tmprangedf['bench'].max()]\
                                        ,'benchMean':[tmprangedf['bench'].mean()]\
                                        ,'benchMedian':[tmprangedf['bench'].median()]\
                                        ,'sBench':[tmprangedf['bench'][0]]\
                                        ,'eBench':[tmprangedf['bench'][-1]]\
                                        ,'vMin':[tmprangedf['Volume'].min()]\
                                        ,'vMax':[tmprangedf['Volume'].max()]\
                                        ,'vMean':[tmprangedf['Volume'].mean()]\
                                        ,'vMedian':[tmprangedf['Volume'].median()]\
                                        ,'vStart':[tmprangedf['Volume'][0]]\
                                        ,'vEnd':[tmprangedf['Volume'][-1]]\
                                        },columns= dfcols)
            datecnt +=1
            MArangesdf.append(tmpdf)
    except Exception,e:
        print 'data mani error ',e
    barsMAranges = pd.DataFrame()
    for rvalue in MArangesdf:
        # print rvalue
        barsMAranges = pd.concat([barsMAranges,rvalue],axis=0)
    
    barsMAranges = barsMAranges[dfcols]
    barsMAranges = barsMAranges.reset_index(drop=True)
    # for value in len(cutranges):
    #     if value == 1:
    # disbars = bars[:30]
    # display(HTML(disbars.to_html()))
    barsMAranges['subtstate'] = 'none'
    barsMAranges['subtstate_changed'] = 'none'
    # display(HTML(barsMAranges.to_html()))
    # barsMAranges.to_csv('barsMAranges.csv')

    global gbarMAranges
    gbarMAranges = barsMAranges
    
    return barsMAranges


def _genVolwithBenchdiff(bars,barsMAranges):
    print '_genVolwithBenchdiff'

    bars['Volbenchdiff'] = 0.0
    for day in range(len(bars)):
        if day > 0:
            negvolsum = bars['Volume'][:day][bars['benchdiff'][:day] <= 0.0].sum()
            posvolsum = bars['Volume'][:day][bars['benchdiff'][:day] > 0.0].sum()

            if len(bars[:day][bars['benchdiff'][:day] <= 0.0]) ==0:
                negvolsum = 0.0
            if len(bars[:day][bars['benchdiff'][:day] > 0.0]) ==0:
                posvolsum = 0.0
            # print 'posvolsum:',posvolsum,'negvolsum:',negvolsum    
            bars['Volbenchdiff'][day] =  posvolsum - negvolsum

    Volbenchdiff_selrange = []
    for day in range(len(bars)):
        if day > 1:
            for sDate in barsMAranges['sDate']:
                if sDate == bars.index[day]:
                    selrange = barsMAranges[barsMAranges['sDate'] == sDate]
                    selindex = selrange.index[0]
                    rangeFound = True

                    sDate = selrange['sDate'][selindex]
                    eDate = selrange['eDate'][selindex]
                    
                    range_negbenchdiff_index = bars[sDate:eDate][bars['benchdiff'][sDate:eDate] <= 0.0].index
                    range_posbenchdiff_index = bars[sDate:eDate][bars['benchdiff'][sDate:eDate] > 0.0].index
                    if len(range_negbenchdiff_index) == 0:
                        negvolsum = 0
                    else:    
                        negvolsum = bars['Volume'][sDate:eDate][range_negbenchdiff_index].sum()
                    if len(range_posbenchdiff_index) == 0:
                        posvolsum = 0
                    else:
                        posvolsum = bars['Volume'][sDate:eDate][range_posbenchdiff_index].sum()

                    Volbenchdiff_selrange.append(posvolsum - negvolsum)
                    break
            
                
        
    Volbenchdiff_selrange_df = pd.DataFrame({'VolRange_benchdiff':Volbenchdiff_selrange},index = barsMAranges.eDate)            
    bars['VolRange_benchdiff'] = Volbenchdiff_selrange_df['VolRange_benchdiff']
    bars['VolRange_benchdiff'] = bars['VolRange_benchdiff'].fillna(0.0)
    
    """
    fig1 = plt.figure(figsize=(20, 10))
    ax1 = fig1.add_subplot(311,  ylabel='volbenchdiff')
    bars['Close'].plot(ax=ax1,lw=2)
    ax2 = fig1.add_subplot(312,  ylabel='volbenchdiff')
    ax2.bar(bars['Volbenchdiff'].index,
             bars['Volbenchdiff'],
              color='r')        
    ax3 = fig1.add_subplot(313,  ylabel='volbenchdiff')
    ax3.bar(Volbenchdiff_selrange_df['Volbenchdiff'].index,
             Volbenchdiff_selrange_df['Volbenchdiff'],
              color='r')        
    plt.show()
    """

def _genVolwithStdsig(bars):
    print '_genVolwithStdsig'    

    bars['Vol_Std'] = 0.0
    bars['Vol_tmp1'] = 0.0
    bars['Vol_tmp2'] = 0.0
    
    bars['Vol_tmp1'][bars['bench'] > bars['StdHighBench']] = bars['Volume']
    bars['Vol_tmp2'][bars['bench'] < bars['StdLowBench']] = bars['Volume']
    

    bars['Vol_Std'] = bars['Vol_tmp1'] + bars['Vol_tmp2']
    bars = bars.drop('Vol_tmp1', 1)
    bars = bars.drop('Vol_tmp2', 1)

    # disbars = bars[['Volume','Vol_Std','StdHighBench','bench','StdLowBench']][:100]
    # display(HTML(disbars.to_html()))            

    # for day in range(len(bars)):
    #     if day > 1:
    #         volmean = bars['Volume'][:day].mean()
    #         if bars['StdLowBench'][day] < bars['bench'][day] < bars['StdHighBench'][day] and bars['Volume'][day] > volmean:
    #             bars['Vol_Std'][day] = bars['Volume'][day]
                # print bars['Vol_Std'][day],bars.index[day]
    
    # disbars = bars[['Volume','Vol_Std','StdHighBench','bench','StdLowBench']][50:100]
    # display(HTML(disbars.to_html()))            


