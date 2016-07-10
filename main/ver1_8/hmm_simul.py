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

%matplotlib inline
pd.set_option('display.width',500)


def stkHMM(lrndata, n_components):
    model = lrn.GaussianHMM(n_components, covariance_type="tied", n_iter=20)
    model.fit([lrndata])

    hidden_states = model.predict(lrndata)
    return [model, hidden_states]


    

global bars
bars = 0
global barsMAranges
barsMAranges = 0
global gfilterbars
gfilterbars = 0
def setup():
    global bars
    bars = pd.DataFrame(pd.read_csv('bars.csv'))
    bars['A'] = 0
    bars['B'] = 0
    bars['C'] = 0
    bars['D'] = 0
    bars['E'] = 0
    
    # bars.Date =[ datetime.strptime(tdate, '%Y-%m-%d %H:%M:%S')  for tdate in bars.Date ]

    global barsMAranges
    barsMAranges = pd.DataFrame(pd.read_csv('barsMAranges.csv'))
    global gfilterbars
    gfilterbars = pd.DataFrame(pd.read_csv('gfilterbars.csv'))
    # display(HTML(barsMAranges.to_html()))

daywindow = 550

def show_args(Code= 30):

    global bars
    global barsMAranges
    global gfilterbars
    
    
    try:
        # fig1 = plt.figure(figsize=(20, 3))
        # ax1 = fig1.add_subplot(111,  ylabel='bench')
        daywindow = Code
        day = daywindow    
        # disbars = bars['bench'][daywindow-20:daywindow+1]
        # disbars.plot(ax=ax1,lw=2)

        rangeFound = False
        for eDate in reversed(barsMAranges['eDate']):
            # print 'eDate',eDate,bars.Date[day],type(eDate) , type(bars.index[day])
            neweDate = datetime.strptime(eDate, '%Y-%m-%d %H:%M:%S')
            barsDate = datetime.strptime(bars.Date[day], '%Y-%m-%d %H:%M:%S')
            
            # print neweDate,barsDate
            if neweDate<barsDate:
                print 'range found!!',barsDate
                selrange = barsMAranges[barsMAranges['eDate'] == eDate]
                print 'selrange.index:',selrange.index[0]
                selindex = selrange.index[0]
                # selrange = selrange.reset_index()
                # print 'selrange.index ',selrange.index
                
                rangeFound = True

                break

        

        for nday in range(len(bars)):
            if eDate == bars.Date[nday]:
                lasteDateDay = nday
                break        
        

        if rangeFound == True:
                            
            scut = selrange['scut'][selindex]
            ecut = selrange['ecut'][selindex]
            esbench = selrange['eBench'][selindex] - selrange['sBench'][selindex]
            print 'esbench',esbench
            
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

        disbars = selrange[['sDate','eDate','subtstate','days']]
        display(HTML(disbars.to_html()))        

        stkdata = bars

        stkmchnlrndata = np.column_stack([stkdata['Close'][:daywindow+1], stkdata['Volume'][:daywindow+1]])
        ncomponents = 5
        lrnmodel, hiddenstates = stkHMM(stkmchnlrndata, ncomponents)
        nxtstateidx = lrnmodel.transmat_[hiddenstates[len(hiddenstates)-1], :]
        nxtstateprob = np.amax(nxtstateidx)
        nxtstate = np.argmax(nxtstateidx)

        # print 'nxtstate',nxtstate,'nxtstateprob',nxtstateprob,'nxtstateidx',nxtstateidx

        # print "means and vars of each hidden state"
        allmeans = []
        for i in xrange(ncomponents):
            print "%dth hidden state" % i,
            print 'mean = ',lrnmodel.means_[i][0],
            covars = np.diag(lrnmodel.covars_[i])
            stdvar = np.sqrt(covars[0])
            stdvarpct = stdvar/lrnmodel.means_[i][0]
            print 'var = ',stdvar, ' pct = ',stdvarpct
            print 'high price:',stdvar+lrnmodel.means_[i][0],'low price:',-stdvar+lrnmodel.means_[i][0]
            print "" 
            allmeans.append(lrnmodel.means_[i][0]) 
        print 'cur price',bars['Close'][Code],'benchdiff',bars['benchdiff'][day]  
        covars = np.diag(lrnmodel.covars_[nxtstate])
        stdvar = np.sqrt(covars[0])
        stdvarpct = stdvar/lrnmodel.means_[nxtstate][0]
        print 'nxtstate',nxtstate,'var = ',stdvar, ' pct = ',stdvarpct
        print 'nxtstate',nxtstate,'high price:',stdvar+lrnmodel.means_[nxtstate][0],'low price:',-stdvar+lrnmodel.means_[nxtstate][0]

        idx = (hiddenstates[:daywindow+1] == nxtstate)
        idx = idx[-20:]
        # print 'idx',idx
        windowbars = bars[:daywindow+1]
        # disbars = windowbars[-20:]
        # display(HTML(windowbars.to_html()))
        print 'nxtstate average',windowbars[-20:]['Close'][idx].mean(),'nxtstate min',windowbars[-20:]['Close'][idx].min()\
                ,'nxtstate max',windowbars[-20:]['Close'][idx].max()
        print len(windowbars)                
        for nday in range(len(windowbars[-20:])):
            # print nday,windowbars['Close'][len(windowbars)-20+nday]
            if windowbars[-20:]['Close'][idx].max() == windowbars['Close'][len(windowbars)-20+nday]:
                print 'nxtstate max day ',windowbars.Date[len(windowbars)-20+nday],windowbars['Close'][len(windowbars)-20+nday]
                print 'nxtstate max day ',len(windowbars)-20+nday
                break
        # print windowbars['Close'][-20:]
        print 'nxtstate max last day',day - (len(windowbars)-20+nday)
        print 'range last days:',day-lasteDateDay
        rangegain = bars['Close'][lasteDateDay:day+1].pct_change().cumsum()[-1:]
        print 'range pct gain:',rangegain

        
        print 'filter_hmm_real:',gfilterbars['filter_hmm_real'][day]
        
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
        if i == 0:
            print 'current state 0'
        elif i == 1:
            print 'current state 1'
        elif i == 2:
            print 'current state 2'
        elif i == 3:
            print 'current state 3'
        elif i == 4:
            print 'current state 4'
        elif i == 5:
            print 'current state 5'    

        for i in range(0,ncomponents):
            # print allmeans[i]
            if int(lrnmodel.means_[hiddenstates[Code-1]][0]) == int(allmeans[i]):
                # print 'previous allmeans[i]',allmeans[i],i
                break    
        if i == 0:
            print 'previous state 0'
        elif i == 1:
            print 'previous state 1'
        elif i == 2:
            print 'previous state 2'
        elif i == 3:
            print 'previous state 3'
        elif i == 4:
            print 'previous state 4'
        elif i == 5:
            print 'previous state 5'    

        sameStatecnt = 0
        for rstate in reversed(hiddenstates):
            if rstate == nxtstate:
                sameStatecnt +=1
            else:
                break
        print 'sameStatecnt:',sameStatecnt

        print 'range sameStatecnt gain:',bars['Close'][day-sameStatecnt:day+1].pct_change().cumsum()[-1:]

        fig1 = plt.figure(figsize=(15, 3))
        ax1 = fig1.add_subplot(111,  ylabel='hmm')
        disbars = bars[:daywindow+1]
        disbars['Close'].plot(ax=ax1,lw=2)
        for i in xrange(ncomponents):
            idx = (hiddenstates == i)
            ax1.plot(disbars.index[idx], disbars['Close'][idx], 'o')

        # fig1 = plt.figure(figsize=(20, 3))
        # ax1 = fig1.add_subplot(111,  ylabel='A')
        # disbars = bars['A'][30:daywindow+1]
        # disbars.plot(ax=ax1,lw=2,label='A')

        # disbars = bars['B'][30:daywindow+1]
        # disbars.plot(ax=ax1,lw=2,label='B')

        # disbars = bars['C'][30:daywindow+1]
        # disbars.plot(ax=ax1,lw=2,label='C')

        # disbars = bars['D'][30:daywindow+1]
        # disbars.plot(ax=ax1,lw=2,label='D')

        # disbars = bars['E'][30:daywindow+1]
        # disbars.plot(ax=ax1,lw=2,label='E')
        # # print 'A',bars['A'][Code]    
        # ax1.legend(["A", "B", "C","D","E"],loc=2);
        
        plt.show()
    except Exception,e:
        PrintException()

setup()
i = interact(show_args,Code=(1,daywindow-1))

# if __name__ == "__main__":
#     setup()


def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print 'EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj)
    