
from IPython.html.widgets import interact, interactive, fixed
from IPython.html import widgets
from IPython.display import clear_output, display, HTML


global gpatternAr
totallen = len(gpatternAr)


def patternAnalysis(num=0,showaccum = 0):
    global gpatternAr
    lpatternAr = gpatternAr

    global gbarsdf

    # print lpatternAr[num].patterndf
    print(num,showaccum)
    
    
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
    

    #''' plot disable
    fig = plt.figure(figsize=(10, 5))

    fig.patch.set_facecolor('white')     # Set the outer colour to white
    ax1 = fig.add_subplot(221,  ylabel='pattern')

    # # Plot the AAPL closing price overlaid with the moving averages
    patternall['ClosePc'].plot(ax=ax1, color='black', lw=2.,label='ClosePc')
    patternall['OpenPc'].plot(ax=ax1, color='red', lw=2.,label='OpenPc')
    patternall['HighPc'].plot(ax=ax1, color='blue', lw=2.,label='HighPc')
    patternall['LowPc'].plot(ax=ax1, color='#EF15C3', lw=2.,label='LowPc')
    ax1.legend(loc = 2,bbox_to_anchor=(0.2, 1.5)).get_frame().set_alpha(0.5)
    ax2 = fig.add_subplot(222,  ylabel='volume')
    patternall['VolPc'].plot(ax=ax2, color='#2EFE64', lw=2.)


    curpat = gbarsdf.reset_index()

    corrclosex1 = curpat['Close'][-10:].pct_change().cumsum()
    closex1 = pd.DataFrame(corrclosex1,columns=['ClosePc']).fillna(0.0)
    closex1 = closex1.reset_index()
    closecorr = closex1['ClosePc'].corr(closepc0['ClosePc'])
    
    corropenx1 = curpat['Open'][-10:].pct_change().cumsum()
    openx1 = pd.DataFrame(corropenx1,columns=['OpenPc']).fillna(0.0)
    openx1 = openx1.reset_index()
    opencorr = openx1['OpenPc'].corr(openpc0['OpenPc'])
    
    corrhighx1 = curpat['High'][-10:].pct_change().cumsum()
    highx1 = pd.DataFrame(corrhighx1,columns=['HighPc']).fillna(0.0)
    highx1 = highx1.reset_index()
    highcorr = highx1['HighPc'].corr(highpc0['HighPc'])

    corrlowx1 = curpat['Low'][-10:].pct_change().cumsum()
    lowx1 = pd.DataFrame(corrlowx1,columns=['LowPc']).fillna(0.0)
    lowx1 = lowx1.reset_index()
    lowcorr = lowx1['LowPc'].corr(lowpc0['LowPc'])

    corrvolx1 = curpat['Volume'][-10:].pct_change().cumsum()
    volx1 = pd.DataFrame(corrvolx1,columns=['VolPc']).fillna(0.0)
    volx1 = volx1.reset_index()
    volcorr = volx1['VolPc'].corr(volpc0['VolPc'])

    ax3 = fig.add_subplot(223,  ylabel='pattern')
    closex1['ClosePc'].plot(ax=ax3, color='black', lw=2.)
    openx1['OpenPc'].plot(ax=ax3, color='red', lw=2.)
    highx1['HighPc'].plot(ax=ax3, color='blue', lw=2.)
    lowx1['LowPc'].plot(ax=ax3, color='#EF15C3', lw=2.)

    ax4 = fig.add_subplot(224,  ylabel='volume')
    volx1['VolPc'].plot(ax=ax4, color='#2EFE64', lw=2.)
    plt.show()

    print 'close corr:',closecorr,' open corr:',opencorr,' high corr:',highcorr,' low corr:',lowcorr,' vol corr:',volcorr
    print patternall
    
    print gbarsdf[-10:]
    

v = interactive(patternAnalysis, num=(0,totallen-1), f2=(0,1))
display(v)    

def patternRecAuto():
    print 'pattern matching'

    global gpatternAr
    lpatternAr = gpatternAr
    totallen = len(gpatternAr)
    global gbarsdf
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
    

    for num in range(totallen):

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

        print 'close corr:',closecorr,' open corr:',opencorr,' high corr:',highcorr,' low corr:',lowcorr,' vol corr:',volcorr
        if closecorr > 0.5 \
            and volcorr > 0.5\
            and opencorr > 0.5\
            and highcorr > 0.5\
            and lowcorr > 0.5:
            print 'found pattern :',num
            break
    print 'loop end!!'    

def patternRecAuto(gbarsdf):
    print 'pattern matching'

    global gpatternAr
    lpatternAr = gpatternAr
    totallen = len(gpatternAr)
    
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
    

    for num in range(totallen):

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

        print 'close corr:',closecorr,' open corr:',opencorr,' high corr:',highcorr,' low corr:',lowcorr,' vol corr:',volcorr
        if closecorr > 0.5 \
            and volcorr > 0.5\
            and opencorr > 0.5\
            and highcorr > 0.5\
            and lowcorr > 0.5:
            print 'found pattern :',num
            break
    print 'loop end!!'    

global gbarsdf
day = 100
patternRecAuto(gbarsdf[day-10:day])
# patternRecAuto()