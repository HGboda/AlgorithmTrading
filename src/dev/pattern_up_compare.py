from IPython.html.widgets import interact, interactive, fixed
from IPython.html import widgets
from IPython.display import clear_output, display, HTML




global gselectedpattern
gselectedpattern = []
global gselectedpatternLen
gselectedpatternLen = 0


global gpatternAr
global patterntotallen
patterntotallen = len(gpatternAr)

global gcurday
gcurday =0
global v2
v2 =0
global targetlen
targetlen = 0

def caseAdjust():
    global targetnp

    stopcnt = 0
    stance = 'none'
    for numcnt in range(len(targetnp)):
        if targetnp[numcnt] == 1 :
            if stance == 'none':
                stance = 'holding'
                stopcnt += 1
        elif targetnp[numcnt] == 0 and stance == 'holding':
            stance = 'none'
    global targetlen            
    targetlen= stopcnt            
    
def patternCompare(num =0):

    global targetnp

    stopcnt = 0
    numstopcnt = num +1
    currentday = 0
    stance = 'none'
    for numcnt in range(len(targetnp)):
        if targetnp[numcnt] == 1 :
            if stance == 'none':
                stance = 'holding'
                stopcnt += 1
        elif targetnp[numcnt] == 0 and stance == 'holding':
            stance = 'none'

        if stopcnt == numstopcnt:
            currentday = numcnt
            break    
        
    global gcurday
    gcurday = currentday            
    print 'pattern Compare gcurday:',gcurday
    patternSelect()

        
def patternSelect():
    global gpatternAr
    lpatternAr = gpatternAr

    global gbarsdf
    
    global gselectedpattern

    curpat = gbarsdf.reset_index()
    global gcurday
    print 'patternSelect gcurday:',gcurday

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

    global gextractid

    selectpattern = []
    for patternnum in gextractid:

        pattern0 = lpatternAr[patternnum].patterndf.reset_index()
        
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

        corrsum = (closecorr+opencorr+highcorr+lowcorr+volcorr)/5
        print 'corrsum:',corrsum
        if corrsum > 0.8 :
            selectpattern.append(lpatternAr[patternnum]) 
            print 'found corrsum:',corrsum

    if len(selectpattern) == 0:
        print 'pattern not found'

    gselectedpattern = selectpattern        
    
    global gselectedpatternLen
    if len(gselectedpattern) == 0:
        gselectedpatternLen =1
    else:
        gselectedpatternLen = len(gselectedpattern)   
        if gselectedpatternLen > 1:
            gselectedpatternLen = gselectedpatternLen -1
    # print 'gselectedpattern len:',gselectedpatternLen

    global v2
    v2.close()
    v2 = interactive(patternAnalysis, patternnum=(0,gselectedpatternLen), f2=(0,1))
    display(v2)             




def patternAnalysis(patternnum=0,showaccum = 0):
    # global gpatternAr
    global gselectedpattern
    lpatternAr = gselectedpattern#gpatternAr

   

    global targetdf 
    global gbarsdf

    # print lpatternAr[num].patterndf
    print(patternnum,showaccum)

    #''' plot disable
    fig = plt.figure(figsize=(10, 5))
    fig.patch.set_facecolor('white')     # Set the outer colour to white

    global gcurday
    print 'gcurday:',gcurday
    global gselectedpatternLen    
    
    print 'gselectedpattern len:',gselectedpatternLen+1
    curpat = gbarsdf.reset_index()

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
    

    ax1 = fig.add_subplot(221,  ylabel='current pattern')
    closex1['ClosePc'].plot(ax=ax1, color='black', lw=2.)
    openx1['OpenPc'].plot(ax=ax1, color='red', lw=2.)
    highx1['HighPc'].plot(ax=ax1, color='blue', lw=2.)
    lowx1['LowPc'].plot(ax=ax1, color='#EF15C3', lw=2.)

    ax2 = fig.add_subplot(222,  ylabel='volume')
    volx1['VolPc'].plot(ax=ax2, color='#2EFE64', lw=2.)

    if len(gselectedpattern) == 0:
        print 'pattern not found'
        if gcurday+ 5 < len(gbarsdf):
            # print targetdf[gcurday-9:gcurday+2]            
            print gbarsdf[gcurday-9:gcurday+5]            
        else:
            # print targetdf[gcurday-9:gcurday+(len(targetdf)-gcurday)]            
            print gbarsdf[gcurday-9:gcurday+(len(targetdf)-gcurday)]            
        return

    pattern0 = lpatternAr[patternnum].patterndf.reset_index()
    
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
    

    
    ax3 = fig.add_subplot(223,  ylabel='selected pattern')

    # # Plot the AAPL closing price overlaid with the moving averages
    patternall['ClosePc'].plot(ax=ax3, color='black', lw=2.,label='ClosePc')
    patternall['OpenPc'].plot(ax=ax3, color='red', lw=2.,label='OpenPc')
    patternall['HighPc'].plot(ax=ax3, color='blue', lw=2.,label='HighPc')
    patternall['LowPc'].plot(ax=ax3, color='#EF15C3', lw=2.,label='LowPc')
    ax3.legend(loc = 2,bbox_to_anchor=(0.2, 1.5)).get_frame().set_alpha(0.5)
    ax4 = fig.add_subplot(224,  ylabel='volume')
    patternall['VolPc'].plot(ax=ax4, color='#2EFE64', lw=2.)


    
    
    closecorr = closex1['ClosePc'].corr(closepc0['ClosePc'])
    opencorr = openx1['OpenPc'].corr(openpc0['OpenPc'])
    highcorr = highx1['HighPc'].corr(highpc0['HighPc'])
    lowcorr = lowx1['LowPc'].corr(lowpc0['LowPc'])
    volcorr = volx1['VolPc'].corr(volpc0['VolPc'])

    plt.show()

    print 'close corr:',closecorr,' open corr:',opencorr,' high corr:',highcorr,' low corr:',lowcorr,' vol corr:',volcorr
    print patternall
    
    # print gbarsdf[gcurday-9:gcurday+1]
    
    if gcurday+ 5 < len(gbarsdf):
        # print targetdf[gcurday-9:gcurday+2]            
        print gbarsdf[gcurday-9:gcurday+5]            
    else:
        # print targetdf[gcurday-9:gcurday+(len(targetdf)-gcurday)]            
        print gbarsdf[gcurday-9:gcurday+(len(targetdf)-gcurday)]            

    


caseAdjust()

v1 = interactive(patternCompare, num=(0,targetlen-1))
display(v1)    


v2 = interactive(patternAnalysis, patternnum=(0,gselectedpatternLen), f2=(0,1))
display(v2)    
