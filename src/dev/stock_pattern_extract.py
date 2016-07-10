from IPython.html.widgets import interact, interactive, fixed
from IPython.html import widgets
from IPython.display import clear_output, display, HTML
from copy import deepcopy

global goodcasedf
global fixcasedf
global gbarsdf
global gpatternAr

case1 = goodcasedf
case2 = fixcasedf
global targetlen
targetlen = 0
global targetdf 
targetdf = 0

global gselectedpattern
gselectedpattern = []
global gselectedpatternLen
gselectedpatternLen = 0


global gpatternAr
global patterntotallen
patterntotallen = len(gpatternAr)

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


def caseAdjust(goodcasedf,fixcasedf):
    print 'case adjust'
    targetcasedf = pd.concat([goodcasedf,fixcasedf],axis = 1)
    # print goodcasedf.tail()
    # print fixcasedf.tail()
    
    targetcasedf['targetCol'] = targetcasedf['GoodMMSignals'] != targetcasedf['MMSignals'] 
    global gbarsdf
    targetcasedf['Gain'] = (gbarsdf['Close'].pct_change().cumsum()).fillna(0.0)
    
    targetcasedf['targetCol2'] = targetcasedf['targetCol']

    # print targetcasedf[-150:-100]
    global targetlen
    targetlen1 = len(targetcasedf[targetcasedf['targetCol'] == True])
    # targetlen = targetlen1
    targetlen2 = len(targetcasedf[targetcasedf['targetCol'] == False])
    totallen = len(targetcasedf)
    # print totallen,targetlen1,targetlen2,targetlen1+targetlen2
    # len(targetcasedf[targetcasedf['targetCol2'] == True]),len(targetcasedf[targetcasedf['targetCol2'] == False])

    global gpatternAr
    lpatternAr = gpatternAr
    
    global patterntotallen

    for numcnt in range(len(targetcasedf)):
        if targetcasedf['targetCol'][numcnt] == True:
            gcurday = numcnt

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


            for patternnum in range(patterntotallen):

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
                if corrsum > 0.8 :
                    targetcasedf['targetCol2'][numcnt] = True
                    break
                else:
                    targetcasedf['targetCol2'][numcnt] = False
                    
    
    targetlen1 = len(targetcasedf[targetcasedf['targetCol2'] == True])
    targetlen = targetlen1
    print 'targetlen:',targetlen

    global targetdf 
    targetdf = targetcasedf


global gcurday
gcurday =0


global allselectpattern
allselectpattern = []
def patternAllRun():
    global targetdf 
    # global targetlen
    global gbarsdf

    global gpatternAr
    lpatternAr = gpatternAr    
    global patterntotallen

    targetpatNum = 0
    searchpatNum = 0
    missingpat = []
    for numcnt in range(len(targetdf)):
        # print targetdf['targetCol'][numcnt] 
        if targetdf['targetCol2'][numcnt] == True:
            gcurday = numcnt

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
            
            targetpatNum +=1 

            global allselectpattern
            
            targetpatFound = False
            for patternnum in range(patterntotallen):

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
                if corrsum > 0.8:
                    allselectpattern.append(PatternExtractData(lpatternAr[patternnum].patterndf,patternnum,gcurday,targetpatNum)) 
                    targetpatFound = True
            if targetpatFound == True:
                searchpatNum += 1   
            else:
                missingpat.append(targetpatNum)     

    print 'targetpatNum:',targetpatNum,'searchpatNum:',searchpatNum 
    print 'missingpat:',missingpat

global gfoundnumlist
gfoundnumlist = 0


def patternCompareAndExtract():
    global allselectpattern
    allsellen = len(allselectpattern)
    
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
    # for patternnum in range(allsellen):    
    #     print allselectpattern[patternnum].patterndf
    #     print 'patternid:',allselectpattern[patternnum].patternid
    #     print 'parentid:',allselectpattern[patternnum].parentid
    #     print 'foundnum:',allselectpattern[patternnum].getFoundCount()

    foundnumlist = []
    for patternnum in range(allsellen):    
        foundnumlist.append(allselectpattern[patternnum].getFoundCount()+1)
    print len(foundnumlist),len(allselectpattern)    
    global gfoundnumlist
    gfoundnumlist = foundnumlist

                

caseAdjust(case1,case2)
patternAllRun()
patternCompareAndExtract()



