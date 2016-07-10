global gmaxsigdf
gmaxsigdf 
global gmaxsignp
gmaxsignp 
global gminsignp
gminsignp 
global gminsigdf
gminsigdf 
global gbars
global gmaxqueue
global gminqueue
print len(gminsigdf),len(gminsignp)
# print gminsigdf.index[9]
day = 60
# print gminsignp[0:day]
npidx = np.where(gminsignp[0:day] == 1)
# print npidx
# print npidx[0]
dfidx = gminsigdf.index[npidx]
# print dfidx,type(dfidx)
# print len(gbars['Close'][dfidx])
# print gbars['Close'][dfidx][0],gbars['Close'][dfidx][1],gbars['Close'][dfidx][2]
# print gbars['Close'][dfidx]

import statsmodels.api as sm
p0 = gbars['Close'][dfidx].values
p1 = sm.add_constant(npidx[0], prepend=False)
slope, intercept = sm.OLS(p0, p1).fit().params
print slope,intercept

signalnp = gbars['Close'].values
maxsignp2 = np.zeros(len(signalnp))
minsignp2 = np.zeros(len(signalnp))

for maxcnt in range(gmaxqueue.size()):
    maxsignp2[gmaxqueue.getValue(maxcnt)] = 1

maxsigdf2 = pd.DataFrame(maxsignp2,index = gbars.index,columns=['MAXSignals']).fillna(0.0)
maxsigdf2.index.name = 'Date'    


for mincnt in range(gminqueue.size()):
    minsignp2[gminqueue.getValue(mincnt)] = 1

   
minsigdf2 = pd.DataFrame(minsignp2,index = gbars.index,columns=['MINSignals']).fillna(0.0)
minsigdf2.index.name = 'Date'    

# print minsigdf2[-20:],maxsigdf2[-20:]
minmaxdf = pd.concat([maxsigdf2,minsigdf2,gbars],axis=1).fillna(0.0)
print minmaxdf[0:30]
day =30

# print minmaxdf[minmaxdf['MAXSignals']==2][0:minmaxdf.index[day]]

#found low level inflection point
def getMinMaxPoint(searchtype,fromday,today,mincount,maxcount,minmaxdf,signaldf):
#     searchtype = 1 #1: max 2:min
#     mincnt = 1
#     maxcnt = 0
    mincnt = mincount
    maxcnt = maxcount
    minpoint = 0
    maxpoint = 0
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

        
# print minmaxdf.index[day]    


# p0 = [9,7,5,3,1]
# p1 = sm.add_constant([1,2,3,4,5], prepend=False)
# slope, intercept = sm.OLS(p0, p1).fit().params
# print slope,intercept,np.arctan(slope)*57.3

signaldf2 = pd.DataFrame(signalnp,index = gbars.index,columns=['Value']).fillna(0.0)
signaldf2.index.name = 'Date'    


# print signaldf2.tail()
searchtype=2
fromday = 100
curday=200
mincount=0
maxcount=0
signaldf =signaldf2

# day1,value1 = getMinMaxPoint(searchtype,curday,mincount,maxcount,minmaxdf,signaldf)
# print day1,value1,minmaxdf.index[day]

# mincount=1
# day2,value2 = getMinMaxPoint(searchtype,curday,mincount,maxcount,minmaxdf,signaldf)
# print day2,value2,minmaxdf.index[day]

# slope = (value2-value1)/(day2-day1)
# print np.arctan(slope)*57.3

day0 = []
value0 = []
totalmincount = len(minmaxdf[minmaxdf['MINSignals']==1][0:minmaxdf.index[curday]])
print 'totalmincount:',totalmincount
rangemincount = 0
for mincount in range(0,totalmincount):
    day,value = getMinMaxPoint(searchtype,fromday,curday,mincount,maxcount,minmaxdf,signaldf)
    if not day == -1 and not value == -1:
        day0.append(day)
        value0.append(value)
        rangemincount = rangemincount +1

print day0
print value0
print 'rangemincount:',rangemincount
p0 = value0
p1 = sm.add_constant(day0, prepend=False)
slope, intercept = sm.OLS(p0, p1).fit().params
print slope,',',intercept,'slope:',np.arctan(slope)*57.3
