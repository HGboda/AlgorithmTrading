%matplotlib inline


import xlrd
import xlwt
from xlutils.copy import copy 
from xlrd import open_workbook 
from xlwt import easyxf 

from IPython.html.widgets import interact, interactive, fixed
from IPython.html import widgets
from IPython.display import clear_output, display, HTML

pd.set_option('display.width',500)

global gdaychoice
gdaychoice = 20

global portdbname
portdbname = u''
def show_portdbname(portName=u''):
    global portdbname
    portdbname = portName
    print portName
    


i1 = interact(show_portdbname,Name=u'portdb')        
display(i1)



global stocktitle
stocktitle = u''
def show_title(title=u''):
    global stocktitle
    stocktitle = title
    print stocktitle
    
i2 = interact(show_title,Name=u'stock title')        
display(i2)

def analysisPort(widget):
    global portdbname
    print portdbname
     
    title = '../../data/port/'+portdbname+'.sqlite'
    con = sqlite3.connect(title)
    tbname = 'portfolio_result'
    try:
        portdf = pd_sql.read_frame("SELECT * from "+tbname, con)
    except Exception,e:
        print 'db error',e
        con.close()
    con.close()
    titles = portdf['title']
    filename = '../../Kospi_Symbols.xls'
    book = xlrd.open_workbook(filename)
    sheet = book.sheet_by_name('kospi')

    code = '069500'
    name = 'KODEX200'
    
    # histmode = 'histdb'
    histmode = 'none'
    srcsite = 1#google
    startdatemode = 2

    kodexbars = RunSimul_realData(code,1,name,'realtime','none',histmode,srcsite,startdatemode)       

    selecttitles = []
    corrs = []
    mins = []
    maxs =[]
    closemins = []
    closemaxs = []
    pctcums= []
    stdbarscums = []
    currentgains = []
    benchmarks = []
    bar50ps = []
    stds = []
    uppers = []
    lowers = []
    currentprices = []
    buydates = []
    gainmoneys = []
    for title in titles:
        try:
            for cnt in range(sheet.nrows):
                if sheet.row_values(cnt)[1] == title:
                    code = '{0:06d}'.format(int(sheet.row_values(cnt)[0]))
                    name = sheet.row_values(cnt)[1]
                    print code,name
                    break

            selectbars = RunSimul_realData(code,1,name,'realtime','none',histmode,srcsite,startdatemode)               
            selectstd = selectbars[-10:]
            selectstd['Diff'] = selectstd['Close'] - selectstd['Open']
            selectstd['pct'] = selectstd['Diff']/selectstd['Close']
            selectstd['closepct'] = selectstd['Close'].pct_change()
            mins.append(selectstd['pct'].min())
            maxs.append(selectstd['pct'].max())
            closemins.append(selectstd['closepct'].min())
            closemaxs.append(selectstd['closepct'].max())

            corrval = kodexbars['Close'].corr(selectbars['Close'])
            selecttitles.append(title)
            corrs.append(corrval)

            price = selectbars[-1:]['Close']
            buyprice = portdf['Price'][portdf['title'] == name].values[0]
            # print name,buyprice
            gain = (float(price) - float(buyprice))/float(buyprice)
            currentgains.append(gain)

            stocknum = portdf['StockNum'][portdf['title'] == name].values[0]
            summoney = stocknum *(float(price) - float(buyprice))
            gainmoneys.append(summoney)

            benchmarks.append(selectbars['Close'].pct_change().cumsum()[-1:])

            bars_25p,bars_50p,bars_75p,bars_90p = readStdargs(1,title)
            bar50ps.append(bars_50p)

            
            # upperprice = portdf['UpperPrice'][portdf[portdf['title'] == title].index] 
            # lowerprice = portdf['LowerPrice'][portdf[portdf['title'] == title].index] 
            
            # uppers.append(float(upperprice))
            # lowers.append(float(lowerprice))

            stdval = selectbars['Close'].pct_change().cumsum().std()
            stds.append(stdval)
            upperprice = float(buyprice)*(1.0+float(stdval))
            uppers.append(upperprice)
            lowerprice = float(buyprice)*(1.0-float(stdval))
            lowers.append(lowerprice)

            currentprices.append(selectbars['Close'][-1:].values[0])

            stdarg = 'stddb'
            retbars = analysisStd(selectbars,stdarg,code,name)
            retbars = retbars[-10:]
            retbars['pctcum'] = retbars['Close'].pct_change().cumsum()
            retbars['stddiff'] = retbars['Stdsig'].diff()
            retbars['stdcum'] = retbars['stddiff'].cumsum()

            pctcums.append(retbars['pctcum'][-1])
            stdbarscums.append(retbars['stdcum'][-1])

            buydates.append(portdf['Date'][portdf.title == title])
        except Exception,e:
            print 'data get error',e
            pass 
               
    corrdfs = pd.DataFrame({'title':selecttitles,'corrval':corrs,'min':mins,'max':maxs,'closemins':closemins,'closemaxs':closemaxs,'pctcum':pctcums,'stdbarscum':stdbarscums\
        ,'currentgain':currentgains,'bench':benchmarks,'bar50p':bar50ps,'stds':stds,'upperPrice':uppers,'lowerPrice':lowers,'Price':currentprices,'buydate':buydates\
        ,'gainmoney':gainmoneys}
        ,columns = ['title','upperPrice','Price','lowerPrice','bench','currentgain','corrval','min','max','closemins','closemaxs','pctcum','stdbarscum','bar50p','stds','buydate'\
        ,'gainmoney'])
    corrdfs = corrdfs.sort(['min'],ascending = False)
    clear_output()
    display(HTML(corrdfs.to_html()))
    print '-----total summary--------'
    print 'port gain sum',corrdfs.currentgain.sum()
    print 'port total gain money',corrdfs.gainmoney.sum()
    portdf['Price'] = portdf['Price'].astype(np.float64)
    portdf['StockNum'] = portdf['StockNum'].astype(np.float64)
    portdf['invest'] = portdf['Price'] * portdf['StockNum']
    # print portdf
    print 'port total invest',portdf['invest'].sum()
    print 'port total gain rate', float(corrdfs.gainmoney.sum())/float(portdf['invest'].sum())
    print 'port corr val',corrdfs['corrval'].mean()
    title = '../../data/port/'+portdbname+'.sqlite'
    
    con = sqlite3.connect(title)
    tbname = 'portfolio_sell_result'
    try:
        sellportdf = pd_sql.read_frame("SELECT * from "+tbname, con)
    except Exception,e:
        print 'db error',e
        con.close()
    con.close()
    print 'total money sum:',sellportdf['totalMoney'].sum()+corrdfs.gainmoney.sum()

    try:
        ''' ----- read analysis db-----'''
        print ' ----- read analysis db-----'
        dbname1 = '../../data/analysis/analysis_db.sqlite'
        global chbox_click3 
        print dbname1
        con1 = sqlite3.connect(dbname1)
        print 'db open ok'
        tablename = 'analysis_table'
        tablelist = []
        for cnt in range(4):
            tblname = tablename + '_'+str(cnt)
            print tblname
            tablelist.append(pd_sql.read_frame("SELECT * from "+tblname, con1))

        analysisdf = pd.concat(tablelist,ignore_index=True)

        # display(HTML(analysisdf.to_html()))    
        con1.close()

    except Exception,e:
        print e    
        con1.close()

    titles = portdf['title']    
    stdtitles = analysisdf['title']
    stddfs = pd.DataFrame().fillna(0.0)
    for title in titles:
        founddf = analysisdf[analysisdf['title'] == title]
        founddf = founddf[['title','algo2_curstd','bars_25p','bars_50p','bars_75p','bars_90p']]
        stddfs = pd.concat([stddfs,founddf])

    display(HTML(stddfs.to_html()))    



run_button1 = widgets.ButtonWidget(description="Portfolio Analysis")        
run_button1.on_click(analysisPort)
display(run_button1)


def analysisPortPattern(widget):
    global gdaychoice
    _inanalysisPortPattern(gdaychoice)





run_button2 = widgets.ButtonWidget(description="Port Pattern Aanlysis")        
run_button2.on_click(analysisPortPattern)
display(run_button2)


def analysisStd(bars,stdarg,code,title):
    print 'analysisStd'
    if stdarg == 'stddb':
        try:
            bars_25p,bars_50p,bars_75p,bars_90p = readStdargs(1,title)
        except Exception,e:
            algo_title = []
            algo_bars25p = []
            algo_bars50p = []
            algo_bars75p = []
            algo_bars90p = []

            srcsite = 1#google
            # srcsite = 2#yahoo
            runcount = 1
            # writedblog = 'writedblog'
            writedblog = 'autotest'
            # updbpattern = 'updbpattern'
            updbpattern = 'none'
            appenddb = 'appenddb'
            # appenddb = 'none'
            startdatemode = 1
            runcount = 0
            # dbtradinghist = 'dbtradinghist'
            dbtradinghist = 'none'
            # histmode = 'histdb'
            histmode = 'none'
            plotly = 'plotly'
            # stdmode = 'stddb'
            # stdmode = 'none'
            stdmode = 'generate'

            bars_25p,bars_50p,bars_75p,bars_90p = \
                RunSimul_std(code,1,title,'realtime','none',histmode,srcsite,startdatemode)
            
            algo_title.append(title)
            algo_bars25p.append(bars_25p)
            algo_bars50p.append(bars_50p)
            algo_bars75p.append(bars_75p)
            algo_bars90p.append(bars_90p)
            
            tradingresultdf = pd.DataFrame({'title':algo_title
            ,'bars_25p':algo_bars25p,'bars_50p':algo_bars50p,'bars_75p':algo_bars75p,'bars_90p':algo_bars90p\
            })        

            try:
                # todaydate = datetime.today()
                
                dbname = '../../data/analysis/analysis_std_db.sqlite'
                # if os.path.isfile(dbname):
                #     os.remove(dbname)
                con = sqlite3.connect(dbname)

                tablename = 'analysis_table'
                
                analysisdf = pd_sql.read_frame("SELECT * from "+tablename, con)
                # con.close()
                tradingresultdf = pd.concat([analysisdf, tradingresultdf])
                con.execute("DROP TABLE IF EXISTS "+tablename)
                
                pd_sql.write_frame(tradingresultdf, tablename, con)
                con.close()

                
            except Exception,e:
                print e    
                con.close()

    bars['Std'] = 0
    bars['Avg'] = 0
    bars['CumStd'] = 0
    bars['VolPrice'] = 0

    for day in range(len(bars)):
    #     print bars['Close'][day]
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
            volmean = bars['Volume'][1:day].mean()
            if len(bars['Volume'][1:day][bars['Volume'][1:day] > volmean].index) > 0 :
                bars['VolPrice'][day] = bars['Close'][bars['Volume'][1:day][bars['Volume'][1:day] > volmean].index].mean()
            else:
                bars['VolPrice'][day] = bars['Close'][day]
            
            # print volmean,bars['Volume'][day]
            # print bars.index[day]
            # print bars['Close'][bars['Volume'][:day][bars['Volume'][:day] > volmean].index]
        elif day >20:
            volmean = bars['Volume'][day-20:day].mean()
            bars['VolPrice'][day] = bars['Close'][bars['Volume'][day-20:day][bars['Volume'][day-20:day] > volmean].index].mean()    

    ''' std inflection point'''    
    bars2 = deepcopy(bars)
    
    bars2['Std'] = bars2['Std'].astype(np.float64)
    bars2['Avg'] = bars2['Avg'].astype(np.float64)
    barsStddf = bars2['Stdsig'] = (bars2['Std'] + bars2['Std']) /(bars2['Avg']-bars2['Std'])

    bars['pctcum'] = bars['Close'].pct_change().cumsum()
    bars['Stdsig'] = barsStddf
    return bars


def _inanalysisPortPattern(Daychoice = 20):
    
    global portdbname
    print portdbname
    global stocktitle
    print stocktitle 
    title = '../../data/port/'+portdbname+'.sqlite'
    con = sqlite3.connect(title)
    tbname = 'portfolio_result'
    try:
        portdf = pd_sql.read_frame("SELECT * from "+tbname, con)
    except Exception.e:
        print 'db error',e
        con.close()
    con.close()
    
    titles = portdf['title']
    filename = '../../Kospi_Symbols.xls'
    book = xlrd.open_workbook(filename)
    sheet = book.sheet_by_name('kospi')

    # histmode = 'histdb'
    histmode = 'none'
    srcsite = 1#google
    startdatemode = 3#2007-01-01
    
    closecorrs = []
    highcorrs = []
    opencorrs = []
    lowcorrs = []
    volcorrs = []
    gains10 = []
    gains20 = []
    gains30 = []
    seldates = []
    seltitles = []
    gainstds10 = []
    gainstds20 = []
    gainstds30 = []
    # for title in titles:
    title = stocktitle
    try:
        for cnt in range(sheet.nrows):
            if sheet.row_values(cnt)[1] == title:
                code = '{0:06d}'.format(int(sheet.row_values(cnt)[0]))
                name = sheet.row_values(cnt)[1]
                print code,name
                break
    
        keylimit = Daychoice

        selectbars = RunSimul_realData(code,1,name,'realtime','none',histmode,srcsite,startdatemode)               
        currentbars = selectbars[-1*keylimit:]
        currentbars['ClosePc'] = currentbars['Close'].pct_change().cumsum().fillna(0.0)
        currentbars['HighPc'] = currentbars['High'].pct_change().cumsum().fillna(0.0)
        currentbars['OpenPc'] = currentbars['Open'].pct_change().cumsum().fillna(0.0)
        currentbars['LowPc'] = currentbars['Low'].pct_change().cumsum().fillna(0.0)
        currentbars['VolPc'] = currentbars['Volume'].pct_change().cumsum().fillna(0.0)
        currentbars = currentbars.reset_index()
        
        # print len(currentbars)
        sbarlen = len(selectbars)
        
        

        for day in range(sbarlen,0,-1):
            if 10+keylimit < day < sbarlen - 30:
                comparebars = selectbars[day-10-keylimit:day-10]
                comparebars['ClosePc'] = comparebars['Close'].pct_change().cumsum().fillna(0.0)
                comparebars['HighPc'] = comparebars['High'].pct_change().cumsum().fillna(0.0)
                comparebars['OpenPc'] = comparebars['Open'].pct_change().cumsum().fillna(0.0)
                comparebars['LowPc'] = comparebars['Low'].pct_change().cumsum().fillna(0.0)
                comparebars['VolPc'] = comparebars['Volume'].pct_change().cumsum().fillna(0.0)
                comparebars = comparebars.reset_index()
                # print comparebars
                
                closecorr = currentbars['ClosePc'].corr(comparebars['ClosePc'])
                highcorr = currentbars['HighPc'].corr(comparebars['HighPc'])
                opencorr = currentbars['OpenPc'].corr(comparebars['OpenPc'])
                lowcorr = currentbars['LowPc'].corr(comparebars['LowPc'])
                volcorr = currentbars['VolPc'].corr(comparebars['VolPc'])
                seldate = comparebars.ix[19,['Date']].values[0]
                gain = selectbars[day-10:day]['Close'].pct_change().cumsum()[-1:].values[0]
                gainstd = selectbars[day-10:day]['Close'].pct_change().std()
                gains10.append(gain)
                gainstds10.append(gainstd)
                gain = selectbars[day-10:day+10]['Close'].pct_change().cumsum()[-1:].values[0]
                gainstd = selectbars[day-10:day+10]['Close'].pct_change().std()
                gains20.append(gain)
                gainstds20.append(gainstd)
                gain = selectbars[day-10:day+20]['Close'].pct_change().cumsum()[-1:].values[0]
                gainstd = selectbars[day-10:day+20]['Close'].pct_change().std()
                gains30.append(gain)
                gainstds30.append(gainstd)

                seltitles.append(title)
                closecorrs.append(closecorr)
                highcorrs.append(highcorr)
                opencorrs.append(opencorr)
                lowcorrs.append(lowcorr)
                volcorrs.append(volcorr)
                seldates.append(seldate)
                


                # print 'selectbars',selectbars[day-10:day]
                # print closecorr,volcorr,comparebars.ix[19,['Date']].values[0],gain


    except Exception,e:
        print 'error :',e
    clear_output()
    seldf = pd.DataFrame({'title':seltitles,'closecorr':closecorrs,'highcorr':highcorrs,'opencorr':opencorrs,'lowcorr':lowcorrs,'volcorr':volcorrs,'Date':seldates\
        ,'gain10':gains10\
        ,'gainstd10':gainstds10,'gain20':gains20,'gainstd20':gainstds20,'gain30':gains30,'gainstd30':gainstds30},\
            columns =['title','closecorr','highcorr','opencorr','lowcorr','volcorr','Date','gain10','gainstd10','gain20','gainstd20','gain30','gainstd30'])        
    corrlimit = seldf['closecorr'].quantile(0.8)
    seldf = seldf[seldf['closecorr'] >= corrlimit]
    highlimit = seldf['highcorr'].quantile(0.6)
    seldf = seldf[seldf['highcorr'] >= highlimit]
    openlimit = seldf['opencorr'].quantile(0.6)
    seldf = seldf[seldf['opencorr'] >= openlimit]
    lowlimit = seldf['lowcorr'].quantile(0.6)
    seldf = seldf[seldf['lowcorr'] >= lowlimit]
    vollimit = seldf['volcorr'].quantile(0.6)
    seldf = seldf[seldf['volcorr'] >= vollimit]
    seldf = seldf.sort(['closecorr'],ascending = False)
    # limit = len(seldf)*0.1
    # seldf = seldf[:limit]
    

    print 'closecorr:',seldf['closecorr'].mean(),'highcorr:',seldf['highcorr'].mean(),'opencorr:',seldf['opencorr'].mean(),'lowcorr:',seldf['lowcorr'].mean()\
        ,'volcorr:',seldf['volcorr'].mean()\
        ,'gain10:',seldf['gain10'].mean()\
        ,'gain20:',seldf['gain20'].mean()\
        ,'gain30:',seldf['gain30'].mean()
    display(HTML(seldf.to_html()))



def setwindowDay(Daychoice = 20):
    global gdaychoice
    gdaychoice = Daychoice
    


global i1
i1 = interactive(setwindowDay,Daychoice=(0,100))
# i2 = interact(show_tables,Table=(0,tablelen))        
display(i1)
