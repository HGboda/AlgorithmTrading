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

book = xlrd.open_workbook("../../Kospi_Symbols.xls")
sheet = book.sheet_by_name('kospi')
# length = sheet.nrows
# code = '{0:06d}'.format(int(sheet.row_values(rowcnt)[1]))
# name = sheet.row_values(rowcnt)[2]
sheet_drinkAndfood = book.sheet_by_name(u'음식료')
sheet_fiber = book.sheet_by_name(u'섬유의복')
sheet_paper = book.sheet_by_name(u'종이목재')    
sheet_chemi = book.sheet_by_name(u'화학')
sheet_drug = book.sheet_by_name(u'의약품')
sheet_metal = book.sheet_by_name(u'비금속광물')
sheet_steel = book.sheet_by_name(u'철강금속')
sheet_mecha = book.sheet_by_name(u'기계')
sheet_elec= book.sheet_by_name(u'전기전자')
sheet_medi = book.sheet_by_name(u'의료정밀')
sheet_trans = book.sheet_by_name(u'운수장비')
sheet_distri = book.sheet_by_name(u'유통')
sheet_gas = book.sheet_by_name(u'전기가스')
sheet_construc = book.sheet_by_name(u'건설')
sheet_storage = book.sheet_by_name(u'운수창고')
sheet_tele = book.sheet_by_name(u'통신')
sheet_finan = book.sheet_by_name(u'금융')
sheet_secu = book.sheet_by_name(u'증권')
sheet_insu = book.sheet_by_name(u'보험')
sheet_service = book.sheet_by_name(u'서비스')
sheet_manufac = book.sheet_by_name(u'제조')
sheet_index = book.sheet_by_name(u'인덱스')



global chbox_click 
chbox_click = False
global chbox_click2 
chbox_click2 = False
global dbslidechoice
dbslidechoice = 0

global chbox_click3 
chbox_click3 = False
import glob
global lists
lists = glob.glob("../../data/result2/*.sqlite")
# print lists[0]
global length
length = len(lists)

global dbname
global tablename
tablename =u''
dbname = u''
code = 0        
global tablelen
tablelen = 1

global i2
i2 = 0
global i3
i3 = 0
global tablelen2
tablelen2 = 1
global foundap
foundap = 0
global foundap2
foundap2 = 0

global buyap
buyap = 0
global holdingap
holdingap = 0


def analysisStd(bars,stdarg,code,title):
    print 'analysisStd'
    if stdarg == 'stddb':
        try:
            bars_25p,bars_50p,bars_75p,bars_90p = readStdargs(title)
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


def analysisCorr(dbname,selectdf):
    print 'analysisCorr'

    filename = '../../Kospi_Symbols.xls'
    book = xlrd.open_workbook(filename)
    sheet = book.sheet_by_name('kospi')

    code = '069500'
    name = 'KODEX200'
    
    # histmode = 'histdb'
    histmode = 'none'
    srcsite = 1#google
    startdatemode = 2

    bars = RunSimul_realData(code,1,name,'realtime','none',histmode,srcsite,startdatemode)       

    titles = selectdf['title']
    selecttitles = []
    corrs = []
    mins = []
    maxs =[]
    closemins = []
    closemaxs = []
    pctcums= []
    stdbarscums = []
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

            corrval = bars['Close'].corr(selectbars['Close'])
            selecttitles.append(title)
            corrs.append(corrval)

            stdarg = 'stddb'
            retbars = analysisStd(selectbars,stdarg,code,name)
            retbars = retbars[-10:]
            retbars['pctcum'] = retbars['Close'].pct_change().cumsum()
            retbars['stddiff'] = retbars['Stdsig'].diff()
            retbars['stdcum'] = retbars['stddiff'].cumsum()

            pctcums.append(retbars['pctcum'][-1])
            stdbarscums.append(retbars['stdcum'][-1])

            

        except Exception,e:
            print 'data get error',e
            pass    
    corrdfs = pd.DataFrame({'title':selecttitles,'corrval':corrs,'min':mins,'max':maxs,'closemins':closemins,'closemaxs':closemaxs,'pctcum':pctcums,'stdbarscum':stdbarscums})
    corrmean = corrdfs['corrval'].mean()
    corrdfs['corrdiff'] = abs(corrdfs['corrval'] - corrmean)
    corrdfs = corrdfs.sort(['corrdiff'],ascending = True)
    # display(HTML(corrdfs.to_html()))
    return corrdfs
        



def analysisRecentBench(dbname):
    print 'analysisRecentBench',dbname
    
    try:
        
        con1 = sqlite3.connect(dbname)

        query = "SELECT * FROM sqlite_master WHERE type='table'"

        tablesdf = pd.io.sql.read_frame(query,con1)
        tablelen = len(tablesdf['name'])
        # print 'tablesdf',tablesdf
        # print 'table length',tablelen
        tabledf = []

        titles = []
        benchmarks = []
        for cnt in range(tablelen):

            if 'benchmark' in tablesdf['name'][cnt]:
                # print tablesdf['name'][cnt]
                title =  tablesdf['name'][cnt].split('_')
                title = title[2]
                # print title

                tablename = tablesdf['name'][cnt]
                tabledf = pd.io.sql.read_frame("SELECT * from "+tablename, con1)    
                tabledates = tabledf['Date'].values

                titles.append(title)

                tabledf = pd.io.sql.read_frame("SELECT * from "+tablename, con1)    
                benchval = tabledf.Benchmark[-1:].values[0] - tabledf.Benchmark[-15:-14].values[0]
                benchmarks.append(benchval)
                # print 'bench 10day sum',title, tabledf.Benchmark[-1:].values[0] - tabledf.Benchmark[-10:-9].values[0]
                # benchmarks.append(tabledf['Benchmark'])

        bendf = pd.DataFrame({'title':titles,'benchmark':benchmarks})        
        bendf = bendf[bendf.benchmark > 0.0]
        # bendf = bendf[bendf.benchmark < 0.2]
        # display(HTML(bendf.to_html()))
        return bendf

    except Exception,e:
        print 'db error',e
        con1.close()
    con1.close()

    return -1

def analysisFailcount(dbname):
    print 'analysisFailcount',dbname
    try:
        
        con1 = sqlite3.connect(dbname)

        query = "SELECT * FROM sqlite_master WHERE type='table'"

        tablesdf = pd.io.sql.read_frame(query,con1)
        tablelen = len(tablesdf['name'])
        # print 'tablesdf',tablesdf
        # print 'table length',tablelen
        tabledf = []
        
        titles = []
        failcounts = []
        benchmarks = []
        for cnt in range(tablelen):
            try:            
                if 'signal' in tablesdf['name'][cnt]:
                    # print tablesdf['name'][cnt]
                    title =  tablesdf['name'][cnt].split('_')
                    title = title[2]
                    # print title
                    
                    tablename = tablesdf['name'][cnt]
                    tabledf = pd.io.sql.read_frame("SELECT * from "+tablename, con1)    
                    tabledates = tabledf['Date'].values
                    tabledf['failsig'] = 0
                    tabledf['failsig'] = np.where(tabledf.currentgain < 0.0,tabledf.currentgain,0)

                    titles.append(title)
                    failcounts.append(len(tabledf[tabledf.failsig < 0]))
                    
                    summarytable = tablesdf['name'][cnt]
                    summarytable = summarytable.replace('signal','summary')
                    tabledf = pd.io.sql.read_frame("SELECT * from "+summarytable, con1)    
                    
                    benchmarks.append(tabledf['Benchmark'])
            except Exception,e:
                print 'error',e
                pass        


        faildfs = pd.DataFrame({'title':titles,'failcounts':failcounts,'benchmark':benchmarks})
        faildfs = faildfs.sort(['failcounts'],ascending= True)
        faillimit = faildfs.failcounts.quantile(0.4)
        faildfs = faildfs[faildfs.failcounts < faillimit]
        # faildfs = faildfs[faildfs.benchmark > 0.0]
        # display(HTML(faildfs.to_html()))
        print 'fail total len',len(faildfs)
        return faildfs
    except Exception,e:
        print 'db error',e
        con1.close()
    con1.close()

    return -1



def analysisBuytable(buydf,dbname):
    print 'analysisBuytable'
    print 'dbname',dbname
    
    try:
        con = sqlite3.connect(dbname)
        query = "SELECT * FROM sqlite_master WHERE type='table'"
        print query
        global df
        df = pd.io.sql.read_frame(query,con)
        # print df.tail()
        tablelen = len(df['name'])
        titles = buydf['title']
        
        benchmarks = []
        curprices = [] 
        count = 0

        for title in titles:
            for row in range(tablelen):
                tabletitle = df['name'][row]
                tabletitle = tabletitle.split('_')[2]
                if df['name'][row].find('summary') != -1 and \
                    title in tabletitle and len(title) == len(tabletitle):
                    # print title,count
                    tablename = df['name'][row]
                    tabledf = pd.io.sql.read_frame("SELECT * from "+tablename, con)    
                    
                    if pd.isnull(tabledf[-1:]['Benchmark'].values) == False:                    
                        benchmarks.append(tabledf[-1:]['Benchmark'].values[0])
                    if pd.isnull(tabledf[-1:]['Current'].values) == False:                    
                        curprices.append(tabledf[-1:]['Current'].values[0])
                    count +=1        
                    break
        bendf = pd.DataFrame({'Benchmark':benchmarks,'Current':curprices})
        print 'buydf len',len(buydf),'bendf len',len(bendf)
        buydf = buydf.reset_index()
        buydf = pd.concat([buydf,bendf],axis = 1)
        hdflen = len(buydf)

        seldf = buydf[buydf['Benchmark'] > 0.0] 
        seldf = seldf[seldf['totalGain'] > 0.0]
        print 'today buy len',len(seldf)
        display(HTML(seldf.to_html()))    
        
        print 'today negative buy len',len(seldf)
        seldf = buydf[buydf['Benchmark'] < 0.0] 
        display(HTML(seldf.to_html()))    
        
        
    except Exception,e:
        print 'db error',e
        con.close()    

    




def analysisHoldingtable(holdingdf,dbname):
    print 'analysisHoldingtable'
    print 'dbname',dbname
    
    try:
        con = sqlite3.connect(dbname)
        query = "SELECT * FROM sqlite_master WHERE type='table'"
        print query
        global df
        df = pd.io.sql.read_frame(query,con)
        # print df.tail()
        tablelen = len(df['name'])
        titles = holdingdf['title']
        
        benchmarks = []
        curprices = [] 
        count = 0

        for title in titles:
            for row in range(tablelen):
                tabletitle = df['name'][row]
                tabletitle = tabletitle.split('_')[2]
                if df['name'][row].find('summary') != -1 and \
                    title in tabletitle and len(title) == len(tabletitle):
                    # print title,count
                    tablename = df['name'][row]
                    tabledf = pd.io.sql.read_frame("SELECT * from "+tablename, con)    
                    
                    if pd.isnull(tabledf[-1:]['Benchmark'].values) == False:                    
                        benchmarks.append(tabledf[-1:]['Benchmark'].values[0])
                    if pd.isnull(tabledf[-1:]['Current'].values) == False:                    
                        curprices.append(tabledf[-1:]['Current'].values[0])
                    count +=1        
                    break
        bendf = pd.DataFrame({'Benchmark':benchmarks,'Current':curprices})
        print 'holdingdf len',len(holdingdf),'bendf len',len(bendf)
        holdingdf = holdingdf.reset_index()
        holdingdf = pd.concat([holdingdf,bendf],axis = 1)
        hdflen = len(holdingdf)

        for row in range(hdflen):
            holdingdf['currentgain'][row] = (holdingdf['Current'][row]- holdingdf['Price'][row])/holdingdf['Price'][row]

        print 'current total gain sum',holdingdf['currentgain'].sum(),'holding total len',len(holdingdf)    
        seldf = holdingdf[holdingdf['currentgain'] > 0.0] 
        seldf = seldf[seldf['Benchmark'] > 0.0] 
        display(HTML(seldf.to_html()))    
        print 'current gain sum',seldf['currentgain'].sum(),'holding total len',len(seldf)    
        seldf = holdingdf[holdingdf['currentgain'] < 0.0] 
        display(HTML(seldf.to_html()))    
        print 'negative bench current gain sum',seldf['currentgain'].sum(),'holding total len',len(seldf)    

    except Exception,e:
        print 'db error',e
        con.close()    




def readAnalysisDB():
    print 'readAnalysisDB'
    try:
        ''' ----- read analysis db-----'''
        print ' ----- read analysis db-----'
        dbname = 'analysis_db.sqlite'
        con = sqlite3.connect("../../data/analysis/"+dbname)

        tablename = 'analysis_table'
        tablelist = []
        for cnt in range(4):
            tblname = tablename + '_'+str(cnt)
            print tblname
            tablelist.append(pd_sql.read_frame("SELECT * from "+tblname, con))

        analysisdf = pd.concat(tablelist,ignore_index=True)

        # display(HTML(analysisdf.to_html()))    


    except Exception,e:
        print e    
        con.close()

    con.close()

    print len(analysisdf)
    print '----summary------'
    print 'benchmark num:',len(analysisdf[analysisdf['benchmark'] > 0] ), 'percentage:', len(analysisdf[analysisdf['benchmark'] > 0] )/float(len(analysisdf))
    print 'algo1 accum num:',len(analysisdf[analysisdf['algo1_accumgain'] > 0] ), 'percentage:', len(analysisdf[analysisdf['algo1_accumgain'] > 0] )/float(len(analysisdf))
    print 'algo2 accum num:',len(analysisdf[analysisdf['algo2_accumgain'] > 0] ), 'percentage:', len(analysisdf[analysisdf['algo2_accumgain'] > 0] )/float(len(analysisdf))

    print '-----------------'

    resultdf = analysisdf[analysisdf['benchmark'] > 0.1]
    print 'banchmark > 0.1 :',len(resultdf),len(resultdf)/float(len(analysisdf))

    resultdf = analysisdf[analysisdf['algo2_accumgain'] > 0.1]
    print 'algo2_accumgain > 0.1',len(resultdf),len(resultdf)/float(len(analysisdf))

    resultdf = resultdf[resultdf['benchmark'] > 0.1]
    print 'algo2_accumgain > 0.1 and benchmark > 0.1:',len(resultdf),len(resultdf)/float(len(analysisdf))

    resultdf = analysisdf[analysisdf['algo2_tradingnum'] < 10]
    print 'algo2_tradingnum < 10 :',len(resultdf),len(resultdf)/float(len(analysisdf))

    resultdf = resultdf[resultdf['benchmark'] > 0.1]
    print 'algo2_tradingnum < 10 and benchmark > 0.1:',len(resultdf),len(resultdf)/float(len(analysisdf))

    resultdf = resultdf[resultdf['algo2_accumgain'] > 0.1]
    print 'algo2_tradingnum < 10 and benchmark > 0.1 and algo2_accumgain > 0.1:',len(resultdf),len(resultdf)/float(len(analysisdf))

    resultdf = resultdf[resultdf['algo2_curstd'] < resultdf['bars_50p'] ]
    print 'algo2_tradingnum < 10 and benchmark > 0.1 and algo2_accumgain > 0.1 and bars_50p이하:',len(resultdf),len(resultdf)/float(len(analysisdf))
        
    resultdf = resultdf[resultdf['algo2_stance'].str.contains('holding')]
    print 'algo2_tradingnum < 10 and benchmark > 0.1 and algo2_accumgain > 0.1 and bars_50p이하 and holding:',len(resultdf),len(resultdf)/float(len(analysisdf))

    display(HTML(resultdf.to_html()))    

    print '--------------std bar max ---------------'
    resultdf = analysisdf[analysisdf['algo2_curstd'] > analysisdf['bars_90p']]
    resultdf = resultdf[resultdf['benchmark'] > 0.0 ]
    display(HTML(resultdf.to_html()))    
    resultdf = analysisdf[analysisdf['algo2_curstd'] > analysisdf['bars_90p']]
    resultdf = resultdf[resultdf['benchmark'] <= 0.0 ]
    display(HTML(resultdf.to_html()))    
    print '--------------std bar max end---------------'

    print '############# std bar min ##################'
    resultdf = analysisdf[analysisdf['algo2_curstd'] < analysisdf['bars_50p']]
    resultdf = resultdf[resultdf['benchmark'] > 0.0 ]
    display(HTML(resultdf.to_html()))    
    resultdf = analysisdf[analysisdf['algo2_curstd'] < analysisdf['bars_50p']]
    resultdf = resultdf[resultdf['benchmark'] <= 0.0 ]
    display(HTML(resultdf.to_html()))    
    print '############# std bar min end ##################'
    # try:
    #     dbname = '../../data/analysis/analysis_output.sqlite'
    #     if os.path.isfile(dbname):
    #         os.remove(dbname)

    #     dbname = '../../data/analysis/analysis_output.sqlite'
    #     con = sqlite3.connect(dbname)

    #     tablename = 'analysis_output'

    #     con.execute("DROP TABLE IF EXISTS "+tablename)
        
    #     pd_sql.write_frame(resultdf, tablename, con)
    #     con.close()
    #     path = '../../data/analysis/analysis.xls'
    #     resultdf.to_excel(path, sheet_name='sheet1',index=False)
    # except Exception,e:
    #     print e
    #     con.close()


def searchsector(sectorlist,targetdf):
    

    sectordf = pd.DataFrame().fillna(0.0)

    if u'음식료' in sectorlist:
        sectordf_drink = pd.DataFrame().fillna(0.0)
        search_sheet = sheet_drinkAndfood
        for rowcnt in range(search_sheet.nrows):
            if pd.isnull(targetdf[targetdf['title'] == search_sheet.row_values(rowcnt)[1]]) == False:
                sectmpdf = pd.DataFrame({u'음식료':[search_sheet.row_values(rowcnt)[1]]}).fillna(0.0)        
                # sectordf_drink = pd.concat([sectordf, sectmpdf],index=sectordf.index)
                sectordf_drink = sectordf_drink.append(sectmpdf)
        if len(sectordf_drink) > 0:                
            sectordf_drink = sectordf_drink.reset_index(drop= True)
            sectordf = pd.concat([sectordf,sectordf_drink],axis = 1)


    if u'전기전자' in sectorlist:
        sectordf_elec = pd.DataFrame().fillna(0.0)
        search_sheet = sheet_elec
        for rowcnt in range(search_sheet.nrows):
            if pd.isnull(targetdf[targetdf['title'] == search_sheet.row_values(rowcnt)[1]]) == False:
                sectmpdf = pd.DataFrame({u'전기전자':[search_sheet.row_values(rowcnt)[1]]}).fillna(0.0)        
                # sectordf = pd.concat([sectordf, sectmpdf],index=sectordf.index)
                sectordf_elec = sectordf_elec.append(sectmpdf)
        if len(sectordf_elec) > 0:                
            sectordf_elec = sectordf_elec.reset_index(drop= True)
            sectordf = pd.concat([sectordf,sectordf_elec],axis = 1)


    if u'운수장비' in sectorlist:
        sectordf_trans = pd.DataFrame().fillna(0.0)
        search_sheet = sheet_trans
        for rowcnt in range(search_sheet.nrows):
            if pd.isnull(targetdf[targetdf['title'] == search_sheet.row_values(rowcnt)[1]]) == False:
                sectmpdf = pd.DataFrame({u'운수장비':[search_sheet.row_values(rowcnt)[1]]}).fillna(0.0)        
                # sectordf = pd.concat([sectordf, sectmpdf],index=sectordf.index)
                sectordf_trans = sectordf_trans.append(sectmpdf)
        if len(sectordf_trans) > 0:                
            sectordf_trans = sectordf_trans.reset_index(drop= True)
            sectordf = pd.concat([sectordf,sectordf_trans],axis = 1)

    if u'화학' in sectorlist:
        sectordf_chemi = pd.DataFrame().fillna(0.0)
        search_sheet = sheet_chemi
        for rowcnt in range(search_sheet.nrows):
            if pd.isnull(targetdf[targetdf['title'] == search_sheet.row_values(rowcnt)[1]]) == False:
                sectmpdf = pd.DataFrame({u'화학':[search_sheet.row_values(rowcnt)[1]]}).fillna(0.0)        
                # sectordf = pd.concat([sectordf, sectmpdf],index=sectordf.index)
                sectordf_chemi = sectordf_chemi.append(sectmpdf)
        if len(sectordf_chemi) > 0:                
            sectordf_chemi = sectordf_chemi.reset_index(drop= True)
            sectordf = pd.concat([sectordf,sectordf_chemi],axis = 1)
    
    if u'건설' in sectorlist:
        sectordf_construc = pd.DataFrame().fillna(0.0)
        search_sheet = sheet_construc
        for rowcnt in range(search_sheet.nrows):
            if pd.isnull(targetdf[targetdf['title'] == search_sheet.row_values(rowcnt)[1]]) == False:
                sectmpdf = pd.DataFrame({u'건설':[search_sheet.row_values(rowcnt)[1]]}).fillna(0.0)        
                sectordf_construc = sectordf_construc.append(sectmpdf)
        if len(sectordf_construc) > 0:                
            sectordf_construc = sectordf_construc.reset_index(drop= True)
            sectordf = pd.concat([sectordf,sectordf_construc],axis = 1)
    
    if u'금융' in sectorlist:
        sectordf_finance = pd.DataFrame().fillna(0.0)
        search_sheet = sheet_finan
        for rowcnt in range(search_sheet.nrows):
            if pd.isnull(targetdf[targetdf['title'] == search_sheet.row_values(rowcnt)[1]]) == False:
                sectmpdf = pd.DataFrame({u'금융':[search_sheet.row_values(rowcnt)[1]]}).fillna(0.0)        
                sectordf_finance = sectordf_finance.append(sectmpdf)
        if len(sectordf_finance) > 0:                
            sectordf_finance = sectordf_finance.reset_index(drop= True)
            sectordf = pd.concat([sectordf,sectordf_finance],axis = 1)

    if u'유통' in sectorlist:
        sectordf_distri = pd.DataFrame().fillna(0.0)
        search_sheet = sheet_distri
        for rowcnt in range(search_sheet.nrows):
            if pd.isnull(targetdf[targetdf['title'] == search_sheet.row_values(rowcnt)[1]]) == False:
                sectmpdf = pd.DataFrame({u'유통':[search_sheet.row_values(rowcnt)[1]]}).fillna(0.0)        
                sectordf_distri = sectordf_distri.append(sectmpdf)
        if len(sectordf_distri) > 0:                
            sectordf_distri = sectordf_distri.reset_index(drop= True)
            sectordf = pd.concat([sectordf,sectordf_distri],axis = 1)        

    if u'통신' in sectorlist:
        sectordf_tele = pd.DataFrame().fillna(0.0)
        search_sheet = sheet_tele
        for rowcnt in range(search_sheet.nrows):
            if pd.isnull(targetdf[targetdf['title'] == search_sheet.row_values(rowcnt)[1]]) == False:
                sectmpdf = pd.DataFrame({u'통신':[search_sheet.row_values(rowcnt)[1]]}).fillna(0.0)        
                sectordf_tele = sectordf_tele.append(sectmpdf)
        if len(sectordf_tele) > 0:                
            sectordf_tele = sectordf_tele.reset_index(drop= True)
            sectordf = pd.concat([sectordf,sectordf_tele],axis = 1)                    

    if u'증권' in sectorlist:
        sectordf_secu = pd.DataFrame().fillna(0.0)
        search_sheet = sheet_secu
        for rowcnt in range(search_sheet.nrows):
            if pd.isnull(targetdf[targetdf['title'] == search_sheet.row_values(rowcnt)[1]]) == False:
                sectmpdf = pd.DataFrame({u'증권':[search_sheet.row_values(rowcnt)[1]]}).fillna(0.0)        
                sectordf_secu = sectordf_secu.append(sectmpdf)
        if len(sectordf_secu) > 0:                
            sectordf_secu = sectordf_secu.reset_index(drop= True)
            sectordf = pd.concat([sectordf,sectordf_secu],axis = 1)                    

    if u'보험' in sectorlist:
        sectordf_insu = pd.DataFrame().fillna(0.0)
        search_sheet = sheet_insu
        for rowcnt in range(search_sheet.nrows):
            if pd.isnull(targetdf[targetdf['title'] == search_sheet.row_values(rowcnt)[1]]) == False:
                sectmpdf = pd.DataFrame({u'보험':[search_sheet.row_values(rowcnt)[1]]}).fillna(0.0)        
                sectordf_insu = sectordf_insu.append(sectmpdf)
        if len(sectordf_insu) > 0:                
            sectordf_insu = sectordf_insu.reset_index(drop= True)
            sectordf = pd.concat([sectordf,sectordf_insu],axis = 1)                    


    if u'철강금속' in sectorlist:
        sectordf_steel = pd.DataFrame().fillna(0.0)
        search_sheet = sheet_steel
        for rowcnt in range(search_sheet.nrows):
            if pd.isnull(targetdf[targetdf['title'] == search_sheet.row_values(rowcnt)[1]]) == False:
                sectmpdf = pd.DataFrame({u'철강금속':[search_sheet.row_values(rowcnt)[1]]}).fillna(0.0)        
                sectordf_steel = sectordf_steel.append(sectmpdf)
        if len(sectordf_steel) > 0:                
            sectordf_steel = sectordf_steel.reset_index(drop= True)
            sectordf = pd.concat([sectordf,sectordf_steel],axis = 1)                    

    if u'기계' in sectorlist:
        sectordf_mecha = pd.DataFrame().fillna(0.0)
        search_sheet = sheet_mecha
        for rowcnt in range(search_sheet.nrows):
            if pd.isnull(targetdf[targetdf['title'] == search_sheet.row_values(rowcnt)[1]]) == False:
                sectmpdf = pd.DataFrame({u'기계':[search_sheet.row_values(rowcnt)[1]]}).fillna(0.0)        
                sectordf_mecha = sectordf_mecha.append(sectmpdf)
        if len(sectordf_mecha) > 0:                
            sectordf_mecha = sectordf_mecha.reset_index(drop= True)
            sectordf = pd.concat([sectordf,sectordf_mecha],axis = 1)                    

    if u'의약품' in sectorlist:
        sectordf_drug = pd.DataFrame().fillna(0.0)
        search_sheet = sheet_drug
        for rowcnt in range(search_sheet.nrows):
            if pd.isnull(targetdf[targetdf['title'] == search_sheet.row_values(rowcnt)[1]]) == False:
                sectmpdf = pd.DataFrame({u'의약품':[search_sheet.row_values(rowcnt)[1]]}).fillna(0.0)        
                sectordf_drug = sectordf_drug.append(sectmpdf)
        if len(sectordf_drug) > 0:                
            sectordf_drug = sectordf_drug.reset_index(drop= True)
            sectordf = pd.concat([sectordf,sectordf_drug],axis = 1)                    


    if u'비금속광물' in sectorlist:
        sectordf_metal = pd.DataFrame().fillna(0.0)
        search_sheet = sheet_metal
        for rowcnt in range(search_sheet.nrows):
            if pd.isnull(targetdf[targetdf['title'] == search_sheet.row_values(rowcnt)[1]]) == False:
                sectmpdf = pd.DataFrame({u'비금속광물':[search_sheet.row_values(rowcnt)[1]]}).fillna(0.0)        
                sectordf_metal = sectordf_metal.append(sectmpdf)
        if len(sectordf_metal) > 0:                
            sectordf_metal = sectordf_metal.reset_index(drop= True)
            sectordf = pd.concat([sectordf,sectordf_metal],axis = 1)                    

    if u'운수창고' in sectorlist:
        sectordf_storage = pd.DataFrame().fillna(0.0)
        search_sheet = sheet_storage
        for rowcnt in range(search_sheet.nrows):
            if pd.isnull(targetdf[targetdf['title'] == search_sheet.row_values(rowcnt)[1]]) == False:
                sectmpdf = pd.DataFrame({u'운수창고':[search_sheet.row_values(rowcnt)[1]]}).fillna(0.0)        
                sectordf_storage = sectordf_storage.append(sectmpdf)
        if len(sectordf_storage) > 0:                
            sectordf_storage = sectordf_storage.reset_index(drop= True)
            sectordf = pd.concat([sectordf,sectordf_storage],axis = 1)                    
    
    if u'전기가스' in sectorlist:
        sectordf_gas = pd.DataFrame().fillna(0.0)
        search_sheet = sheet_gas
        for rowcnt in range(search_sheet.nrows):
            if pd.isnull(targetdf[targetdf['title'] == search_sheet.row_values(rowcnt)[1]]) == False:
                sectmpdf = pd.DataFrame({u'전기가스':[search_sheet.row_values(rowcnt)[1]]}).fillna(0.0)        
                sectordf_gas = sectordf_gas.append(sectmpdf)
        if len(sectordf_gas) > 0:                
            sectordf_gas = sectordf_gas.reset_index(drop= True)
            sectordf = pd.concat([sectordf,sectordf_gas],axis = 1)                    


    if u'서비스' in sectorlist:
        sectordf_service = pd.DataFrame().fillna(0.0)
        search_sheet = sheet_service
        for rowcnt in range(search_sheet.nrows):
            if pd.isnull(targetdf[targetdf['title'] == search_sheet.row_values(rowcnt)[1]]) == False:
                sectmpdf = pd.DataFrame({u'서비스':[search_sheet.row_values(rowcnt)[1]]}).fillna(0.0)        
                sectordf_service = sectordf_service.append(sectmpdf)
        if len(sectordf_service) > 0:                
            sectordf_service = sectordf_service.reset_index(drop= True)
            sectordf = pd.concat([sectordf,sectordf_service],axis = 1)                    
        

    if u'의료정밀' in sectorlist:
        sectordf_medi = pd.DataFrame().fillna(0.0)
        search_sheet = sheet_medi
        for rowcnt in range(search_sheet.nrows):
            if pd.isnull(targetdf[targetdf['title'] == search_sheet.row_values(rowcnt)[1]]) == False:
                sectmpdf = pd.DataFrame({u'의료정밀':[search_sheet.row_values(rowcnt)[1]]}).fillna(0.0)        
                sectordf_medi = sectordf_medi.append(sectmpdf)
        if len(sectordf_medi) > 0:            
            sectordf_medi = sectordf_medi.reset_index(drop= True)
            sectordf = pd.concat([sectordf,sectordf_medi],axis = 1)                    

    if u'섬유의복' in sectorlist:
        sectordf_fiber = pd.DataFrame().fillna(0.0)
        search_sheet = sheet_fiber
        for rowcnt in range(search_sheet.nrows):
            if pd.isnull(targetdf[targetdf['title'] == search_sheet.row_values(rowcnt)[1]]) == False:
                sectmpdf = pd.DataFrame({u'섬유의복':[search_sheet.row_values(rowcnt)[1]]}).fillna(0.0)        
                sectordf_fiber = sectordf_fiber.append(sectmpdf)
        if len(sectordf_fiber) > 0:            
            sectordf_fiber = sectordf_fiber.reset_index(drop= True)
            sectordf = pd.concat([sectordf,sectordf_fiber],axis = 1)                    

    if u'종이목재' in sectorlist:
        sectordf_paper = pd.DataFrame().fillna(0.0)
        search_sheet = sheet_paper
        for rowcnt in range(search_sheet.nrows):
            if pd.isnull(targetdf[targetdf['title'] == search_sheet.row_values(rowcnt)[1]]) == False:
                sectmpdf = pd.DataFrame({u'종이목재':[search_sheet.row_values(rowcnt)[1]]}).fillna(0.0)        
                sectordf_paper = sectordf_paper.append(sectmpdf)
        if len(sectordf_paper) > 0:            
            sectordf_paper = sectordf_paper.reset_index(drop= True)
            sectordf = pd.concat([sectordf,sectordf_paper],axis = 1)                    

    if u'인덱스' in sectorlist:
        sectordf_index = pd.DataFrame().fillna(0.0)
        search_sheet = sheet_index
        for rowcnt in range(search_sheet.nrows):
            if pd.isnull(targetdf[targetdf['title'] == search_sheet.row_values(rowcnt)[1]]) == False:
                sectmpdf = pd.DataFrame({u'인덱스':[search_sheet.row_values(rowcnt)[1]]}).fillna(0.0)        
                sectordf_index = sectordf_index.append(sectmpdf)
        if len(sectordf_index) > 0:            
            sectordf_index = sectordf_index.reset_index(drop= True)
            sectordf = pd.concat([sectordf,sectordf_index],axis = 1)                    

    # print sectordf[[u'전기전자',u'운수장비',u'화학',u'건설',u'음식료',u'통신',u'철강금속',u'비금속광물',u'금융']]
    # print sectordf[[u'증권',u'보험',u'서비스',u'전기가스',u'운수창고',u'유통',u'섬유의복',u'의약품',u'기계',u'의료정밀',u'종이목재']]

    return sectordf




def RunSimul(codearg, typearg, namearg, mode, dbmode, histmode, runcount, srcsite):

    code = codearg  # '097950'#'005930' #'005380'#009540 #036570
    if srcsite == 1:
        if typearg == 1:
            symbol = 'GOOG/KRX_' + code
        elif typearg == 2:
            symbol = 'GOOG/KOSDAQ_' + code
        elif typearg == 3:
            symbol = 'GOOG/INDEXKRX_KOSPI200'
    elif srcsite == 2:
        if typearg == 1:
            symbol = 'YAHOO/KS_' + code
        elif typearg == 2:
            symbol = 'YAHOO/KQ_' + code
        elif typearg == 3:
            symbol = 'YAHOO/INDEX_KS11'

    startdate = '2011-01-01'
    # enddate = '2008-12-30'
    print symbol, namearg
    if mode == 'realtime':
        if histmode == 'none':
            bars_org = fetchRealData(code, symbol, typearg, startdate)
        elif histmode == 'histdb':
            bars_org = fetchHistData(codearg, namearg, symbol, startdate)
    elif mode == 'dbpattern':
        bars_org = ReadHistFromDB(codearg, typearg, namearg, mode)

    if typearg == 1:
        rtsymbol = code + '.KS'
    elif typearg == 2:
        rtsymbol = code + '.KQ'
    elif typearg == 3:
        rtsymbol = '^KS11'  # '^KS200'
    # rtsymbol = '^KS200'
    realtimeURL = 'http://finance.yahoo.com/d/quotes.csv?s=' + \
        rtsymbol + '&f=sl1d1t1c1ohgv&e=.csv'
    # print realtimeURL
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
        # print 'rtsymbol:', rtsymbol, 'rtclose:', rtclose, rtdate, rttime, rtchange, 'rtopen:', rtopen, 'rthigh:', rthigh, 'rtlow:', rtlow, 'rtvolume:', rtvolume

    # print date2num(datetime.strptime(rtdate.replace('/',' '),'%m %d %Y'))

    # print bars.index[-1]
    # print date_object > bars.index[-1]
    # date_object  = date_object- dt.timedelta(days=1)
    # print date_object > bars.index[-1]
    date_object = datetime.strptime(rtdate.replace('/', ' '), '%m %d %Y')
    rtdf = pd.date_range(date_object, date_object)

    date_append = False
    # print len(bars_org),len(bars_org['Close']),len(bars_org['Volume'])
    if date_object > bars_org.index[-1]:
        d = {'Open': rtopen, 'High': rthigh, 'Low':
             rtlow, 'Close': rtclose, 'Volume': rtvolume}
        appenddf = pd.DataFrame(d, index=rtdf)
        appenddf.index.name = 'Date'
        date_append = True
        # print appenddf, date_append
        bars = pd.concat([bars_org, appenddf])
        # print '----------'
        # print bars.tail()
    else:
        bars = bars_org

    bars = bars.sort_index()
    # print '---------final bars-----------'
    # print bars.tail()
    '''
    pattern up analysis
    '''
    curday = len(bars['Close']) - 1
    # print 'today', curday
    # print bars['Close'][bars.index[curday]]

    # if dbmode == 'dbpattern':
    patternAr, extractid = ReadUpPatternsFromDB(codearg, typearg, namearg, mode)
    if patternAr == -1:
        print 'real time gen db pattern'
        signalnp = bars['Close'].values
        dayslim = 5
        barsdf = bars
        mmsigdf4,mmsignp4,maxsigdf4,minsigdf4,maxsignp4,minsignp4,maxqueue4,minqueue4 = inflectionPoint(signalnp,dayslim,barsdf)

        negsigdf,possigdf,patternAr = PatternSave(bars,mmsigdf4,mmsignp4)
        
        allselectpattern = patternAllRunUp(mmsigdf4,mmsignp4,bars,patternAr)

        foundnumlist = patternCompareAndExtract(allselectpattern)

        extractid = patternExtractCandidates(foundnumlist,allselectpattern,patternAr)

        print 'save starts !!'
        dbname = 'pattern_db_'+codearg+'_'+namearg+'.sqlite'
        con2 = sqlite3.connect("../../data/pattern/up/"+dbname)
        dblen = len(patternAr)
        tablename_base = 'result_'+codearg+'_'+namearg

        for cnt in range(dblen):
            tablename = tablename_base+'_'+str(cnt)
            # print 'writetable:',tablename
            con2.execute("DROP TABLE IF EXISTS "+tablename)
            patternAr[cnt].patterndf = patternAr[cnt].patterndf.reset_index()
            pd_sql.write_frame(patternAr[cnt].patterndf, tablename, con2)

        
        dbname = 'extractid_db_'+codearg+'_'+namearg+'.sqlite'
        con3 = sqlite3.connect("../../data/pattern/up/"+dbname)
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
        

    ispatFound = patternSelect(curday, bars, patternAr, extractid)
    return ispatFound


def readAnalysisCorrDB():
    print 'readAnalysisCorrDB'
    try:
        ''' ----- read analysis db-----'''
        print ' ----- read analysis db-----'
        dbname = 'analysis_corr_auto.sqlite'
        con = sqlite3.connect("../../data/analysis/"+dbname)

        tablename = 'corr_auto_table'
        tablelist = []
        for cnt in range(1):
            tblname = tablename + '_'+str(cnt)
            print tblname
            tablelist.append(pd_sql.read_frame("SELECT * from "+tblname, con))

        analysisdf = pd.concat(tablelist,ignore_index=True)

        # display(HTML(analysisdf.to_html()))    


    except Exception,e:
        print e    
        con.close()

    con.close()
    return analysisdf


def readAnalysisBenchDB():
    print 'readAnalysisBenchDB'
    try:
        ''' ----- read analysis bench db-----'''
        print ' ----- read analysis bench db-----'
        dbname = 'analysis_db.sqlite'
        
        con = sqlite3.connect("../../data/analysis/"+dbname)
        tablename = 'analysis_table'
        tablelist = []
        for cnt in range(4):
            tblname = tablename + '_'+str(cnt)
            print tblname
            tablelist.append(pd_sql.read_frame("SELECT * from "+tblname, con))

        analysisdf = pd.concat(tablelist,ignore_index=True)

        # display(HTML(analysisdf.to_html()))    


    except Exception,e:
        print e    
        con.close()

    con.close()
    return analysisdf


def openDB(DBchoice = 0):
    global dbslidechoice
    dbslidechoice = DBchoice
    global dbname
    global lists
    dbname = lists[DBchoice]
    print 'dbname',dbname
    con = sqlite3.connect(dbname)
    query = "SELECT * FROM sqlite_master WHERE type='table'"
    print query
    global df
    df = pd.io.sql.read_frame(query,con)
    # print df.tail()
    tablelen = len(df['name'])

    ap = pd.DataFrame().fillna(0.0)
    ap2 = pd.DataFrame().fillna(0.0)
    ap_holding = pd.DataFrame().fillna(0.0)

    selldf = pd.DataFrame().fillna(0.0)
    prevruncount = -1

    list2d=[[] for i in xrange(20)]
    list2dlen = 0
    prevtitlecnt = 0
    totalgainlist = []
    totalgainpctlist = []
    totalgainsum = 0
    totalgain = 0

    seedmoney = 5000000.0

    offsetday = 0#default = 0

    for cnt in range(tablelen):
        if df['name'][cnt].find('signal') != -1:
            # print df['name'][cnt]
            title = df['name'][cnt].split('_')
            # print 'title',title,title[2]
            # con1 = sqlite3.connect(dbname)
            global tablename
            tablename = df['name'][cnt]
            tabledf = pd.io.sql.read_frame("SELECT * from "+tablename, con)    
            # print tabledf[-1:]['BuyorSell'].values
            # print tablename,title[5],float(tabledf[-1:]['totalGain'].values)
            if pd.isnull(tabledf[-1:]['totalGain'].values) == False:
                if prevtitlecnt == 0:
                    totalgainsum = float(tabledf[-1:]['totalGain'].values)*seedmoney
                    totalgain = float(tabledf[-1:]['totalGain'].values)
                elif prevtitlecnt != 0 and prevtitlecnt == title[5]:
                    totalgainsum = totalgainsum + float(tabledf[-1:]['totalGain'].values)*seedmoney
                    # print tablename,totalgainsum,float(tabledf[-1:]['totalGain'].values)*seedmoney
                    totalgain = totalgain + float(tabledf[-1:]['totalGain'].values)

                if prevtitlecnt != 0 and prevtitlecnt != title[5]:
                    totalgainlist.append(totalgainsum)   
                    totalgainpctlist.append(totalgain) 
                    totalgainsum = 0
                    totalgain = 0
                    totalgainsum = float(tabledf[-1:]['totalGain'].values) *seedmoney
                    totalgain = float(tabledf[-1:]['totalGain'].values)

                prevtitlecnt = title[5]


            if str(tabledf[-1:]['BuyorSell'].values).find('buyprice') != -1:
                tmpstr = tabledf['Date'][-1:].values
                tabledate = datetime.strptime(tmpstr[0], '%Y-%m-%d %H:%M:%S')
                
                if tabledate > datetime.today()-timedelta(days=offsetday) - timedelta(days=1):# and \
                    # float(tabledf[-1:]['totalGain'].values) > 0.0:

                    tmpdf = deepcopy(tabledf[-1:])
                    tmpdf['title'] = pd.Series([title[2]], index=tmpdf.index)
                    tmpdf['time'] = pd.Series([title[4]+'_'+title[5]], index=tmpdf.index)
                    ap = pd.concat([ap,tmpdf])
                    # print title
                    # print title[5]
                    listindex = int(title[5]) -1
                    list2d[listindex].append(title[2]) 
                if tabledate <= datetime.today()-timedelta(days=offsetday) - timedelta(days=1):

                    tmpdf = deepcopy(tabledf[-1:])
                    tmpdf['title'] = pd.Series([title[2]], index=tmpdf.index)
                    tmpdf['time'] = pd.Series([title[4]+'_'+title[5]], index=tmpdf.index)
                    ap_holding = pd.concat([ap_holding,tmpdf])
                    
                    

            if len(tabledf) >= 2:
                tmpdf2 = deepcopy(tabledf[-2:])
                # print tmpdf2
                tmpdf2['title'] = pd.Series([title[2] ,title[2]], index=tmpdf2.index)
                name2 = title[4]+'_'+title[5]
                tmpdf2['time'] = pd.Series([name2,name2], index=tmpdf2.index)
                ap2 = pd.concat([ap2,tmpdf2])
            # print title,title[5],title[5],type(title[5]),int(title[5])
            
            if str(tabledf[-1:]['BuyorSell'].values).find('sellprice') != -1:
                todaydate = datetime.today()
                today = datetime(int(todaydate.year),int(todaydate.month),int(todaydate.day))
                startday = todaydate- timedelta(days=1)
                lowerday =datetime(int(startday.year),int(startday.month),int(startday.day))
                
                tmpstr = tabledf['Date'][-1:].values
                # print tmpstr[0]
                tabledate = datetime.strptime(tmpstr[0], '%Y-%m-%d %H:%M:%S')
                # print 'startday:',lowerday,':',today,':',tabledate
                # print 'sell title,',title[2],'tabledate:',tabledate,'lowerday:',lowerday,'today:',today    
                if tabledate > lowerday and tabledate <= today:
                    # print 'selected sell title,',title[2],'tabledate:',tabledate,'lowerday:',lowerday,'today:',today    
                    tmpdf = deepcopy(tabledf[-1:])
                    tmpdf['title'] = pd.Series([title[2]], index=tmpdf.index)
                    tmpdf['time'] = pd.Series([title[4]+'_'+title[5]], index=tmpdf.index)
                    selldf = pd.concat([selldf,tmpdf])
            
            

    # print 'duplicates:',list(duplicates)
                
    global foundap
    foundap = ap            
    global foundap2
    foundap2 = ap2

    # if len(ap) >0 :
    
    #     print foundap[['BuyorSell','Date','Price','time','title','totalGain']][-100:-50]
    #     print foundap[['BuyorSell','Date','Price','time','title','totalGain']][-50:]
    # if len(ap_holding)>0:
    #     print '--------holding----------'
    #     print ap_holding[['BuyorSell','Date','Price','time','title','totalGain']][-100:-50]
    #     print ap_holding[['BuyorSell','Date','Price','time','title','totalGain']][-50:]

    # for cnt in range(len(list2d)):
    #     if cnt == 0:
    #         duplicates = list2d[0]
    #         # print 'cnt0:',duplicates
    #     if cnt >= 1 and len(list2d[cnt]) > 0:
    #         duplicates = set(x for x in list2d[cnt] if x in duplicates)
    #         # print 'cnt >= 1',duplicates
    #     if len(list2d[cnt]) == 0:
    #         list2dlen = cnt 
    #         break

    # dulist = list(duplicates)
    # print '--------duplicates lists------------'
    # global chbox_click 
    # uplist = []
    # for title in dulist:
        
    #     if chbox_click == False:
    #         print title        
    #     else:
    #         for rowcnt in range(1,sheet.nrows):
    #             code = '{0:06d}'.format(int(sheet.row_values(rowcnt)[0]))
    #             name = sheet.row_values(rowcnt)[1]
    #             # print type(name),name,title,type(title)
    #             try:
    #                 if name == title:
    #                     # print 'matching found:', name, code
    #                     ispatFound = RunSimul(str(code), 1, name, 'realtime', 'dbpattern', 'none', 1, 1)
    #                     if ispatFound == True:
    #                         # print 'up pattern title:',title
    #                         uplist.append(title)
    #                     # break
    #             except:
    #                 pass
    #         # break
    
    # for title in uplist:
    #     print 'up pattern title:',title

    # print '--------today sell lists------------'        
    # print selldf[-100:-50]
    # print selldf[-50:]

    print '--------total Gain sum lists------------'        
    totalgainlist.append(totalgainsum)
    totalgainpctlist.append(totalgain) 
    print totalgainlist
    print totalgainpctlist

    
    sectorlist = [u'전기전자',u'운수장비',u'화학',u'건설',u'음식료',u'통신',u'철강금속',u'비금속광물',u'금융',u'서비스',\
        u'전기가스',u'운수창고',u'유통',u'섬유의복',u'의약품',u'기계',u'의료정밀',u'종이목재',u'인덱스']#, u'보험',u'증권'

    try:
        ''' ----- read std db-----'''
        print ' ----- read std db-----'
        dbname1 = '../../data/analysis/select25p.sqlite'
        con1 = sqlite3.connect(dbname1)
        tablename = 'result_25p'
        
        select25p = pd_sql.read_frame("SELECT * from "+tablename, con1)
        con1.close()

        dbname1 = '../../data/analysis/select50p.sqlite'
        con1 = sqlite3.connect(dbname1)
        tablename = 'result_50p'
        
        select50p = pd_sql.read_frame("SELECT * from "+tablename, con1)
        con1.close()

        dbname1 = '../../data/analysis/select75p.sqlite'
        con1 = sqlite3.connect(dbname1)
        tablename = 'result_75p'
        
        select75p = pd_sql.read_frame("SELECT * from "+tablename, con1)
        con1.close()

    except Exception,e:
        print e    
        con1.close()
    
    if len(ap) > 0:        
        
        global buyap
        buyap  = ap
        print '--------sector analysis buylists-----------------'
        sectordf = searchsector(sectorlist,ap)
        display(HTML(sectordf.to_html()))
        display(HTML(ap.to_html()))
        print '--------duplicate ~ 25p-----------------'
        buyaptitles = buyap['title']
        selecttitles = select25p['title']
        found25pdf = pd.DataFrame().fillna(0.0)
        for title in buyaptitles:
            for stitle in selecttitles:
                if title == stitle:
                    tmpdf = select25p[select25p['title'] == title]
                    found25pdf = pd.concat([found25pdf,tmpdf])
        display(HTML(found25pdf.to_html()))                    

        print '--------duplicate  25p ~ 50p -----------------'
        
        selecttitles = select50p['title']
        found50pdf = pd.DataFrame().fillna(0.0)
        for title in buyaptitles:
            for stitle in selecttitles:
                if title == stitle:
                    tmpdf = select50p[select50p['title'] == title]
                    found50pdf = pd.concat([found50pdf,tmpdf])
        display(HTML(found50pdf.to_html()))                    

        print '--------duplicate  50p ~ 75p -----------------'
        
        selecttitles = select75p['title']
        found75pdf = pd.DataFrame().fillna(0.0)
        for title in buyaptitles:
            for stitle in selecttitles:
                if title == stitle:
                    tmpdf = select75p[select75p['title'] == title]
                    found75pdf = pd.concat([found75pdf,tmpdf])
        display(HTML(found75pdf.to_html()))                    

    if len(ap_holding) > 0:         
        global holdingap
        holdingap  = ap_holding

        print '--------sector analysis holding lists-----------------'
        sectordf = searchsector(sectorlist,ap_holding)
        display(HTML(sectordf.to_html()))
        display(HTML(ap_holding.to_html()))
        print '--------duplicate ~ 25p-----------------'
        holdaptitles = holdingap['title']
        selecttitles = select25p['title']
        found25pdf = pd.DataFrame().fillna(0.0)
        for title in holdaptitles:
            for stitle in selecttitles:
                if title == stitle:
                    tmpdf = select25p[select25p['title'] == title]
                    found25pdf = pd.concat([found25pdf,tmpdf])
        display(HTML(found25pdf.to_html()))                    

        print '--------duplicate 25p ~ 50p -----------------'
        
        selecttitles = select50p['title']
        found50pdf = pd.DataFrame().fillna(0.0)
        for title in holdaptitles:
            for stitle in selecttitles:
                if title == stitle:
                    tmpdf = select50p[select50p['title'] == title]
                    found50pdf = pd.concat([found50pdf,tmpdf])
        display(HTML(found50pdf.to_html()))                    
        
        print '--------duplicate 50p ~ 75p -----------------'
        
        selecttitles = select75p['title']
        found75pdf = pd.DataFrame().fillna(0.0)
        for title in holdaptitles:
            for stitle in selecttitles:
                if title == stitle:
                    tmpdf = select75p[select75p['title'] == title]
                    found75pdf = pd.concat([found75pdf,tmpdf])
        display(HTML(found75pdf.to_html()))                    
    # readAnalysisDB()   
    # analysiscorrdf = readAnalysisCorrDB()
    # analysiscorrdf = analysiscorrdf[analysiscorrdf['gain'] > 0.0]
    
    # limit = analysiscorrdf['gain'].quantile(0.6)
    # analysiscorrdf = analysiscorrdf[analysiscorrdf['gain'] > limit]
    # analysiscorrdf = analysiscorrdf.sort(['gain'],ascending = False)

    # if len(ap) > 0:

    #     print '---------sector today  analysis table--------------------'
    #     analysisBuytable(ap,dbname)
    #     aptitles = ap['title']
    #     selapdf = pd.DataFrame(columns= ['title','closecorr','highcorr','opencorr','lowcorr','volcorr','gain']).fillna(0.0)
        # for title in analysiscorrdf['title']:
        #     for aptitle in aptitles:
        #         if title == aptitle:
        #             # print title
        #             tmpdf = analysiscorrdf[analysiscorrdf['title'] == title]
        #             selapdf = pd.concat([selapdf,tmpdf])
        #             break
        # display(HTML(selapdf.to_html()))
    # print '---------sector holding analysis table--------------------'
    # analysisHoldingtable(ap_holding,dbname)
    # aptitles = ap_holding['title']
    # seldf = pd.DataFrame(columns= ['title','closecorr','highcorr','opencorr','lowcorr','volcorr','gain']).fillna(0.0)
    # for title in analysiscorrdf['title']:
    #     for aptitle in aptitles:
    #         if title == aptitle:
    #             # print title
    #             tmpdf = analysiscorrdf[analysiscorrdf['title'] == title]
    #             seldf = pd.concat([seldf,tmpdf])
    #             break

    # display(HTML(seldf.to_html()))
    # print '---------fail counts analysis--------------------'
    # faildfs = analysisFailcount(dbname)

    # print '---------choice analysis--------------------'
    # if len(ap) > 0:
        
    #     print '---------buy today corr chocie--------------'
    #     global chbox_click2 
    #     if chbox_click2 == True:
    #         corrdfs_buy = analysisCorr(dbname,ap)
    #         # clear_output()
    #         # display(HTML(corrdfs_buy.to_html()))

    # if len(ap_holding)>0:
        
    #     print '---------holding corr chocie--------------'
    #     global chbox_click2 
    #     if chbox_click2 == True:
    #         corrdfs = analysisCorr(dbname,ap_holding)
    #         clear_output()
    #         if len(ap) > 0 :
    #             display(HTML(corrdfs_buy.to_html()))
    #             print 'corr mean:',corrdfs_buy['corrval'].mean()
    #             
    #         display(HTML(corrdfs.to_html()))
    #         print 'corr mean:',corrdfs['corrval'].mean()

global i1
global length
i1 = interactive(openDB,DBchoice=(0,length-1))
# i2 = interact(show_tables,Table=(0,tablelen))        
display(i1)

global stocktitle
stocktitle = u''
def show_tables(Name=u'Type'):
    print Name
    global stocktitle
    stocktitle = Name

i2 = interact(show_tables,Name=u'Type')        
display(i2)

# global stockprice
# stockprice = u''
# def show_price(Price=u''):
#     print Price
#     global stockprice
#     stockprice = Price

# i3 = interact(show_price,Price=u'')        
# display(i3)

# global investmoney
# investmoney = u''
# def show_money(Money=u''):
#     print Money
#     global investmoney
#     investmoney = Money

# i3 = interact(show_money,Money=u'')        
# display(i3)

# global portdbname
# portdbname = u''
# def show_portdbname(portName=u''):
#     print portName
#     global portdbname
#     portdbname = portName

# i4 = interact(show_portdbname,Name=u'portdb')        
# display(i4)


# global buydate
# buydate = u''
# def show_buydate(buydateIn=u''):
#     print buydateIn
#     global buydate
#     buydate = buydateIn

# i5 = interact(show_buydate,Name=u'buydateIn')        
# display(i5)


# global selldate
# selldate = u''
# def show_selldate(selldateIn=u''):
#     print selldate
#     global selldate
#     selldate = selldateIn

# i6 = interact(show_selldate,Name=u'selldateIn')        
# display(i6)

def stocksignalhist(widget):
    print 'stock signal hist'
    global df
    global dbname
    # print dbname
    global foundap
    global stocktitle
    # print foundap
    # print stocktitle
    print foundap[foundap['title'] == stocktitle]
    print '----------------------------------------'
    print foundap2[foundap2['title'] == stocktitle]
    # con1 = sqlite3.connect(dbname)
    # global tablename
    # tabledf = pd.io.sql.read_frame("SELECT * from "+tablename, con1)

    # print tabledf[-10:]


def portfoliobuy(widget):
    print 'portfolio buy'

    global df
    global dbname
    # print dbname
    global foundap
    global stocktitle
    # print foundap
    # print stocktitle
    global stockprice
    global investmoney
    
    # print stocktitle
    global buydate
    stocknum = int(investmoney)/int(stockprice)

    investmoney = int(stocknum) * int(stockprice)
    global portdbname
    title = '../../data/port/'+portdbname+'.sqlite'
    tablename = 'portfolio_result'
    
    if os.path.isfile(title):
        try:
            con = sqlite3.connect(title)
            portdf = pd_sql.read_frame("SELECT * from "+tablename, con)
            con.close()

            if portdf[portdf['title'] == stocktitle]:
                
                portdf['Price'][portdf[portdf['title'] == stocktitle].index] = stockprice
                portdf['StockNum'][portdf[portdf['title'] == stocktitle].index] = stocknum
                portdf['Money'][portdf[portdf['title'] == stocktitle].index] = investmoney
                portdf['Date'][portdf[portdf['title'] == stocktitle].index] = buydate
                
                con = sqlite3.connect(title)
                con.execute("DROP TABLE IF EXISTS "+tablename)

                pd_sql.write_frame(portdf, tablename, con)
                con.close()
                display(HTML(portdf.to_html()))
                return
        except Exception,e:
            con.close()
    else:
        portdf = pd.DataFrame().fillna(0.0)


    # gettime = foundap[foundap['title'] == stocktitle][-1:]['time'].values
    # print gettime[0]
    # print foundap[foundap['title'] == stocktitle][-1:],stockprice
    # tmpdf = deepcopy(foundap[foundap['title'] == stocktitle][-1:])
    # tmpdf['Price'] = pd.Series([stockprice], index=tmpdf.index)
    # tmpdf['StockNum'] = pd.Series([stocknum], index=tmpdf.index)
    tmpdf = pd.DataFrame({'title':[stocktitle],'Price':[stockprice],'StockNum':[stocknum],'Money':[investmoney],'Date':[buydate]})
    portdf = pd.concat([portdf,tmpdf])
    display(HTML(portdf.to_html()))
    try:
        con = sqlite3.connect(title)
        con.execute("DROP TABLE IF EXISTS "+tablename)

        pd_sql.write_frame(portdf, tablename, con)
        con.close()
    except Exception,e:
        con.close()

def portfoliosell(widget):
    print 'portfolio sell'
    
    global stockprice
    global stocktitle
    global portdbname
    title = '../../data/port/'+portdbname+'.sqlite'
    tablename = 'portfolio_result'
    
    today = datetime.now()
    todaydate = today.strftime('%Y-%m-%d')
    if os.path.isfile(title):
        
        # today = datetime(int(todaydate.year),int(todaydate.month),int(todaydate.day))

        
        con = sqlite3.connect(title)
        portdf = pd_sql.read_frame("SELECT * from "+tablename, con)
        
        if not portdf[portdf['title'] == 'PortGain']:
            newdf = pd.DataFrame({'title':['PortGain'],'time':[todaydate],'currentgain':[0],'totalGain':[0]}).fillna(0.0)

            con.execute("DROP TABLE IF EXISTS "+tablename)
            portdf = pd.concat([portdf,newdf])
            pd_sql.write_frame(portdf, tablename, con)
            con.close()
        else:
            
            con.close()
            con = sqlite3.connect(title)

            
            portdf['time'][portdf[portdf['title'] == stocktitle].index] = todaydate
            buyprice = portdf['Price'][portdf[portdf['title'] == stocktitle].index].values
            print 'buyprice:',buyprice[0]

            gain = (float(stockprice) - float(buyprice[0]))/float(buyprice[0])
            accumgain = portdf['currentgain'][portdf[portdf['title'] == 'PortGain'].index].values
            portdf['currentgain'][portdf[portdf['title'] == 'PortGain'].index] =  gain + accumgain[0]

            print 'gain:',gain

            stocknum = portdf['StockNum'][portdf[portdf['title'] == stocktitle].index].values
            print 'stocknum,',stocknum[0]
            accumTotal = portdf['totalGain'][portdf[portdf['title'] == 'PortGain'].index].values
            portdf['totalGain'][portdf[portdf['title'] == 'PortGain'].index] =  (float(stockprice) - float(buyprice[0]))*float(stocknum[0]) \
                + accumTotal[0]

            print portdf
            con.execute("DROP TABLE IF EXISTS "+tablename)
            pd_sql.write_frame(portdf, tablename, con)

            con.close()


def totalGain(widget):
    print 'totalGain'
    global dbname
    global lists
    
    print 'dbname',dbname
    con = sqlite3.connect(dbname)
    query = "SELECT * FROM sqlite_master WHERE type='table'"
    
    global df
    df = pd.io.sql.read_frame(query,con)
    # print df.tail()
    tablelen = len(df['name'])

    clear_output()
    gainsum = 0.0
    count = 0
    for cnt in range(tablelen):
        if df['name'][cnt].find('signal') != -1:
            # print df['name'][cnt]
            title = df['name'][cnt].split('_')
            # print 'title',title,title[2]
            con1 = sqlite3.connect(dbname)
            global tablename
            tablename = df['name'][cnt]
            tabledf = pd.io.sql.read_frame("SELECT * from "+tablename, con1)    
            if tabledf[-1:]['totalGain'] > 0.0:
                # print title[2],float(tabledf[-1:]['totalGain'].values)
                # gainsum += float(tabledf[-1:]['totalGain'].values)
                if float(tabledf[-1:]['totalGain'].values) >= 0.3:
                    count += 1
                    print title[2],float(tabledf[-1:]['totalGain'].values)
                    gainsum += float(tabledf[-1:]['totalGain'].values)
                    print gainsum,count
            

    

def Refresh(widget):
    clear_output()
    global dbslidechoice
    openDB(dbslidechoice)

def BenchCorr(widget):
    clear_output()


    global buyap
    if buyap != 0 and len(buyap) > 0:
        
        allbuycorrdfs = analysisCorr('none',buyap)        
        
        clear_output()
        

    global holdingap
    if holdingap != 0 and len(holdingap)>0:
        
        allholdcorrdfs = analysisCorr('none',holdingap)        
        clear_output()
        
        if buyap != 0 and len(buyap) > 0:
            print '----------- buy list correlation --------------'
            display(HTML(allbuycorrdfs.to_html()))
            print 'corr mean:',allbuycorrdfs['corrval'].mean()
        print '----------- hold list correlation --------------'
        display(HTML(allholdcorrdfs.to_html()))
        print 'corr mean:',allholdcorrdfs['corrval'].mean()
        print '#######################################'

# def BenchCorr(widget):
#     clear_output()
    

#     analysisdf = readAnalysisBenchDB()
#     limit = analysisdf['barsgain5'].quantile(0.9)
#     seldf5 = analysisdf[analysisdf['barsgain5'] > limit]
#     seldf5 = seldf5[['title','barsgain5','barsgain15','barsgain30','barsgain50']]
#     seldf5 = seldf5.sort(['barsgain5'],ascending = False)    

#     limit = analysisdf['barsgain15'].quantile(0.9)
#     seldf15 = analysisdf[analysisdf['barsgain15'] > limit]
#     seldf15 = seldf15[['title','barsgain5','barsgain15','barsgain30','barsgain50']]
#     seldf15 = seldf15.sort(['barsgain15'],ascending = False)    

#     limit = analysisdf['barsgain30'].quantile(0.9)
#     seldf30 = analysisdf[analysisdf['barsgain5'] > limit]
#     seldf30 = seldf30[['title','barsgain5','barsgain15','barsgain30','barsgain50']]
#     seldf30 = seldf30.sort(['barsgain30'],ascending = False)    

#     limit = analysisdf['barsgain50'].quantile(0.9)
#     seldf50 = analysisdf[analysisdf['barsgain50'] > limit]
#     seldf50 = seldf50[['title','barsgain5','barsgain15','barsgain30','barsgain50']]
#     seldf50 = seldf50.sort(['barsgain50'],ascending = False)    
#     # display(HTML(seldf.to_html()))

#     global buyap
#     if buyap != 0 and len(buyap) > 0:
        
#         allbuycorrdfs = analysisCorr('none',buyap)        
        
#         buytitles = buyap['title']
#         selbuydf5 = pd.DataFrame(columns= ['title','barsgain5','barsgain15','barsgain30','barsgain50']).fillna(0.0)
#         for title in seldf5['title']:
#             for buytitle in buytitles:
#                 if title == buytitle:
#                     tmpdf = seldf5[seldf5['title'] == title]
#                     selbuydf5 = pd.concat([selbuydf5,tmpdf])    
        
#         buycorrdfs5 = analysisCorr('none',selbuydf5)        
#         clear_output()

        
#         selbuydf15 = pd.DataFrame(columns= ['title','barsgain5','barsgain15','barsgain30','barsgain50']).fillna(0.0)
#         for title in seldf15['title']:
#             for buytitle in buytitles:
#                 if title == buytitle:
#                     tmpdf = seldf15[seldf15['title'] == title]
#                     selbuydf15 = pd.concat([selbuydf15,tmpdf])    
        
#         buycorrdfs15 = analysisCorr('none',selbuydf15)        
#         clear_output()


#         selbuydf30 = pd.DataFrame(columns= ['title','barsgain5','barsgain15','barsgain30','barsgain50']).fillna(0.0)
#         for title in seldf30['title']:
#             for buytitle in buytitles:
#                 if title == buytitle:
#                     tmpdf = seldf30[seldf30['title'] == title]
#                     selbuydf30 = pd.concat([selbuydf30,tmpdf])    
        
#         buycorrdfs30 = analysisCorr('none',selbuydf30)        
#         clear_output()

#         selbuydf50 = pd.DataFrame(columns= ['title','barsgain5','barsgain15','barsgain30','barsgain50']).fillna(0.0)
#         for title in seldf50['title']:
#             for buytitle in buytitles:
#                 if title == buytitle:
#                     tmpdf = seldf50[seldf50['title'] == title]
#                     selbuydf50 = pd.concat([selbuydf50,tmpdf])    
        
#         buycorrdfs50 = analysisCorr('none',selbuydf50)        
#         clear_output()
        

#     global holdingap
#     if holdingap != 0 and len(holdingap)>0:
        
#         holdtitles = holdingap['title']
#         selholddf5 = pd.DataFrame(columns= ['title','barsgain5','barsgain15','barsgain30','barsgain50']).fillna(0.0)
#         for title in seldf5['title']:
#             for holdtitle in holdtitles:
#                 if title == holdtitle:
#                     tmpdf = seldf5[seldf5['title'] == title]
#                     selholddf5 = pd.concat([selholddf5,tmpdf])    
#         corrdfs5 = analysisCorr('none',selholddf5)        
#         clear_output()
        

        
        
#         selholddf15 = pd.DataFrame(columns= ['title','barsgain5','barsgain15','barsgain30','barsgain50']).fillna(0.0)
#         for title in seldf15['title']:
#             for holdtitle in holdtitles:
#                 if title == holdtitle:
#                     tmpdf = seldf15[seldf15['title'] == title]
#                     selholddf15 = pd.concat([selholddf15,tmpdf])    
#         corrdfs15 = analysisCorr('none',selholddf15)        
#         clear_output()
        

        
        
#         selholddf30 = pd.DataFrame(columns= ['title','barsgain5','barsgain15','barsgain30','barsgain50']).fillna(0.0)
#         for title in seldf30['title']:
#             for holdtitle in holdtitles:
#                 if title == holdtitle:
#                     tmpdf = seldf30[seldf30['title'] == title]
#                     selholddf30 = pd.concat([selholddf30,tmpdf])    
#         corrdfs30 = analysisCorr('none',selholddf30)        
#         clear_output()
        

        
        
#         selholddf50 = pd.DataFrame(columns= ['title','barsgain5','barsgain15','barsgain30','barsgain50']).fillna(0.0)
#         for title in seldf50['title']:
#             for holdtitle in holdtitles:
#                 if title == holdtitle:
#                     tmpdf = seldf50[seldf50['title'] == title]
#                     selholddf50 = pd.concat([selholddf50,tmpdf])    
#         corrdfs50 = analysisCorr('none',selholddf50)        
#         clear_output()
        

#         print '----------- 5 day choice --------------'
#         selbuydf5 = pd.merge(selbuydf5,buycorrdfs5, on = 'title')
#         display(HTML(selbuydf5.to_html()))    
#         print '5 day corr mean:',buycorrdfs5['corrval'].mean()
#         print '----------- 15 day choice --------------'
#         selbuydf15 = pd.merge(selbuydf15,buycorrdfs15, on = 'title')
#         display(HTML(selbuydf15.to_html()))
#         print '15 day corr mean:',buycorrdfs15['corrval'].mean()
#         print '----------- 30 day choice --------------'
#         selbuydf30 = pd.merge(selbuydf30,buycorrdfs30, on = 'title')
#         display(HTML(selbuydf30.to_html()))
#         print '30 day corr mean:',buycorrdfs30['corrval'].mean()
#         print '----------- 50 day choice --------------'
#         selbuydf50 = pd.merge(selbuydf50,buycorrdfs50, on = 'title')
#         display(HTML(selbuydf50.to_html()))
#         print '50 day corr mean:',buycorrdfs50['corrval'].mean()
#         print 'all buy list corr'
#         display(HTML(allbuycorrdfs.to_html()))

#         print '#######################################'

#         print '----------- 5 day holding choice --------------'
#         selholddf5 = pd.merge(selholddf5,corrdfs5, on = 'title')
#         display(HTML(selholddf5.to_html()))    
#         print '5 day corr mean:',corrdfs5['corrval'].mean()
#         print '----------- 15 day holding choice --------------'
#         selholddf15 = pd.merge(selholddf15,corrdfs15, on = 'title')
#         display(HTML(selholddf15.to_html()))
#         print '15 day corr mean:',corrdfs15['corrval'].mean()
#         print '----------- 30 day holding choice --------------'
#         selholddf30 = pd.merge(selholddf30,corrdfs30, on = 'title')
#         display(HTML(selholddf30.to_html()))
#         print '30 day corr mean:',corrdfs30['corrval'].mean()
#         print '----------- 50 day holding choice --------------'
#         selholddf50 = pd.merge(selholddf50,corrdfs50, on = 'title')
#         display(HTML(selholddf50.to_html()))
#         print '50 day corr mean:',corrdfs50['corrval'].mean()


def Select(widget):
    clear_output()
    global dbslidechoice
    global dbname
    global lists
    dbname = lists[dbslidechoice]
    print 'dbname',dbname
    con = sqlite3.connect(dbname)
    query = "SELECT * FROM sqlite_master WHERE type='table'"
    
    global df
    df = pd.io.sql.read_frame(query,con)
    # print df.tail()
    tablelen = len(df['name'])

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

    selectdf25 = pd.DataFrame().fillna(0.0)
    selectdf50 = pd.DataFrame().fillna(0.0)
    selectdf75 = pd.DataFrame().fillna(0.0)

    for cnt in range(tablelen):
        if df['name'][cnt].find('benchmark') != -1:
            # print df['name'][cnt]
            title = df['name'][cnt].split('_')
            
            if analysisdf[analysisdf['title'] == title[2]]:
                founddf = analysisdf[analysisdf['title'] == title[2]]
                bar25p = founddf['bars_25p'].values[0]
                bar50p = founddf['bars_50p'].values[0]
                bar75p = founddf['bars_75p'].values[0]
                bar90p = founddf['bars_90p'].values[0]
                # print 'title',title[1],title[2],founddf['bars_25p'].values[0]    
            # con1 = sqlite3.connect(dbname)
            global tablename
            tablename = df['name'][cnt]
            tabledf = pd.io.sql.read_frame("SELECT * from "+tablename, con)
            # print title[2],tabledf['stddf'][-1:].values[0]    
            if tabledf['Benchmark'][-1:].values[0] > tabledf['MA1'][-1:].values[0] \
                and tabledf['stddf'][-1:].values[0] < bar25p:
                # print title[2]
                tmptabledf = tabledf[-1:]
                tmptabledf['title'] = title[2]
                selectdf25 = pd.concat([selectdf25,tmptabledf])

            if tabledf['Benchmark'][-1:].values[0] > tabledf['MA1'][-1:].values[0] \
                and tabledf['stddf'][-1:].values[0] < bar50p \
                and tabledf['stddf'][-1:].values[0] >= bar25p:
                # print title[2]
                tmptabledf = tabledf[-1:]
                tmptabledf['title'] = title[2]
                selectdf50 = pd.concat([selectdf50,tmptabledf])

            if tabledf['Benchmark'][-1:].values[0] > tabledf['MA1'][-1:].values[0] \
                and tabledf['stddf'][-1:].values[0] < bar75p \
                and tabledf['stddf'][-1:].values[0] >= bar50p:
                # print title[2]
                tmptabledf = tabledf[-1:]
                tmptabledf['title'] = title[2]
                selectdf75 = pd.concat([selectdf75,tmptabledf])

    con.close()

    print '------------ ~ 25 -------------'                
    display(HTML(selectdf25.to_html()))
    print '------------ 25 ~ 50 -------------'                
    display(HTML(selectdf50.to_html()))
    print '------------ 50 ~ 75 -------------'                
    display(HTML(selectdf75.to_html()))

    try:
        writedbname = '../../data/analysis/select25p.sqlite'
        con2 = sqlite3.connect(writedbname)
        tablename = 'result_25p'
        con2.execute("DROP TABLE IF EXISTS "+tablename)
        pd_sql.write_frame(selectdf25, tablename, con2)

        con2.close()

        writedbname = '../../data/analysis/select50p.sqlite'
        con2 = sqlite3.connect(writedbname)
        tablename = 'result_50p'
        con2.execute("DROP TABLE IF EXISTS "+tablename)
        pd_sql.write_frame(selectdf50, tablename, con2)

        con2.close()

        writedbname = '../../data/analysis/select75p.sqlite'
        con2 = sqlite3.connect(writedbname)
        tablename = 'result_75p'
        con2.execute("DROP TABLE IF EXISTS "+tablename)
        pd_sql.write_frame(selectdf75, tablename, con2)

        con2.close()
    except Exception,e:
        print 'error ',e
        con2.close()


def ShowResult(widget):
    print 'show result'
    clear_output()
    global dbslidechoice
    global dbname
    global lists
    dbname = lists[dbslidechoice]
    print 'dbname',dbname
    con = sqlite3.connect(dbname)
    query = "SELECT * FROM sqlite_master WHERE type='table'"
    
    global df
    df = pd.io.sql.read_frame(query,con)
    
    # print df.tail()
    tablelen = len(df['name'])
    gains_1 = []
    gains_2 = []
    gains_3 = []
    gains_4 = []
    gains_5 = []
    gains_6 = []
    gains_7 = []
    gains_8 = []
    gains_9 = []
    gains_10 = []
    gains_11 = []
    gains_12 = []

    title1pos = []
    title1neg = []
    title2pos = []
    title2neg = []
    title3pos = []
    title3neg = []
    title4pos = []
    title4neg = []
    title5pos = []
    title5neg = []
    title6pos = []
    title6neg = []
    title7pos = []
    title7neg = []
    title8pos = []
    title8neg = []
    title9pos = []
    title9neg = []
    title10pos = []
    title10neg = []
    title11pos = []
    title11neg = []
    title12pos = []
    title12neg = []
    
    
    for cnt in range(tablelen):
        if df['name'][cnt].find('signal') != -1:
            tablename = df['name'][cnt]
            titlename = tablename.split('_')[2]
            tabledf = pd.io.sql.read_frame("SELECT * from "+tablename, con)

            for cnt in range(len(tabledf)):
                str1 = tabledf.Date[cnt].split('-')
                dtdate = datetime(int(str1[0]),int(str1[1]),int(str1[2].split(' ')[0]),0)
                tabledf.Date[cnt]= dtdate
    
            tabledf.index = tabledf.Date
            curyear = 2014
            tabledf = tabledf[tabledf.index.year == curyear]
            try:
                gains_1.append(tabledf['currentgain'][tabledf.index.month ==1].sum())
                gains_2.append(tabledf['currentgain'][tabledf.index.month ==2].sum())
                gains_3.append(tabledf['currentgain'][tabledf.index.month ==3].sum())
                gains_4.append(tabledf['currentgain'][tabledf.index.month ==4].sum())
                gains_5.append(tabledf['currentgain'][tabledf.index.month ==5].sum())
                gains_6.append(tabledf['currentgain'][tabledf.index.month ==6].sum())
                gains_7.append(tabledf['currentgain'][tabledf.index.month ==7].sum())
                gains_8.append(tabledf['currentgain'][tabledf.index.month ==8].sum())
                gains_9.append(tabledf['currentgain'][tabledf.index.month ==9].sum())
                gains_10.append(tabledf['currentgain'][tabledf.index.month ==10].sum())
                gains_11.append(tabledf['currentgain'][tabledf.index.month ==11].sum())
                gains_12.append(tabledf['currentgain'][tabledf.index.month ==12].sum())

                if tabledf['currentgain'][tabledf.index.month ==1].sum() > 0.0:
                    title1pos.append(titlename)
                if tabledf['currentgain'][tabledf.index.month ==1].sum() < 0.0:
                    title1neg.append(titlename)    
                if tabledf['currentgain'][tabledf.index.month ==2].sum() > 0.0:
                    title2pos.append(titlename)
                if tabledf['currentgain'][tabledf.index.month ==2].sum() < 0.0:
                    title2neg.append(titlename)    
                if tabledf['currentgain'][tabledf.index.month ==3].sum() > 0.0:
                    title3pos.append(titlename)
                if tabledf['currentgain'][tabledf.index.month ==3].sum() < 0.0:
                    title3neg.append(titlename)
                if tabledf['currentgain'][tabledf.index.month ==4].sum() > 0.0:
                    title4pos.append(titlename)
                if tabledf['currentgain'][tabledf.index.month ==4].sum() < 0.0:
                    title4neg.append(titlename)    
                if tabledf['currentgain'][tabledf.index.month ==5].sum() > 0.0:
                    title5pos.append(titlename)
                if tabledf['currentgain'][tabledf.index.month ==5].sum() < 0.0:
                    title5neg.append(titlename)        
                if tabledf['currentgain'][tabledf.index.month ==6].sum() > 0.0:
                    title6pos.append(titlename)
                if tabledf['currentgain'][tabledf.index.month ==6].sum() < 0.0:
                    title6neg.append(titlename)        
                if tabledf['currentgain'][tabledf.index.month ==7].sum() > 0.0:
                    title7pos.append(titlename)
                if tabledf['currentgain'][tabledf.index.month ==7].sum() < 0.0:
                    title7neg.append(titlename)    
                if tabledf['currentgain'][tabledf.index.month ==8].sum() > 0.0:
                    title8pos.append(titlename)
                if tabledf['currentgain'][tabledf.index.month ==8].sum() < 0.0:
                    title8neg.append(titlename)    
                if tabledf['currentgain'][tabledf.index.month ==9].sum() > 0.0:
                    title9pos.append(titlename)
                if tabledf['currentgain'][tabledf.index.month ==9].sum() < 0.0:
                    title9neg.append(titlename)            
                if tabledf['currentgain'][tabledf.index.month ==10].sum() > 0.0:
                    title10pos.append(titlename)
                if tabledf['currentgain'][tabledf.index.month ==10].sum() < 0.0:
                    title10neg.append(titlename)       
                if tabledf['currentgain'][tabledf.index.month ==11].sum() > 0.0:
                    title11pos.append(titlename)
                if tabledf['currentgain'][tabledf.index.month ==11].sum() < 0.0:
                    title11neg.append(titlename)        
                if tabledf['currentgain'][tabledf.index.month ==12].sum() > 0.0:
                    title12pos.append(titlename)
                if tabledf['currentgain'][tabledf.index.month ==12].sum() < 0.0:
                    title12neg.append(titlename)    

            except Exception,e:
                pass
            # print tabledf['currentgain'][tabledf.index.month ==12].sum()
            # print tabledf[tabledf.index.month ==12]
            # print tablename
            # break 
    showresults = pd.DataFrame({'1':gains_1,'2':gains_2,'3':gains_3,'4':gains_4,'5':gains_5,'6':gains_6,'7':gains_7\
        ,'8':gains_8,'9':gains_9,'10':gains_10,'11':gains_11,'12':gains_12})
    

    # display(HTML(showresults.to_html()))
    print '1 mean:',showresults['1'].mean(),'1 sum:',showresults['1'].sum(),'pos count:',len(showresults[showresults['1'] > 0]),'neg count:',len(showresults[showresults['1'] < 0])
    print '2 mean:',showresults['2'].mean(),'2 sum:',showresults['2'].sum(),'pos count:',len(showresults[showresults['2'] > 0]),'neg count:',len(showresults[showresults['2'] < 0])
    print '3 mean:',showresults['3'].mean(),'3 sum:',showresults['3'].sum(),'pos count:',len(showresults[showresults['3'] > 0]),'neg count:',len(showresults[showresults['3'] < 0])
    print '4 mean:',showresults['4'].mean(),'4 sum:',showresults['4'].sum(),'pos count:',len(showresults[showresults['4'] > 0]),'neg count:',len(showresults[showresults['4'] < 0])
    print '5 mean:',showresults['5'].mean(),'5 sum:',showresults['5'].sum(),'pos count:',len(showresults[showresults['5'] > 0]),'neg count:',len(showresults[showresults['5'] < 0])
    print '6 mean:',showresults['6'].mean(),'6 sum:',showresults['6'].sum(),'pos count:',len(showresults[showresults['6'] > 0]),'neg count:',len(showresults[showresults['6'] < 0])
    print '7 mean:',showresults['7'].mean(),'7 sum:',showresults['7'].sum(),'pos count:',len(showresults[showresults['7'] > 0]),'neg count:',len(showresults[showresults['7'] < 0])
    print '8 mean:',showresults['8'].mean(),'8 sum:',showresults['8'].sum(),'pos count:',len(showresults[showresults['8'] > 0]),'neg count:',len(showresults[showresults['8'] < 0])
    print '9 mean:',showresults['9'].mean(),'9 sum:',showresults['9'].sum(),'pos count:',len(showresults[showresults['9'] > 0]),'neg count:',len(showresults[showresults['9'] < 0])
    print '10 mean:',showresults['10'].mean(),'10 sum:',showresults['10'].sum(),'pos count:',len(showresults[showresults['10'] > 0]),'neg count:',len(showresults[showresults['10'] < 0])
    print '11 mean:',showresults['11'].mean(),'11 sum:',showresults['11'].sum(),'pos count:',len(showresults[showresults['11'] > 0]),'neg count:',len(showresults[showresults['11'] < 0])
    print '12 mean:',showresults['12'].mean(),'12 sum:',showresults['12'].sum(),'pos count:',len(showresults[showresults['12'] > 0]),'neg count:',len(showresults[showresults['12'] < 0])
    
    showrtitles = pd.DataFrame({'1pos':title1pos})
    display(HTML(showrtitles.to_html()))
    showrtitles = pd.DataFrame({'1neg':title1neg})
    display(HTML(showrtitles.to_html()))

    showrtitles = pd.DataFrame({'2pos':title2pos})
    display(HTML(showrtitles.to_html()))
    showrtitles = pd.DataFrame({'2neg':title2neg})
    display(HTML(showrtitles.to_html()))

    showrtitles = pd.DataFrame({'3pos':title3pos})
    display(HTML(showrtitles.to_html()))
    showrtitles = pd.DataFrame({'3neg':title3neg})
    display(HTML(showrtitles.to_html()))

    showrtitles = pd.DataFrame({'4pos':title4pos})
    display(HTML(showrtitles.to_html()))
    showrtitles = pd.DataFrame({'4neg':title4neg})
    display(HTML(showrtitles.to_html()))

    showrtitles = pd.DataFrame({'5pos':title5pos})
    display(HTML(showrtitles.to_html()))
    showrtitles = pd.DataFrame({'5neg':title5neg})
    display(HTML(showrtitles.to_html()))

    showrtitles = pd.DataFrame({'6pos':title6pos})
    display(HTML(showrtitles.to_html()))
    showrtitles = pd.DataFrame({'6neg':title6neg})
    display(HTML(showrtitles.to_html()))

    showrtitles = pd.DataFrame({'7pos':title7pos})
    display(HTML(showrtitles.to_html()))
    showrtitles = pd.DataFrame({'7neg':title7neg})
    display(HTML(showrtitles.to_html()))

    showrtitles = pd.DataFrame({'8pos':title8pos})
    display(HTML(showrtitles.to_html()))
    showrtitles = pd.DataFrame({'8neg':title8neg})
    display(HTML(showrtitles.to_html()))

    showrtitles = pd.DataFrame({'9pos':title9pos})
    display(HTML(showrtitles.to_html()))
    showrtitles = pd.DataFrame({'9neg':title9neg})
    display(HTML(showrtitles.to_html()))

    showrtitles = pd.DataFrame({'10pos':title10pos})
    display(HTML(showrtitles.to_html()))
    showrtitles = pd.DataFrame({'10neg':title10neg})
    display(HTML(showrtitles.to_html()))

    showrtitles = pd.DataFrame({'11pos':title11pos})
    display(HTML(showrtitles.to_html()))
    showrtitles = pd.DataFrame({'11neg':title11neg})
    display(HTML(showrtitles.to_html()))

    showrtitles = pd.DataFrame({'12pos':title12pos})
    display(HTML(showrtitles.to_html()))
    showrtitles = pd.DataFrame({'12neg':title12neg})
    display(HTML(showrtitles.to_html()))






    # ,'count:',len(gains_3),'4:',showresults['4'].mean()\
    # ,'count:',len(gains_4),'5:',showresults['5'].mean(),'count:',len(gains_5)\
    # ,'6:',showresults['6'].mean(),'count:',len(gains_6),'7:',showresults['7'].mean(),'count:',len(gains_7),'8:',showresults['8'].mean(),'count:',len(gains_8)\
    # ,'9:'showresults['9'].mean()\
    # ,'count:',len(gains_9),'10:',showresults['10'].mean(),'count:',len(gains_10)\
    # ,'11:',showresults['11'].mean(),'count:',len(gains_11),'12:',showresults['12'].mean(),'count:',len(gains_12)
    # print gains_12
    # print len(showresults[showresults['10'] > 0])


def SaveBuyDB(widget):
    print 'save buy db'
    clear_output()
    global foundap
    display(HTML(foundap.to_html()))    
    
    todaydate = datetime.today()

    try:
        dbname = '../../data/port/buy/buy_db_'+str(todaydate.year)+'{0:02d}'.format(int(todaydate.month))+'{0:02d}'.format(int(todaydate.day))+'.sqlite'
        tablename = 'buy_table'
        con = sqlite3.connect(dbname)
        con.execute("DROP TABLE IF EXISTS "+tablename)

        pd_sql.write_frame(foundap, tablename, con)
        con.close()
    except Exception,e:
        con.close()


run_button = widgets.ButtonWidget(description="Individual History Open")        
run_button.on_click(stocksignalhist)
display(run_button)

# run_button2 = widgets.ButtonWidget(description="Portfolio Buy")        
# run_button2.on_click(portfoliobuy)
# display(run_button2)

# run_button3 = widgets.ButtonWidget(description="Portfolio Sell")        
# run_button3.on_click(portfoliosell)
# display(run_button3)


run_button4 = widgets.ButtonWidget(description="Total Gain")        
run_button4.on_click(totalGain)
display(run_button4)


run_button5 = widgets.ButtonWidget(description="Refresh")        
run_button5.on_click(Refresh)
display(run_button5)

run_button6 = widgets.ButtonWidget(description="Benchmark Corr")        
run_button6.on_click(BenchCorr)
display(run_button6)

run_button7 = widgets.ButtonWidget(description="Select titles")        
run_button7.on_click(Select)
display(run_button7)

run_button8 = widgets.ButtonWidget(description="Show Results")        
run_button8.on_click(ShowResult)
display(run_button8)

run_button9 = widgets.ButtonWidget(description="Save Buy DB")        
run_button9.on_click(SaveBuyDB)
display(run_button9)

chwid = widgets.CheckboxWidget(description = 'Apply Up Trend', value=False)
display(chwid)


def checkbox_click_handler(name,value):
    global chbox_click 
    if value == True:
        chbox_click = value
        print 'click:',value
    elif value == False:
        chbox_click = value
        print 'click:',value
# help(widgets.CheckboxWidget)    
chwid.on_trait_change(checkbox_click_handler)


chwid2 = widgets.CheckboxWidget(description = 'Corr', value=False)
display(chwid2)


def checkbox_click_handler2(name,value):
    global chbox_click2 
    if value == True:
        chbox_click2 = value
        print 'click:',value
    elif value == False:
        chbox_click2 = value
        print 'click:',value
# help(widgets.CheckboxWidget)    
chwid2.on_trait_change(checkbox_click_handler2)


chwid3 = widgets.CheckboxWidget(description = 'Small Group', value=False)
display(chwid3)


def checkbox_click_handler3(name,value):
    global chbox_click3 
    if value == True:
        chbox_click3 = value
        print 'click:',value
        global lists
        lists = glob.glob("../../data/result2/small/*.sqlite")
        global length
        length = len(lists)
        global i1
        i1.close()
        print 'db length ',length
        i1 = interactive(openDB,DBchoice=(0,length-1))
        display(i1)        

    elif value == False:
        chbox_click3 = value
        print 'click:',value
        global lists
        lists = glob.glob("../../data/result2/*.sqlite")
        global length
        length = len(lists)
        global i1
        i1.close()
        print 'db length ',length
        i1 = interactive(openDB,DBchoice=(0,length-1))
        display(i1)        
        
# help(widgets.CheckboxWidget)    
chwid3.on_trait_change(checkbox_click_handler3)



