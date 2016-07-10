%matplotlib inline
# from matplotlib import interactive
# interactive(True)

# from guiqwt.plot import CurveDialog
# from guiqwt.builder import make



def RunSimul(codearg,typearg,namearg,mode,dbmode,histmode,runcount,srcsite,writedblog,updbpattern,appenddb,startdatemode 
    ,dbtradinghist):

    code = codearg #'097950'#'005930' #'005380'#009540 #036570

    if codearg == '005490' or codearg == '000660' or codearg == '068870'\
        or codearg == '078520':
        srcsite = 2
    
    if srcsite == 1:
        if typearg == 1:
            symbol = 'GOOG/KRX_'+code
        elif typearg == 2:
            symbol = 'GOOG/KOSDAQ_'+code
        elif typearg == 3:
            symbol = 'GOOG/INDEXKRX_KOSPI200'  
    elif srcsite == 2:
        if typearg == 1:
            symbol = 'YAHOO/KS_'+code
        elif typearg == 2:
            symbol = 'YAHOO/KQ_'+code
        elif typearg == 3:
            symbol = 'YAHOO/INDEX_KS11'  

    if startdatemode == 1:            
        startdate = '2007-01-01'
    else:
        startday =  datetime.today() - timedelta(days=150)
        startdate = str(startday).split(' ')[0]
    print 'startdate',startdate

    print symbol,namearg

    today = datetime.today()
    bars_org =  Quandl.get(symbol, collapse='Daily', trim_start=startdate, trim_end=today,authtoken="")
    # bars = Quandl.get(symbol, collapse='Daily', trim_start=startdate, trim_end=today,authtoken="")
    # print bars[-10:]
    # print bars_org.tail()
    print '---------'
    #print len(bars)

    
    
    startday = today- timedelta(days=7 )
    # print today.year,today.month,today.day
    # print startday.year,startday.month,startday.day
    
    if typearg == 3:
        histurl = 'http://ichart.yahoo.com/table.csv?s=^KS11'+'&a='+str(startday.month-1)+\
        '&b='+str(startday.day)+'&c='+str(startday.year)+'&d='+str(today.month-1)+'&e='+str(today.day)+'&f='+str(today.year)+'&ignore=.csv'
        # print histurl
    else:
        histurl = 'http://ichart.yahoo.com/table.csv?s='+code+'.KS'+'&a='+str(startday.month-1)+\
        '&b='+str(startday.day)+'&c='+str(startday.year)+'&d='+str(today.month-1)+'&e='+str(today.day)+'&f='+str(today.year)+'&ignore=.csv'
        # print histurl
    '''
    yahoo scrape api 
    '''
    try:
        response = urllib2.urlopen(histurl)
        histdf = pd.read_csv(response)

        datelen = len(histdf.Date)

        for cnt in range(datelen):
            str1 = histdf.Date[cnt].split('-')
            dtdate = datetime(int(str1[0]),int(str1[1]),int(str1[2]),0)
            histdf.Date[cnt]= dtdate

        histdf = histdf[histdf.Volume != 0]
        histdf = histdf.drop('Adj Close',1)
        histdf.index= histdf.Date
        histdf.index.name = 'Date'
        histdf = histdf.drop('Date',1)
        print '----date adjust start---'
        bars_new_unique = histdf[~histdf.index.isin(bars_org.index)]
        bars_org = pd.concat([bars_org, bars_new_unique])
        # print bars_org.tail()
        print '----date adjust end-----'

        today  = datetime.today()
        todayweek = today.weekday()

        bars_org['week'] = bars_org.index.weekday
        tailweekdays = bars_org['week'][-5:]
        # print tailweekdays

        if 0 <= todayweek <=4 :
            for cnt in range(0,len(tailweekdays)):
                day = cnt +1 
                # print gbars2['week'][-1*day]
                checkday = bars_org['week'][-1*day]
                
                # print todayweek,checkday,bars_org.index[-1*day]
                if todayweek != checkday:
                    raise Exception("week check error")
                if todayweek == 0:
                    todayweek = 4
                else:
                    todayweek = todayweek - 1          
    except Exception,e:
        print e
        ''' 
        naver scrape for yahoo ichart alternative
        '''
        histdf = fetchData(code) 
        histdf = histdf[histdf.Volume != 0]
        # print histdf
        print '----date adjust start---'
        bars_new_unique = histdf[~histdf.index.isin(bars_org.index)]
        bars_org = pd.concat([bars_org, bars_new_unique])
        print bars_org.tail()
        print '----date adjust end-----'
    

    start = time.clock()
    if typearg == 1:
        rtsymbol = code+'.KS'
    elif typearg == 2:        
        rtsymbol = code+'.KQ'
    elif typearg == 3:
        rtsymbol = '^KS11'#'^KS200'
    # rtsymbol = '^KS200'
    realtimeURL = 'http://finance.yahoo.com/d/quotes.csv?s='+rtsymbol+'&f=sl1d1t1c1ohgv&e=.csv'
    print realtimeURL
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
        print 'rtsymbol:',rtsymbol,'rtclose:',rtclose,rtdate,rttime,rtchange,'rtopen:',rtopen,'rthigh:',rthigh,'rtlow:',rtlow,'rtvolume:',rtvolume

    # print date2num(datetime.strptime(rtdate.replace('/',' '),'%m %d %Y'))

    # print bars.index[-1]
    # print date_object > bars.index[-1]
    # date_object  = date_object- dt.timedelta(days=1) 
    # print date_object > bars.index[-1]
    date_object = datetime.strptime(rtdate.replace('/',' '), '%m %d %Y')
    rtdf = pd.date_range(date_object, date_object)

    date_append = False
    
    bars_org = bars_org.sort_index()
    # print len(bars_org),len(bars_org['Close']),len(bars_org['Volume'])
    if date_object > bars_org.index[-1]:
        d={'Open':rtopen, 'High':rthigh,'Low':rtlow,'Close':rtclose,'Volume':rtvolume }
        appenddf = pd.DataFrame(d,index=rtdf)
        appenddf.index.name = 'Date'
        date_append = True
        print appenddf,date_append
        # print bars_org.index.isin(appenddf.index)
    
        bars = pd.concat([bars_org,appenddf])
        print '----------'
        # print bars.tail()
    else:
        bars = bars_org

    # bars = []
    # startdate = '2013-01-01'
    # enddate = '2013-06-01'
    # bars =  Quandl.get(symbol, collapse='Daily', trim_start=startdate, trim_end=enddate,authtoken="")
    bars = bars.drop_duplicates()
    bars = bars.sort_index()
    print '----------final sorted bars---------'
    print bars.tail()
    elapsed = (time.clock() - start)
    print 'real time data web gathering elapsed time:',elapsed


    today  = datetime.today()
    todayweek = today.weekday()

    bars['week'] = bars.index.weekday
    tailweekdays = bars['week'][-5:]
    # print tailweekdays

    if 0 <= todayweek <=4 :
        for cnt in range(0,len(tailweekdays)):
            day = cnt +1 
            # print gbars2['week'][-1*day]
            checkday = bars['week'][-1*day]
            
            # print todayweek,checkday,bars.index[-1*day]
            if todayweek != checkday:
                raise Exception("week check error")
            if todayweek == 0:
                todayweek = 4
            else:
                todayweek = todayweek - 1                

    #file delete
    basepos = u"../../data/hist/"
    
    
    deletename = basepos+u'hist_db_'+codearg+u'_'+namearg+u'.sqlite'
    if os.path.isfile(deletename):
        os.remove(deletename)
    
    print 'hist saving start'
    import sqlite3
    import pandas.io.sql as pd_sql
    dbname = 'hist_db_'+codearg+'_'+namearg+'.sqlite'
    con = sqlite3.connect("../../data/hist/"+dbname)
    tablename_base = 'result_'+codearg+'_'+namearg

    # for cnt in range(dblen):
    tablename = tablename_base
    # print 'writetable:',tablename
    con.execute("DROP TABLE IF EXISTS "+tablename)
    bars2 = bars.reset_index()
    print 'bars2'
    print bars2.tail()
    pd_sql.write_frame(bars2, tablename, con)

    
    # readlist = []    
    # for cnt in range(dblen):
    #     tablename = tablename_base+'_'+str(cnt)
    #     # print 'readtable:',tablename
    #     patterndf = pd_sql.read_frame("SELECT * from "+tablename, con)
    #     readlist.append(PatternData(patterndf))
    #     readlist[cnt].patterndf.index = readlist[cnt].patterndf['Date']
    #     readlist[cnt].patterndf = readlist[cnt].patterndf.drop('Date',1)


    # print 'read pattern:',readlist[0].patterndf
    # print 'org patternAr:',patternAr_org[0].patterndf
    
    # con.close()    
    
    
    con.close()        
    
    print 'histdb save done'

