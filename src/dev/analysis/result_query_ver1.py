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



global chbox_click 
chbox_click = False

import glob
global lists
lists = glob.glob("../../data/result/*.sqlite")
# print lists[0]

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


def openDB(DBchoice = 0):
    
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

    seedmoney = 2000000.0

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
            # print tablename,title[6],float(tabledf[-1:]['totalGain'].values)
            if pd.isnull(tabledf[-1:]['totalGain'].values) == False:
                if prevtitlecnt == 0:
                    totalgainsum = float(tabledf[-1:]['totalGain'].values)*seedmoney
                    totalgain = float(tabledf[-1:]['totalGain'].values)
                elif prevtitlecnt != 0 and prevtitlecnt == title[6]:
                    totalgainsum = totalgainsum + float(tabledf[-1:]['totalGain'].values)*seedmoney
                    # print tablename,totalgainsum,float(tabledf[-1:]['totalGain'].values)*seedmoney
                    totalgain = totalgain + float(tabledf[-1:]['totalGain'].values)

                if prevtitlecnt != 0 and prevtitlecnt != title[6]:
                    totalgainlist.append(totalgainsum)   
                    totalgainpctlist.append(totalgain) 
                    totalgainsum = 0
                    totalgain = 0
                    totalgainsum = float(tabledf[-1:]['totalGain'].values) *seedmoney
                    totalgain = float(tabledf[-1:]['totalGain'].values)

                prevtitlecnt = title[6]


            if str(tabledf[-1:]['BuyorSell'].values).find('buyprice') != -1:
                tmpstr = tabledf['Date'][-1:].values
                tabledate = datetime.strptime(tmpstr[0], '%Y-%m-%d %H:%M:%S')
                
                if tabledate > datetime.today()-timedelta(days=offsetday) - timedelta(days=1) and \
                    float(tabledf[-1:]['totalGain'].values) > 0.0:

                    tmpdf = deepcopy(tabledf[-1:])
                    tmpdf['title'] = pd.Series([title[2]], index=tmpdf.index)
                    tmpdf['time'] = pd.Series([title[4]+'_'+title[5]], index=tmpdf.index)
                    ap = pd.concat([ap,tmpdf])
                    # print tmpdf
                    listindex = int(title[6]) -1
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
            # print title,title[5],title[6],type(title[6]),int(title[6])
            
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

    if len(ap) >0 :
    # print foundap[['BuyorSell','Price','time','title']][-100:-50]
        print foundap[['BuyorSell','Date','Price','time','title','totalGain']][-100:-50]
        print foundap[['BuyorSell','Date','Price','time','title','totalGain']][-50:]
    if len(ap_holding)>0:
        print '--------holding----------'
        print ap_holding[['BuyorSell','Date','Price','time','title','totalGain']][-100:-50]
        print ap_holding[['BuyorSell','Date','Price','time','title','totalGain']][-50:]

    for cnt in range(len(list2d)):
        if cnt == 0:
            duplicates = list2d[0]
            # print 'cnt0:',duplicates
        if cnt >= 1 and len(list2d[cnt]) > 0:
            duplicates = set(x for x in list2d[cnt] if x in duplicates)
            # print 'cnt >= 1',duplicates
        if len(list2d[cnt]) == 0:
            list2dlen = cnt 
            break

    dulist = list(duplicates)
    print '--------duplicates lists------------'
    global chbox_click 
    uplist = []
    for title in dulist:
        
        if chbox_click == False:
            print title        
        else:
            for rowcnt in range(1,sheet.nrows):
                code = '{0:06d}'.format(int(sheet.row_values(rowcnt)[0]))
                name = sheet.row_values(rowcnt)[1]
                # print type(name),name,title,type(title)
                try:
                    if name == title:
                        print 'matching found:', name, code
                        ispatFound = RunSimul(str(code), 1, name, 'realtime', 'dbpattern', 'none', 1, 1)
                        if ispatFound == True:
                            print 'up pattern title:',title
                            uplist.append(title)
                        # break
                except:
                    pass
            # break
    
    for title in uplist:
        print 'up pattern title:',title

    print '--------today sell lists------------'        
    print selldf[-100:-50]
    print selldf[-50:]

    print '--------total Gain sum lists------------'        
    totalgainlist.append(totalgainsum)
    totalgainpctlist.append(totalgain) 
    print totalgainlist
    print totalgainpctlist

    
    sectorlist = [u'전기전자',u'운수장비',u'화학',u'건설',u'음식료',u'통신',u'철강금속',u'비금속광물',u'금융',u'증권',u'보험',u'서비스',\
        u'전기가스',u'운수창고',u'유통',u'섬유의복',u'의약품',u'기계',u'의료정밀',u'종이목재']
    if len(ap) > 0:        
        print '--------sector analysis buylists-----------------'
        sectordf = searchsector(sectorlist,ap)
        display(HTML(sectordf.to_html()))
    if len(ap_holding) > 0:         
        print '--------sector analysis holding lists-----------------'
        sectordf = searchsector(sectorlist,ap_holding)
        display(HTML(sectordf.to_html()))
    


global i1
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

global stockprice
stockprice = u''
def show_price(Price=u''):
    print Price
    global stockprice
    stockprice = Price

i3 = interact(show_price,Price=u'')        
display(i3)

global investmoney
investmoney = u''
def show_money(Money=u''):
    print Money
    global investmoney
    investmoney = Money

i3 = interact(show_money,Money=u'')        
display(i3)


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
    
    
    stocknum = int(investmoney)/int(stockprice)

    title = '../../data/port/port_db.sqlite'
    tablename = 'portfolio_result'
    
    if os.path.isfile(title):
        con = sqlite3.connect(title)
        portdf = pd_sql.read_frame("SELECT * from "+tablename, con)
        con.close()

        if portdf[portdf['title'] == stocktitle]:
            
            portdf['Price'][portdf[portdf['title'] == stocktitle].index] = stockprice
            portdf['StockNum'][portdf[portdf['title'] == stocktitle].index] = stocknum
            
            con = sqlite3.connect(title)
            con.execute("DROP TABLE IF EXISTS "+tablename)

            pd_sql.write_frame(portdf, tablename, con)
            con.close()
            print portdf
            return
    else:
        portdf = pd.DataFrame().fillna(0.0)


    gettime = foundap[foundap['title'] == stocktitle][-1:]['time'].values
    # print gettime[0]
    # print foundap[foundap['title'] == stocktitle][-1:],stockprice
    tmpdf = deepcopy(foundap[foundap['title'] == stocktitle][-1:])
    tmpdf['Price'] = pd.Series([stockprice], index=tmpdf.index)
    tmpdf['StockNum'] = pd.Series([stocknum], index=tmpdf.index)
    portdf = pd.concat([portdf,tmpdf])
    print portdf
    
    con = sqlite3.connect(title)
    con.execute("DROP TABLE IF EXISTS "+tablename)

    pd_sql.write_frame(portdf, tablename, con)
    con.close()

def portfoliosell(widget):
    print 'portfolio sell'
    
    global stockprice
    global stocktitle

    title = '../../data/port/port_db.sqlite'
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
            

    



run_button = widgets.ButtonWidget(description="Individual History Open")        
run_button.on_click(stocksignalhist)
display(run_button)

run_button2 = widgets.ButtonWidget(description="Portfolio Buy")        
run_button2.on_click(portfoliobuy)
display(run_button2)

run_button3 = widgets.ButtonWidget(description="Portfolio Sell")        
run_button3.on_click(portfoliosell)
display(run_button3)


run_button4 = widgets.ButtonWidget(description="Total Gain")        
run_button4.on_click(totalGain)
display(run_button4)


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