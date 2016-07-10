
import sys
sys.path.append('../../src/main/ver2_0/')
sys.path.append('../../lib/')

from stockcore import *
from tradingalgo import *
from data_mani import *
import stockcore as stcore
import tradingalgo as talgo 
import data_mani as dmani

# from selenium import webdriver
# from selenium.common.exceptions import TimeoutException
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from pytagcloud import create_tag_image, make_tags
# from pytagcloud.lang.counter import get_tag_counts

import webbrowser
import csv
import os
from datetime import datetime,timedelta

import sqlite3
import pandas.io.sql as pd_sql
import pandas as pd

from IPython.html.widgets import interact, interactive, fixed
from IPython.html import widgets
from IPython.display import clear_output, display, HTML
pd.set_option('display.width',500)

import urllib
import time
import xlrd
import xlwt
from xlutils.copy import copy 
from xlrd import open_workbook 
from xlwt import easyxf 
# import ver2_0_i as v20




global office
office = False


from IPython import parallel
clients = parallel.Client()
view = clients.direct_view()
view.block = False


def RunSimul(codearg,typearg,namearg,mode,dbmode,histmode,runcount,srcsite,writedblog,updbpattern,appenddb,startdatemode 
    ,dbtradinghist):

    try:
        start = time.clock()
        code = codearg #'097950'#'005930' #'005380'#009540 #036570
        if codearg == '005490' or codearg == '000660' or codearg == '068870'\
            or codearg == '078520' :
            srcsite = 2
        print 'typearg',typearg
        # symbol = codearg    
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
            startdate = '2008-01-01'
        else:
            startday =  datetime.today() - timedelta(days=150)
            startdate = str(startday).split(' ')[0]
        print 'startdate',startdate,codearg,namearg,type(namearg),typearg

        

        if typearg != 4:

            bars_org = stcore.fetchRealData(codearg,symbol,typearg,startdate)
            print 'fetch end'
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
        else:
            print 'getting world hist db ',codearg,startdate
            bars = Quandl.get(codearg,  trim_start=startdate, trim_end=datetime.today(),authtoken="")
    
            today  = datetime.today()
            todayweek = today.weekday()

            bars['week'] = bars.index.weekday
            tailweekdays = bars['week'][-5:]
            codearg = codearg.split('/')[1]

            

        # bars = []
        # startdate = '2013-01-01'
        # enddate = '2013-06-01'
        # bars =  Quandl.get(symbol, collapse='Daily', trim_start=startdate, trim_end=enddate,authtoken="")
        bars = bars.drop_duplicates()
        bars = bars.sort_index()
        print '----------final sorted bars---------'
        if startdatemode == 3:
            # endday_arg = datetime.datetime.strptime(endday_arg, "%Y-%m-%d %H:%M:%S")
            bars = bars[bars.index <= endday_arg]
        if namearg == 'kospi':
            bars = bars[:-1]
        print bars.tail()

        elapsed = (time.clock() - start)
        print 'real time data web gathering elapsed time:',elapsed

        # #file delete
        basepos = u"../../data/hist/"
        
        
        # deletename = basepos+u'hist_db_'+codearg+u'_'+namearg+u'.sqlite'
        # if os.path.isfile(deletename):
        #     os.remove(deletename)
        
        print 'hist saving start'
        # import sqlite3
        # import pandas.io.sql as pd_sql
        
        subdbname = 'hist_db_'+codearg+'.sqlite'
        if office == False:
            dbname = 'C:/Users/Administrator/Documents/IPython Notebooks/stock_simul/data/hist/'+subdbname
        else:
            dbname = 'C:/Users/AUTRON/Documents/IPython Notebooks/simul/data/hist/'+subdbname
        print dbname
        try:
            con = sqlite3.connect(dbname)
            tablename_base = 'result_'+codearg

            # for cnt in range(dblen):
            tablename = tablename_base
            # print 'writetable:',tablename
            con.execute("DROP TABLE IF EXISTS "+tablename)
            bars2 = bars.reset_index()
            print 'bars2'
            print bars2.tail()
            pd_sql.write_frame(bars2, tablename, con)

            
            con.close()        
        except Exception,e:
            PrintException()  
            con.close()   
        print 'histdb save done'
    except Exception,e:
        PrintException()    

def test2():
    print 'test2'

def RunSimul_Wrap(listarg):

 

    typearg = listarg[0]
    startIdx = listarg[1]
    endIdx = listarg[2]

   

    global office
    if office == False:
        book_kosdaq = xlrd.open_workbook("C:/Users/Administrator/Documents/IPython Notebooks/stock_simul/Kosdaq_symbols.xls")
        book_kospi = xlrd.open_workbook('C:/Users/Administrator/Documents/IPython Notebooks/stock_simul/Kospi_Symbols.xls')
        book_sp = xlrd.open_workbook('C:/Users/Administrator/Documents/IPython Notebooks/stock_simul/SP500.xls')
        book_nas = xlrd.open_workbook('C:/Users/Administrator/Documents/IPython Notebooks/stock_simul/NASDAQComposite.xls')

    else:
        book_kosdaq = xlrd.open_workbook("C:/Users/AUTRON/Documents/IPython Notebooks/simul/Kosdaq_symbols.xls")
        book_kospi = xlrd.open_workbook('C:/Users/AUTRON/Documents/IPython Notebooks/simul/Kospi_Symbols.xls')
        book_sp = xlrd.open_workbook('C:/Users/AUTRON/Documents/IPython Notebooks/simul/SP500.xls')
        book_nas = xlrd.open_workbook('C:/Users/AUTRON/Documents/IPython Notebooks/simul/NASDAQComposite.xls')

    sheet_kosdaq = book_kosdaq.sheet_by_name('kosdaq')
    sheet_kospi = book_kospi.sheet_by_name('kospi')    
    sheet_sp = book_sp.sheet_by_name('SP500')
    sheet_nas = book_nas.sheet_by_name('NASDAQComposite')

    srcsite = 1#google
    # srcsite = 2#yahoo
    runcount = 1
    writedblog = 'writedblog'
    # updbpattern = 'updbpattern'
    updbpattern = 'none'
    appenddb = 'appenddb'
    startdatemode = 1
    runcount = 1
#     dbtradinghist = 'dbtradinghist'
    dbtradinghist = 'none'
    histmode = 'histdb'
#     histmode = 'none'
    
  
    if typearg =='kospi':
        for cnt in range(startIdx,endIdx):
            code = '{0:06d}'.format(int(sheet_kospi.row_values(cnt)[0]))
            title = sheet_kospi.row_values(cnt)[1]
            try:
                RunSimul(code,1,title,'realtime','dbpattern',histmode,runcount,srcsite,writedblog,updbpattern,appenddb,startdatemode,
                         dbtradinghist)
                clear_output()
            except:
                # errortitles.append(title)
                pass   

    if typearg == 'kosdaq':
        for cnt in range(startIdx,endIdx):
            code = '{0:06d}'.format(int(sheet_kosdaq.row_values(cnt)[0]))
            title = sheet_kosdaq.row_values(cnt)[1]
            try:
                RunSimul(code,2,title,'realtime','dbpattern',histmode,runcount,srcsite,writedblog,updbpattern,appenddb,startdatemode,
                         dbtradinghist)
                clear_output()
            except:
                # errortitles.append(title)
                pass   

             
    if typearg =='kospi_index':
        for cnt in range(startIdx,endIdx):
            # code = '000000'
            # title = 'kospi'
            code = '069500'
            title = 'KODEX200'
            startdatemode = 1
            try:
                
                RunSimul(code,1,title,'realtime','dbpattern',histmode,runcount,srcsite\
                    ,writedblog,updbpattern,appenddb,startdatemode,
                         dbtradinghist)
                clear_output()
            except Exception,e:
                print e
                # errortitles.append(title)
                pass           

    if typearg =='sp':
        # print startIdx,endIdx
        for cnt in range(startIdx,endIdx):
            code = str(sheet_sp.row_values(cnt)[1])
            title = str(sheet_sp.row_values(cnt)[0])
            print code,title
            startdatemode = 1
            try:
                # test2()
                _RunSimul_world(code,4,title,'realtime','dbpattern',histmode,runcount,srcsite,writedblog,updbpattern,appenddb,startdatemode,dbtradinghist)
                clear_output()
            except Exception,e:
                # errortitles.append(title)
                PrintException()
                pass   

    if typearg =='dow_index':
        for cnt in range(startIdx,endIdx):
            
            code = 'GOOG/INDEXDJX_DJI'
            title = 'dow'
            startdatemode = 1
            try:
                
                RunSimul(code,4,title,'realtime','dbpattern',histmode,runcount,srcsite\
                    ,writedblog,updbpattern,appenddb,startdatemode,
                         dbtradinghist)
                clear_output()
            except Exception,e:
                print e
                # errortitles.append(title)
                pass  

    if typearg =='nas':
        # print startIdx,endIdx
        for cnt in range(startIdx,endIdx):
            code = str(sheet_nas.row_values(cnt)[1])
            title = str(sheet_nas.row_values(cnt)[0])
            print code,title
            startdatemode = 1
            try:
                # test2()
                _RunSimul_world(code,4,title,'realtime','dbpattern',histmode,runcount,srcsite,writedblog,updbpattern,appenddb,startdatemode,dbtradinghist)
                clear_output()
            except Exception,e:
                # errortitles.append(title)
                PrintException()
                pass   


def simulAuto():
    # book = xlrd.open_workbook("../../Kospi_symbols.xls")
    # sheet = book.sheet_by_name('kospi')

    
    
    
    start = time.clock()
    errortitles = []

    a_args = [['kospi',1,200]\
            ,['kospi',200,400]\
            ,['kospi',400,600]\
            ,['kospi',600,950]\
            ,['kosdaq',1,200]\
            ,['kosdaq',200,400]\
            ,['kosdaq',400,600]\
            ,['kosdaq',600,800]]    
           
    results = view.map(RunSimul_Wrap,a_args)

    while not results.ready():
        time.sleep(1)

    # results.display_outputs()    
    print "Results ready!"
    # totaldf = pd.DataFrame()
    # for tmpdf in results:
    #     totaldf = pd.concat([totaldf,tmpdf])
    
    elapsed = (time.clock() - start)
    print 'total run elapsed time:',elapsed
    
    # clients.shutdown(restart=True,hub=True)


def simulAuto_SP():
    
    start = time.clock()
    errortitles = []

    # a_args = [['sp',1,3]\
    #         ]    
    
    a_args = [['sp',1,100]\
            ,['sp',100,200]\
            ,['sp',200,300]\
            ,['sp',300,497]\
            ]           
    results = view.map(RunSimul_Wrap,a_args)
   

    while not results.ready():
        time.sleep(1)

    # results.display_outputs()    
    print "Results ready!"
    # totaldf = pd.DataFrame()
    # for tmpdf in results:
    #     totaldf = pd.concat([totaldf,tmpdf])
    
    elapsed = (time.clock() - start)
    print 'total run elapsed time:',elapsed
    
    # clients.shutdown(restart=True,hub=True)

def simulAuto_NAS():
    
    start = time.clock()
    errortitles = []

    # a_args = [['sp',1,3]\
    #         ]    
    
    a_args = [['nas',1,200]\
            ,['nas',200,400]\
            ,['nas',400,600]\
            ,['nas',600,800]\
            ,['nas',800,1000]\
            ,['nas',1000,1200]\
            ,['nas',1200,1400]\
            ,['nas',1400,1600]\

            ]           
    results = view.map(RunSimul_Wrap,a_args)
   

    while not results.ready():
        time.sleep(1)

    # results.display_outputs()    
    print "Results ready!"
    # totaldf = pd.DataFrame()
    # for tmpdf in results:
    #     totaldf = pd.concat([totaldf,tmpdf])
    
    elapsed = (time.clock() - start)
    print 'total run elapsed time:',elapsed
    
    # clients.shutdown(restart=True,hub=True)


def get_kospi_index():
    a_args = ['kospi_index',0,1]
    RunSimul_Wrap(a_args)  




def get_dow_index():
    a_args = ['dow_index',0,1]
    RunSimul_Wrap(a_args)  



def test():
    RunSimul_Wrap(['sp',1,3])    

def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print 'EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj)




def _RunSimul_world(codearg,typearg,namearg,mode,dbmode,histmode,runcount,srcsite,writedblog,updbpattern,appenddb,startdatemode,dbtradinghist):
    
    try:
        start = time.clock()
        code = codearg #'097950'#'005930' #'005380'#009540 #036570
        symbol = codearg
            

        if startdatemode == 1:            
            startdate = '2008-01-01'
        else:
            startday =  datetime.today() - timedelta(days=150)
            startdate = str(startday).split(' ')[0]
        print 'startdate',startdate,codearg,namearg,type(namearg),typearg

        
        
        print 'getting world hist db ',codearg,startdate
        bars = Quandl.get(codearg,  trim_start=startdate, trim_end=datetime.today(),authtoken="")

        today  = datetime.today()
        todayweek = today.weekday()

        bars['week'] = bars.index.weekday
        tailweekdays = bars['week'][-5:]
        codearg = codearg.split('/')[1]

            

        # bars = []
        # startdate = '2013-01-01'
        # enddate = '2013-06-01'
        # bars =  Quandl.get(symbol, collapse='Daily', trim_start=startdate, trim_end=enddate,authtoken="")
        bars = bars.drop_duplicates()
        bars = bars.sort_index()
        print '----------final sorted bars---------'
        if startdatemode == 3:
            # endday_arg = datetime.datetime.strptime(endday_arg, "%Y-%m-%d %H:%M:%S")
            bars = bars[bars.index <= endday_arg]
        if namearg == 'kospi':
            bars = bars[:-1]
        print bars.tail()

        elapsed = (time.clock() - start)
        print 'real time data web gathering elapsed time:',elapsed

        # #file delete
        basepos = u"../../data/hist/"
        
        
        # deletename = basepos+u'hist_db_'+codearg+u'_'+namearg+u'.sqlite'
        # if os.path.isfile(deletename):
        #     os.remove(deletename)
        
        print 'hist saving start'
        # import sqlite3
        # import pandas.io.sql as pd_sql
        
        subdbname = 'hist_db_'+codearg+'.sqlite'
        if office == False:
            dbname = 'C:/Users/Administrator/Documents/IPython Notebooks/stock_simul/data/hist/'+subdbname
        else:
            dbname = 'C:/Users/AUTRON/Documents/IPython Notebooks/simul/data/hist/'+subdbname
        print dbname
        try:
            con = sqlite3.connect(dbname)
            tablename_base = 'result_'+codearg

            # for cnt in range(dblen):
            tablename = tablename_base
            # print 'writetable:',tablename
            con.execute("DROP TABLE IF EXISTS "+tablename)
            bars2 = bars.reset_index()
            print 'bars2'
            print bars2.tail()
            pd_sql.write_frame(bars2, tablename, con)

            
            con.close()        
        except Exception,e:
            PrintException()  
            con.close()   
        print 'histdb save done'
    except Exception,e:
        PrintException()    
