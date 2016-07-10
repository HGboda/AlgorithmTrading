
# %matplotlib inline
# from matplotlib import interactive
# interactive(True)

# from guiqwt.plot import CurveDialog
# from guiqwt.builder import make

# pd.set_option('display.width',500)
# import sys
# sys.path.append('../../lib/')
from stockcore import *
from tradingalgo import *
from data_mani import *
import stockcore as stcore
import tradingalgo as talgo 
import data_mani as dmani

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
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
import ver2_0_i as v20




global office
office = False


def RunSimul(codearg,typearg,namearg,mode,dbmode,histmode,runcount,srcsite,writedblog,updbpattern,appenddb,startdatemode 
    ,dbtradinghist,plotly,*args):


    if typearg == 4:# DOW,NASDAQ,S&P500
        startdate= '2014-01-01'
        arg_index = 0
        stdarg = args[arg_index]
        arg_index += 1
        smallarg = args[arg_index]
        arg_index += 1
        dayselect = args[arg_index]
        arg_index += 1
        tangentmode = args[arg_index]
        print 'stdarg',stdarg,'tangentmode',tangentmode
        tangentmode = 'tan_gen'
        if namearg == 'dow':
            symbol = 'GOOG/INDEXDJX_DJI'
            bars =  Quandl.get(symbol, collapse='Daily', trim_start=startdate, trim_end=datetime.today(),authtoken="")
        elif namearg == 'nasdaq':
            symbol = '^IXIC'
            import pandas.io.data as web
            bars = web.get_data_yahoo(symbol,startdate)
        elif namearg == 'sandp':
            symbol = '^GSPC'
            import pandas.io.data as web
            bars = web.get_data_yahoo(symbol,startdate)

        if mode =='dbpattern'  or dbmode == 'dbpattern':        
            if updbpattern == 'none':
                print 'read DB patterns'
                patternAr, extractid= stcore.ReadPatternsFromDB(codearg,typearg,namearg,mode)
                patternAppendAr = stcore.ReadPatternsAppendFromDB(codearg,namearg)
            elif updbpattern == 'updbpattern':
                print 'read UP DB patterns'
                patternAr, extractid= stcore.ReadUpPatternsFromDB(codearg,typearg,namearg,mode)

            if patternAr == -1:
                print 'real time gen db pattern'
                startdate = '2011-01-01'
                dbmode  = 'none'


    else:
        

        bars,patternAr, extractid,patternAppendAr,bars_25p,bars_50p\
        ,bars_75p,bars_90p,tangent_25p,tangent_50p,tangentmode\
        ,startdate,dbmode,stdarg,smallarg,dayselect,tangentmode =\
            stcore._inRunSimul_FetchData(codearg,typearg,namearg,mode,dbmode,histmode,runcount,srcsite,writedblog,updbpattern,appenddb,startdatemode \
                            ,dbtradinghist,plotly,*args)
        # print 'stdarg',stdarg,'smallarg',smallarg,'dayselect',dayselect,'tangentmode',tangentmode
        
    
    # bars = bars.drop(bars.index[-2])    
    today  = datetime.today()
    todayweek = today.weekday()

    bars['week'] = bars.index.weekday
    tailweekdays = bars['week'][-5:]
    # print tailweekdays

    # if 0 <= todayweek <=4 :
    #     for cnt in range(0,len(tailweekdays)):
    #         day = cnt +1 
    #         # print gbars2['week'][-1*day]
    #         checkday = bars['week'][-1*day]
            
    #         # print todayweek,checkday,bars.index[-1*day]
    #         if todayweek != checkday:
    #             raise Exception("week check error")
    #         if todayweek == 0:
    #             todayweek = 4
    #         else:
    #             todayweek = todayweek - 1

    # bars = bars[:len(bars)-100]  
    # bars = bars[:'2015-07-07']
    # print bars.tail()  
    '''
    test code for inflection point
    '''
    # bars = bars.drop(bars.tail(1).index)
    # print '----------final test bars---------'
    # print bars[['Close']].tail() 
    # bars = bars[:30]   
    '''
    end test code
    '''
    bars['bench'] = bars['Close'].pct_change().cumsum()
    bars['benchdiff'] = bars['bench'].diff()



    day = len(bars)-1
    # if day < 25:
    #     allslopeX = np.arange(1,day+1,1)    
    # elif day >= 25:
    #     allslopeX = np.arange(day-25,day+1,1)    

    if day < 25:
        partslopeX = np.arange(1,day+1,1)    
    elif day >= 25:
        partslopeX = np.arange(day-25,day+1,1)        

    param_partslope = np.polyfit(partslopeX,bars['bench'][partslopeX],1)

    allslopeX = np.arange(1,day+1,1)        
    param_allslope = np.polyfit(allslopeX,bars['bench'][allslopeX],1)

    tempdf = pd.DataFrame({'partX':bars['bench'][partslopeX],'allX':bars['bench'][allslopeX]})
    tempdf = tempdf.fillna(0.0)
    # corrX = tempdf['partX'].corr(tempdf['allX'])
    corrX = 0.0

    recentBench = bars['bench'][-1] - bars['bench'][-5]
    benchstd = bars['bench'].std()

    bars['bench_short'] =  pd.ewma(bars['bench'][:day+1],span = 20)
    bars['bench_long'] =  pd.ewma(bars['bench'][:day+1],span = 30)
    
    bench_sl = 0
    if bars['bench_short'][-1] > bars['bench_long'][-1]:
        bench_sl = 1
    return bars['bench'][-1],param_partslope[0],param_allslope[0],corrX,recentBench,benchstd,bench_sl

def RunSimul_test(codearg,typearg,namearg,mode,dbmode,histmode,runcount,srcsite,writedblog,updbpattern,appenddb,startdatemode 
    ,dbtradinghist,plotly,*args):


    if typearg == 4:# DOW,NASDAQ,S&P500
        startdate= '2014-01-01'
        arg_index = 0
        stdarg = args[arg_index]
        arg_index += 1
        smallarg = args[arg_index]
        arg_index += 1
        dayselect = args[arg_index]
        arg_index += 1
        tangentmode = args[arg_index]
        print 'stdarg',stdarg,'tangentmode',tangentmode
        tangentmode = 'tan_gen'
        if namearg == 'dow':
            symbol = 'GOOG/INDEXDJX_DJI'
            bars =  Quandl.get(symbol, collapse='Daily', trim_start=startdate, trim_end=datetime.today(),authtoken="")
        elif namearg == 'nasdaq':
            symbol = '^IXIC'
            import pandas.io.data as web
            bars = web.get_data_yahoo(symbol,startdate)
        elif namearg == 'sandp':
            symbol = '^GSPC'
            import pandas.io.data as web
            bars = web.get_data_yahoo(symbol,startdate)

        if mode =='dbpattern'  or dbmode == 'dbpattern':        
            if updbpattern == 'none':
                print 'read DB patterns'
                patternAr, extractid= stcore.ReadPatternsFromDB(codearg,typearg,namearg,mode)
                patternAppendAr = stcore.ReadPatternsAppendFromDB(codearg,namearg)
            elif updbpattern == 'updbpattern':
                print 'read UP DB patterns'
                patternAr, extractid= stcore.ReadUpPatternsFromDB(codearg,typearg,namearg,mode)

            if patternAr == -1:
                print 'real time gen db pattern'
                startdate = '2011-01-01'
                dbmode  = 'none'


    else:
        

        bars,patternAr, extractid,patternAppendAr,bars_25p,bars_50p\
        ,bars_75p,bars_90p,tangent_25p,tangent_50p,tangentmode\
        ,startdate,dbmode,stdarg,smallarg,dayselect,tangentmode =\
            stcore._inRunSimul_FetchData(codearg,typearg,namearg,mode,dbmode,histmode,runcount,srcsite,writedblog,updbpattern,appenddb,startdatemode \
                            ,dbtradinghist,plotly,*args)
        # print 'stdarg',stdarg,'smallarg',smallarg,'dayselect',dayselect,'tangentmode',tangentmode
        
    
    # bars = bars.drop(bars.index[-2])    
    today  = datetime.today()
    todayweek = today.weekday()

    bars['week'] = bars.index.weekday
    tailweekdays = bars['week'][-5:]
    # print tailweekdays

    # if 0 <= todayweek <=4 :
    #     for cnt in range(0,len(tailweekdays)):
    #         day = cnt +1 
    #         # print gbars2['week'][-1*day]
    #         checkday = bars['week'][-1*day]
            
    #         # print todayweek,checkday,bars.index[-1*day]
    #         if todayweek != checkday:
    #             raise Exception("week check error")
    #         if todayweek == 0:
    #             todayweek = 4
    #         else:
    #             todayweek = todayweek - 1

    # bars = bars[:len(bars)-100]  
    # bars = bars[:'2015-07-07']
    # print bars.tail()  
    '''
    test code for inflection point
    '''
    # bars = bars.drop(bars.tail(1).index)
    # print '----------final test bars---------'
    # print bars.tail()    
    '''
    end test code
    '''
    bars['bench'] = bars['Close'].pct_change().cumsum()
    bars['benchdiff'] = bars['bench'].diff()



    day = len(bars)-1
    # if day < 25:
    #     allslopeX = np.arange(1,day+1,1)    
    # elif day >= 25:
    #     allslopeX = np.arange(day-25,day+1,1)    

    if day < 15:
        partslopeX = np.arange(1,day+1,1)    
    elif day >= 15:
        partslopeX = np.arange(day-15,day+1,1)        

    param_partslope = np.polyfit(partslopeX,bars['bench'][partslopeX],1)

    allslopeX = np.arange(1,day+1,1)        
    param_allslope = np.polyfit(allslopeX,bars['bench'][allslopeX],1)

    tempdf = pd.DataFrame({'partX':bars['bench'][partslopeX],'allX':bars['bench'][allslopeX]})
    tempdf = tempdf.fillna(0.0)
    corrX = tempdf['partX'].corr(tempdf['allX'])
    return bars['bench'][-1],param_partslope[0],param_allslope[0],corrX

''' 
get last bench, last allslope
'''
def screenGetData(dbdf,saveType):
    try:
        global office
        
        clear_output()
        # display(HTML(dbdf.to_html()))    

        if office == False:
            book_kosdaq = xlrd.open_workbook("C:/Users/Administrator/Documents/IPython Notebooks/stock_simul/Kosdaq_symbols.xls")
            book_kospi = xlrd.open_workbook('C:/Users/Administrator/Documents/IPython Notebooks/stock_simul/Kospi_Symbols.xls')
        else:
            book_kosdaq = xlrd.open_workbook("C:/Users/AUTRON/Documents/IPython Notebooks/simul/Kosdaq_symbols.xls")
            book_kospi = xlrd.open_workbook('C:/Users/AUTRON/Documents/IPython Notebooks/simul/Kospi_Symbols.xls')

        sheet_kosdaq = book_kosdaq.sheet_by_name('kosdaq')
        sheet_kospi = book_kospi.sheet_by_name('kospi')


        algo1_totalaccumgain = []
        algo1_totaltradinggain = []
        algo1_totaltradingnum = []        
        
        algo_title = []
        algo_stance = []    
        errortitles = []
        algo_benchmark = []
        totalsum = []
        algo_lastDate= []
        algo_macdsig = []
        algo_slowsig = []
        algo_paramallslope = []
        algo_parampartslope = []
        algo_corrX = []
        algo_recentBench = []
        algo_benchstd = []
        algo_sl = []
        total_sum = 0
        total_cnt = 0

        # dbdf = dbdf[:10]
        for title in dbdf['title']:
            if ' ' in title:
                title  = title.replace(' ','')
            if '&' in title:
                title  = title.replace('&','and')
            if '-' in title:
                title  = title.replace('-','')    

            for cnt in range(sheet_kospi.nrows):
            
                if sheet_kospi.row_values(cnt)[1] == title:
                    
                    code = '{0:06d}'.format(int(sheet_kospi.row_values(cnt)[0]))
                    name = sheet_kospi.row_values(cnt)[1]
                    print code,name
                    markettype = 1
                    break

            for cnt in range(sheet_kosdaq.nrows):
                
                if sheet_kosdaq.row_values(cnt)[1] == title:
                    
                    code = '{0:06d}'.format(int(sheet_kosdaq.row_values(cnt)[0]))
                    name = sheet_kosdaq.row_values(cnt)[1]
                    print code,name
                    markettype = 2
                    break        

            
            startdatemode = 2
            dbtradinghist = 'none'
            histmode = 'none'
            plotly = 'plotly'
            stdmode = 'stddb'
            tangentmode = 'tangentdb'        
            daych  =0
            runcount = 0
            srcsite = 1#google
            # srcsite = 2#yahoo
            writedblog = 'none'
            updbpattern = 'none'
            appenddb = 'none'

            try:
                lastbench,partslope,allslope,corrX,recentBench,benchstd,benchsl  = RunSimul(str(code),markettype,name,'realtime','dbpattern',histmode,runcount,srcsite,writedblog,updbpattern\
                            ,appenddb,startdatemode,\
                             dbtradinghist,plotly,stdmode,'none',daych,tangentmode)
                algo_title.append(title)
                algo_benchmark.append(lastbench)
                algo_parampartslope.append(partslope)
                algo_paramallslope.append(allslope)
                algo_corrX.append(corrX)
                algo_recentBench.append(recentBench)
                algo_benchstd.append(benchst)
                algo_sl.append(benchsl)
                clear_output()
                
            except:
                errortitles.append(title)
                stcore.PrintException()
                pass   

        tradingresultdf = pd.DataFrame({'title':algo_title\
            ,'benchmark':algo_benchmark\
            ,'partslope':algo_parampartslope\
            ,'allslope':algo_paramallslope\
            ,'corrX':algo_corrX\
            ,'recentBench':algo_recentBench\
            ,'benchstd':algo_benchstd\
            ,'benchsl':algo_sl\
            }\
        )    
        # display(HTML('<font size=1>'+tradingresultdf._repr_html_()+'</font>'))
        display(HTML(tradingresultdf.to_html()))        

        if saveType == True:
            try:
                # todaydate = datetime.today()
                if office == False:
                    dbname = 'C:/Users/Administrator/Documents/IPython Notebooks/stock_simul/data/analysis/screen/'+str(datetime.today()).split(' ')[0].replace('-','')+'.sqlite'
                else:
                    dbname = 'C:/Users/AUTRON/Documents/IPython Notebooks/simul/data/analysis/screen/'+str(datetime.today()).split(' ')[0].replace('-','')+'.sqlite'
                # if os.path.isfile(dbname):
                #     os.remove(dbname)
                con = sqlite3.connect(dbname)
                tablename = 'screen'
                # tablename = tableName#'analysis_table_0'
                # print 'tablename',tablename,type(tablename)
                con.execute("DROP TABLE IF EXISTS "+tablename)
                
                pd_sql.write_frame(tradingresultdf, tablename, con)
                con.close()

                
            except Exception,e:
                con.close()
                print 'analysis db error',e    
                stcore.PrintException()
                pass
        
        return tradingresultdf

    except Exception,e:
        stcore.PrintException()
        con.close()



def screenGetData_fromDefault(saveType):
    try:
        global office
        
        clear_output()
        # display(HTML(dbdf.to_html()))    

        if office == False:
            book_kosdaq = xlrd.open_workbook("C:/Users/Administrator/Documents/IPython Notebooks/stock_simul/Kosdaq_symbols.xls")
            book_kospi = xlrd.open_workbook('C:/Users/Administrator/Documents/IPython Notebooks/stock_simul/Kospi_Symbols.xls')
        else:
            book_kosdaq = xlrd.open_workbook("C:/Users/AUTRON/Documents/IPython Notebooks/simul/Kosdaq_symbols.xls")
            book_kospi = xlrd.open_workbook('C:/Users/AUTRON/Documents/IPython Notebooks/simul/Kospi_Symbols.xls')

        sheet_kosdaq = book_kosdaq.sheet_by_name('kosdaq')
        sheet_kospi = book_kospi.sheet_by_name('kospi')


        algo1_totalaccumgain = []
        algo1_totaltradinggain = []
        algo1_totaltradingnum = []        
        
        algo_title = []
        algo_stance = []    
        errortitles = []
        algo_benchmark = []
        totalsum = []
        algo_lastDate= []
        algo_macdsig = []
        algo_slowsig = []
        algo_parampartslope = []
        algo_paramallslope = []
        algo_corrX = []
        algo_recentBench = []
        algo_benchstd = []
        algo_sl = []
        total_sum = 0
        total_cnt = 0

        # dbdf = dbdf[:10]
        sheet_kospi_length = 700
        # totalcount =  0
        for cnt in range(1,sheet_kospi_length):
            # print sheet_kospi.row_values(cnt)[1],sheet_kospi.row_values(cnt)[0],'{0:06d}'.format(int(sheet_kospi.row_values(cnt)[0]))
            name = sheet_kospi.row_values(cnt)[1]
            title = name
            code = '{0:06d}'.format(int(sheet_kospi.row_values(cnt)[0]))
            print code,name
            markettype = 1
            
            
            startdatemode = 2
            dbtradinghist = 'none'
            histmode = 'none'
            plotly = 'plotly'
            stdmode = 'stddb'
            tangentmode = 'tangentdb'        
            daych  =0
            runcount = 0
            srcsite = 1#google
            # srcsite = 2#yahoo
            writedblog = 'none'
            updbpattern = 'none'
            appenddb = 'none'
            
            try:
                lastbench,partslope,allslope,corrX,recentBench,benchstd,benchsl = RunSimul(str(code),markettype,name,'realtime','dbpattern',histmode,runcount,srcsite,writedblog,updbpattern\
                            ,appenddb,startdatemode,\
                             dbtradinghist,plotly,stdmode,'none',daych,tangentmode)
                algo_title.append(title)
                algo_benchmark.append(lastbench)
                algo_parampartslope.append(partslope)
                algo_paramallslope.append(allslope)
                algo_corrX.append(corrX)
                algo_recentBench.append(recentBench)
                algo_benchstd.append(benchstd)
                algo_sl.append(benchsl)
                clear_output()
                
            except:
                errortitles.append(title)
                stcore.PrintException()
                pass   


        sheet_kosdaq_length = 300
        # totalcount =  0
        for cnt in range(1,sheet_kosdaq_length):
            # print sheet_kospi.row_values(cnt)[1],sheet_kospi.row_values(cnt)[0],'{0:06d}'.format(int(sheet_kospi.row_values(cnt)[0]))
            name = sheet_kosdaq.row_values(cnt)[1]
            title = name
            code = '{0:06d}'.format(int(sheet_kosdaq.row_values(cnt)[0]))
            print code,name
            markettype = 2
            
            
            startdatemode = 2
            dbtradinghist = 'none'
            histmode = 'none'
            plotly = 'plotly'
            stdmode = 'stddb'
            tangentmode = 'tangentdb'        
            daych  =0
            runcount = 0
            srcsite = 1#google
            # srcsite = 2#yahoo
            writedblog = 'none'
            updbpattern = 'none'
            appenddb = 'none'
            
            try:
                lastbench,partslope,allslope,corrX,recentBench,benchstd,benchsl = RunSimul(str(code),markettype,name,'realtime','dbpattern',histmode,runcount,srcsite,writedblog,updbpattern\
                            ,appenddb,startdatemode,\
                             dbtradinghist,plotly,stdmode,'none',daych,tangentmode)
                algo_title.append(title)
                algo_benchmark.append(lastbench)
                algo_parampartslope.append(partslope)
                algo_paramallslope.append(allslope)
                algo_corrX.append(corrX)
                algo_recentBench.append(recentBench)
                algo_benchstd.append(benchstd)
                algo_sl.append(benchsl)
                clear_output()
                
            except:
                errortitles.append(title)
                stcore.PrintException()
                pass                   
        tradingresultdf = pd.DataFrame({'title':algo_title\
            ,'benchmark':algo_benchmark\
            ,'partslope':algo_parampartslope\
            ,'allslope':algo_paramallslope\
            ,'corrX':algo_corrX\
            ,'recentBench':algo_recentBench\
            ,'benchstd':algo_benchstd\
            ,'benchsl':algo_sl\
            }\
        )    
        # display(HTML('<font size=1>'+tradingresultdf._repr_html_()+'</font>'))
        display(HTML(tradingresultdf.to_html()))        

        if saveType == True:
            try:
                # todaydate = datetime.today()
                if office == False:
                    dbname = 'C:/Users/Administrator/Documents/IPython Notebooks/stock_simul/data/analysis/screen/'+str(datetime.today()).split(' ')[0].replace('-','')+'_default.sqlite'
                else:
                    dbname = 'C:/Users/AUTRON/Documents/IPython Notebooks/simul/data/analysis/screen/'+str(datetime.today()).split(' ')[0].replace('-','')+'_default.sqlite'
                # if os.path.isfile(dbname):
                #     os.remove(dbname)
                con = sqlite3.connect(dbname)
                tablename = 'screen'
                # tablename = tableName#'analysis_table_0'
                # print 'tablename',tablename,type(tablename)
                con.execute("DROP TABLE IF EXISTS "+tablename)
                
                pd_sql.write_frame(tradingresultdf, tablename, con)
                con.close()

                
            except Exception,e:
                con.close()
                print 'analysis db error',e    
                stcore.PrintException()
                pass
        
        return tradingresultdf
        
    except Exception,e:
        stcore.PrintException()
        con.close()


# def screenGetData_fromDefault_parallel(sel_type,startIdx,endIdx):
def screenGetData_fromDefault_parallel(listarg):    
    try:
        # print sel_type,startIdx
        print listarg[0],listarg[1],listarg[2]
        sel_type = listarg[0]
        startIdx = listarg[1]
        endIdx = listarg[2]
        startdate_arg = listarg[3]
        endday_arg = listarg[4]
        histmode_arg = listarg[5]

        global office
        office = False
        # clear_output()
        # display(HTML(dbdf.to_html()))    

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


        algo1_totalaccumgain = []
        algo1_totaltradinggain = []
        algo1_totaltradingnum = []        
        
        algo_title = []
        algo_stance = []    
        errortitles = []
        algo_benchmark = []
        totalsum = []
        algo_lastDate= []
        algo_macdsig = []
        algo_slowsig = []
        algo_parampartslope = []
        algo_paramallslope = []
        algo_corrX = []
        algo_recentBench = []
        algo_benchstd = []
        algo_sl = []
        total_sum = 0
        total_cnt = 0

        # dbdf = dbdf[:10]
        # sheet_kospi_length = 700
        # totalcount =  0
        
        if sel_type == 'kospi':
            
            for cnt in range(startIdx,endIdx):
                # print sheet_kospi.row_values(cnt)[1],sheet_kospi.row_values(cnt)[0],'{0:06d}'.format(int(sheet_kospi.row_values(cnt)[0]))
                name = sheet_kospi.row_values(cnt)[1]
                title = name
                code = '{0:06d}'.format(int(sheet_kospi.row_values(cnt)[0]))
                print code,name
                markettype = 1
                
                
                startdatemode = 3
                dbtradinghist = 'none'
                histmode = histmode_arg
                plotly = 'plotly'
                stdmode = 'stddb'
                tangentmode = 'tangentdb'        
                daych  =0
                runcount = 0
                srcsite = 1#google
                # srcsite = 2#yahoo
                writedblog = 'none'
                updbpattern = 'none'
                appenddb = 'none'
                
                try:
                    lastbench,partslope,allslope,corrX,recentBench,benchstd,benchsl = RunSimul(str(code),markettype,name,'realtime','dbpattern',histmode,runcount,srcsite,writedblog,updbpattern\
                                ,appenddb,startdatemode,\
                                 dbtradinghist,plotly,stdmode,'none',daych,tangentmode,startdate_arg,endday_arg)
                    algo_title.append(title)
                    algo_benchmark.append(lastbench)
                    algo_parampartslope.append(partslope)
                    algo_paramallslope.append(allslope)
                    algo_corrX.append(corrX)
                    algo_recentBench.append(recentBench)
                    algo_benchstd.append(benchstd)
                    algo_sl.append(benchsl)
                    clear_output()
                    
                except:
                    errortitles.append(title)
                    stcore.PrintException()
                    pass   


        # sheet_kosdaq_length = 300
        # totalcount =  0
        if sel_type == 'kosdaq':
            for cnt in range(startIdx,endIdx):
                # print sheet_kospi.row_values(cnt)[1],sheet_kospi.row_values(cnt)[0],'{0:06d}'.format(int(sheet_kospi.row_values(cnt)[0]))
                name = sheet_kosdaq.row_values(cnt)[1]
                title = name
                code = '{0:06d}'.format(int(sheet_kosdaq.row_values(cnt)[0]))
                print code,name
                markettype = 2
                
                
                startdatemode = 3
                dbtradinghist = 'none'
                histmode = histmode_arg
                plotly = 'plotly'
                stdmode = 'stddb'
                tangentmode = 'tangentdb'        
                daych  =0
                runcount = 0
                srcsite = 1#google
                # srcsite = 2#yahoo
                writedblog = 'none'
                updbpattern = 'none'
                appenddb = 'none'
                
                try:
                    lastbench,partslope,allslope,corrX,recentBench,benchstd,benchsl = RunSimul(str(code),markettype,name,'realtime','dbpattern',histmode,runcount,srcsite,writedblog,updbpattern\
                                ,appenddb,startdatemode,\
                                 dbtradinghist,plotly,stdmode,'none',daych,tangentmode,startdate_arg,endday_arg)
                    algo_title.append(title)
                    algo_benchmark.append(lastbench)
                    algo_parampartslope.append(partslope)
                    algo_paramallslope.append(allslope)
                    algo_corrX.append(corrX)
                    algo_recentBench.append(recentBench)
                    algo_benchstd.append(benchstd)
                    algo_sl.append(benchsl)
                    clear_output()
                    
                except:
                    errortitles.append(title)
                    stcore.PrintException()
                    pass                   

        

        if sel_type == 'sp':
            
            for cnt in range(startIdx,endIdx):
                # print sheet_sp.row_values(cnt)[1],sheet_sp.row_values(cnt)[0],'{0:06d}'.format(int(sheet_sp.row_values(cnt)[0]))
                name = sheet_sp.row_values(cnt)[1]

                code = str(sheet_sp.row_values(cnt)[1])
                name = str(sheet_sp.row_values(cnt)[0])
                code = code.split('/')[1]

                title = name
                # code = '{0:06d}'.format(int(sheet_sp.row_values(cnt)[0]))
                print code,name
                markettype = 1
                
                
                startdatemode = 3
                dbtradinghist = 'none'
                histmode = histmode_arg
                plotly = 'plotly'
                stdmode = 'stddb'
                tangentmode = 'tangentdb'        
                daych  =0
                runcount = 0
                srcsite = 1#google
                # srcsite = 2#yahoo
                writedblog = 'none'
                updbpattern = 'none'
                appenddb = 'none'
                
                try:
                    lastbench,partslope,allslope,corrX,recentBench,benchstd,benchsl = RunSimul(code,markettype,name,'realtime','dbpattern',histmode,runcount,srcsite,writedblog,updbpattern\
                                ,appenddb,startdatemode,\
                                 dbtradinghist,plotly,stdmode,'none',daych,tangentmode,startdate_arg,endday_arg)
                    algo_title.append(title)
                    algo_benchmark.append(lastbench)
                    algo_parampartslope.append(partslope)
                    algo_paramallslope.append(allslope)
                    algo_corrX.append(corrX)
                    algo_recentBench.append(recentBench)
                    algo_benchstd.append(benchstd)
                    algo_sl.append(benchsl)
                    clear_output()
                    
                except:
                    errortitles.append(title)
                    stcore.PrintException()
                    pass   
        

        if sel_type == 'nas':
            
            for cnt in range(startIdx,endIdx):
                # print sheet_nas.row_values(cnt)[1],sheet_nas.row_values(cnt)[0],'{0:06d}'.format(int(sheet_nas.row_values(cnt)[0]))
                name = sheet_nas.row_values(cnt)[1]

                code = str(sheet_nas.row_values(cnt)[1])
                name = str(sheet_nas.row_values(cnt)[0])
                code = code.split('/')[1]

                title = name
                # code = '{0:06d}'.format(int(sheet_nas.row_values(cnt)[0]))
                print code,name
                markettype = 1
                
                
                startdatemode = 3
                dbtradinghist = 'none'
                histmode = histmode_arg
                plotly = 'plotly'
                stdmode = 'stddb'
                tangentmode = 'tangentdb'        
                daych  =0
                runcount = 0
                srcsite = 1#google
                # srcsite = 2#yahoo
                writedblog = 'none'
                updbpattern = 'none'
                appenddb = 'none'
                
                try:
                    lastbench,partslope,allslope,corrX,recentBench,benchstd,benchsl = RunSimul(code,markettype,name,'realtime','dbpattern',histmode,runcount,srcsite,writedblog,updbpattern\
                                ,appenddb,startdatemode,\
                                 dbtradinghist,plotly,stdmode,'none',daych,tangentmode,startdate_arg,endday_arg)
                    algo_title.append(title)
                    algo_benchmark.append(lastbench)
                    algo_parampartslope.append(partslope)
                    algo_paramallslope.append(allslope)
                    algo_corrX.append(corrX)
                    algo_recentBench.append(recentBench)
                    algo_benchstd.append(benchstd)
                    algo_sl.append(benchsl)
                    clear_output()
                    
                except:
                    errortitles.append(title)
                    stcore.PrintException()
                    pass   
            
    

        tradingresultdf = pd.DataFrame({'title':algo_title\
            ,'benchmark':algo_benchmark\
            ,'partslope':algo_parampartslope\
            ,'allslope':algo_paramallslope\
            ,'corrX':algo_corrX\
            ,'recentBench':algo_recentBench\
            ,'benchstd':algo_benchstd\
            ,'benchsl':algo_sl\
            }\
        )    
        # display(HTML('<font size=1>'+tradingresultdf._repr_html_()+'</font>'))
        display(HTML(tradingresultdf.to_html()))    

        '''
        try:
            if len(errortitles) > 0:
                if office == False:
                    dbname = 'C:/Users/Administrator/Documents/IPython Notebooks/stock_simul/data/error/scraperrortitle'+'_'+str(startIdx)+'_'+str(endIdx)+'.sqlite'
                else:
                    dbname = 'C:/Users/AUTRON/Documents/IPython Notebooks/simul/data/error/scraperrortitle'+'_'+str(startIdx)+'_'+str(endIdx)+'.sqlite'

                con = sqlite3.connect(dbname)
                tablename = 'scraperrortitle'
                
                con.execute("DROP TABLE IF EXISTS "+tablename)
                errordf = pd.DataFrame({'titles':errortitles})

                pd_sql.write_frame(errordf, tablename, con)
                con.close()

            
        except Exception,e:
            con.close()
            stcore.PrintException()
            pass  
        '''

            
        '''
        if saveType == True:
            try:
                # todaydate = datetime.today()
                if office == False:
                    dbname = 'C:/Users/Administrator/Documents/IPython Notebooks/stock_simul/data/analysis/screen/'+str(datetime.today()).split(' ')[0].replace('-','')+'_default.sqlite'
                else:
                    dbname = 'C:/Users/AUTRON/Documents/IPython Notebooks/simul/data/analysis/screen/'+str(datetime.today()).split(' ')[0].replace('-','')+'_default.sqlite'
                # if os.path.isfile(dbname):
                #     os.remove(dbname)
                con = sqlite3.connect(dbname)
                tablename = 'screen'
                # tablename = tableName#'analysis_table_0'
                # print 'tablename',tablename,type(tablename)
                con.execute("DROP TABLE IF EXISTS "+tablename)
                
                pd_sql.write_frame(tradingresultdf, tablename, con)
                con.close()

                
            except Exception,e:
                con.close()
                print 'analysis db error',e    
                stcore.PrintException()
                pass
        '''
        return tradingresultdf
        
    except Exception,e:
        stcore.PrintException()
        # con.close()




def screenAutoTest(dbdf):
    try:
        global office
        # dbname = 'screen.sqlite'
        # tablename = 'screen'

        # con = sqlite3.connect("../../data/analysis/"+dbname)

        # dbdf = pd_sql.read_frame("SELECT * from "+tablename, con)

        # dbdf = dbdf[dbdf['allslope'] > 0.0]
        
        # display(HTML(dbdf.to_html()))    

        # print 'dbdf',dbdf
        if office == False:
            book_kosdaq = xlrd.open_workbook("C:/Users/Administrator/Documents/IPython Notebooks/stock_simul/Kosdaq_symbols.xls")
            book_kospi = xlrd.open_workbook('C:/Users/Administrator/Documents/IPython Notebooks/stock_simul/Kospi_Symbols.xls')
        else:    
            book_kosdaq = xlrd.open_workbook("C:/Users/AUTRON/Documents/IPython Notebooks/simul/Kosdaq_symbols.xls")
            book_kospi = xlrd.open_workbook('C:/Users/AUTRON/Documents/IPython Notebooks/simul/Kospi_Symbols.xls')

        sheet_kosdaq = book_kosdaq.sheet_by_name('kosdaq')
        sheet_kospi = book_kospi.sheet_by_name('kospi')


        algo1_totalaccumgain = []
        algo1_totaltradinggain = []
        algo1_totaltradingnum = []        
        
        algo_title = []
        algo_stance = []    
        errortitles = []
        algo_benchmark = []
        totalsum = []
        algo_lastDate= []
        algo_macdsig = []
        algo_slowsig = []
        total_sum = 0
        total_cnt = 0

        # dbdf = dbdf[:70]
        for title in dbdf['title']:
            if ' ' in title:
                title  = title.replace(' ','')
            if '&' in title:
                title  = title.replace('&','and')
            if '-' in title:
                title  = title.replace('-','')    

            for cnt in range(sheet_kospi.nrows):
            
                if sheet_kospi.row_values(cnt)[1] == title:
                    
                    code = '{0:06d}'.format(int(sheet_kospi.row_values(cnt)[0]))
                    name = sheet_kospi.row_values(cnt)[1]
                    print code,name
                    markettype = 1
                    break

            for cnt in range(sheet_kosdaq.nrows):
                
                if sheet_kosdaq.row_values(cnt)[1] == title:
                    
                    code = '{0:06d}'.format(int(sheet_kosdaq.row_values(cnt)[0]))
                    name = sheet_kosdaq.row_values(cnt)[1]
                    print code,name
                    markettype = 2
                    break        

            
            startdatemode = 2
            dbtradinghist = 'none'
            histmode = 'none'
            plotly = 'plotly'
            stdmode = 'stddb'
            tangentmode = 'tangentdb'        
            daych  =0
            runcount = 0
            srcsite = 1#google
            # srcsite = 2#yahoo
            writedblog = 'nodisplay'
            updbpattern = 'none'
            appenddb = 'none'

            try:
                codevar,namevar\
                ,accumGain\
                ,tradingGain\
                ,tradingNum\
                ,tradingStance\
                ,lastbench\
                ,totalmoney\
                ,lastbuydate\
                ,macdsig,slowsig = v20.RunSimul(str(code),markettype,name,'realtime','dbpattern',histmode,runcount,srcsite,writedblog,updbpattern\
                            ,appenddb,startdatemode,\
                             dbtradinghist,plotly,stdmode,'none',daych,tangentmode)
                algo_title.append(title)
                algo1_totalaccumgain.append(accumGain)
                algo1_totaltradinggain.append(tradingGain)
                algo1_totaltradingnum.append(tradingNum)        
                algo_stance.append(tradingStance)
                algo_benchmark.append(lastbench)
                totalsum.append(totalmoney)
                algo_lastDate.append(lastbuydate)
                algo_macdsig.append(macdsig)
                algo_slowsig.append(slowsig)
                clear_output()
                if tradingNum != 0:
                    total_cnt += 1
                    total_sum = total_sum + totalmoney
            except:
                errortitles.append(title)
                stcore.PrintException()
                pass   

        tradingresultdf = pd.DataFrame({'title':algo_title,'algo1_accumgain':algo1_totalaccumgain,'algo1_tradinggain':algo1_totaltradinggain\
        ,'algo1_tradingnum':algo1_totaltradingnum\
        ,'totalsum':totalsum\
        ,'lastDate':algo_lastDate\
        ,'benchmark':algo_benchmark\
        ,'MACDSig':algo_macdsig\
        ,'SLOWSig':algo_slowsig\
        ,'stance':algo_stance}\
        )    
        # display(HTML('<font size=1>'+tradingresultdf._repr_html_()+'</font>'))
        display(HTML(tradingresultdf.to_html()))        

   
        initial_capital = 10000000
        tnum = 99
        # print totalsum
        # print 'totalsum',sum(totalsum),(float(sum(totalsum))- float(initial_capital*tnum))/float(initial_capital*tnum)
        print 'totalsum',sum(totalsum),'total_cnt',total_cnt
        print 'totalsum',total_sum,(float(total_sum)- float(initial_capital*total_cnt))/float(initial_capital*total_cnt)
        print '-----error title-------'
        for title in errortitles:
            print title

        return tradingresultdf    
    except Exception,e:
        stcore.PrintException()
        # con.close()

def screenAutoTest_Kalman(dbdf):
    try:
        global office
        # dbname = 'screen.sqlite'
        # tablename = 'screen'

        # con = sqlite3.connect("../../data/analysis/"+dbname)

        # dbdf = pd_sql.read_frame("SELECT * from "+tablename, con)

        # dbdf = dbdf[dbdf['allslope'] > 0.0]
        
        # display(HTML(dbdf.to_html()))    

        # print 'dbdf',dbdf
        if office == False:
            book_kosdaq = xlrd.open_workbook("C:/Users/Administrator/Documents/IPython Notebooks/stock_simul/Kosdaq_symbols.xls")
            book_kospi = xlrd.open_workbook('C:/Users/Administrator/Documents/IPython Notebooks/stock_simul/Kospi_Symbols.xls')
        else:    
            book_kosdaq = xlrd.open_workbook("C:/Users/AUTRON/Documents/IPython Notebooks/simul/Kosdaq_symbols.xls")
            book_kospi = xlrd.open_workbook('C:/Users/AUTRON/Documents/IPython Notebooks/simul/Kospi_Symbols.xls')

        sheet_kosdaq = book_kosdaq.sheet_by_name('kosdaq')
        sheet_kospi = book_kospi.sheet_by_name('kospi')


        algo1_totalaccumgain = []
        algo1_totaltradinggain = []
        algo1_totaltradingnum = []        
        
        algo_title = []
        algo_stance = []    
        errortitles = []
        algo_benchmark = []
        totalsum = []
        algo_lastDate= []
        algo_price = []
        algo_holdingnum = []
        algo_rank = []
        algo_codes = []
        total_sum = 0
        total_cnt = 0

        # dbdf = dbdf[:70]
        for title in dbdf['title']:
            if ' ' in title:
                title  = title.replace(' ','')
            if '&' in title:
                title  = title.replace('&','and')
            if '-' in title:
                title  = title.replace('-','')    

            for cnt in range(sheet_kospi.nrows):
            
                if sheet_kospi.row_values(cnt)[1] == title:
                    
                    code = '{0:06d}'.format(int(sheet_kospi.row_values(cnt)[0]))
                    name = sheet_kospi.row_values(cnt)[1]
                    print code,name
                    markettype = 1
                    rank = sheet_kospi.row_values(cnt)[2]
                    rank = 'kospi '+rank
                    algo_rank.append(rank)
                    break

            for cnt in range(sheet_kosdaq.nrows):
                
                if sheet_kosdaq.row_values(cnt)[1] == title:
                    
                    code = '{0:06d}'.format(int(sheet_kosdaq.row_values(cnt)[0]))
                    name = sheet_kosdaq.row_values(cnt)[1]
                    print code,name
                    markettype = 2
                    rank = sheet_kosdaq.row_values(cnt)[2]
                    rank = 'kosdaq '+rank
                    algo_rank.append(rank)
                    break        

            
            startdatemode = 2
            dbtradinghist = 'none'
            histmode = 'none'
            plotly = 'plotly'
            stdmode = 'stddb'
            tangentmode = 'tangentdb'        
            daych  =0
            runcount = 0
            srcsite = 1#google
            # srcsite = 2#yahoo
            writedblog = 'nodisplay'
            updbpattern = 'none'
            appenddb = 'none'

            try:
                codevar,namevar\
                ,accumGain\
                ,tradingGain\
                ,tradingNum\
                ,tradingStance\
                ,lastbench\
                ,totalmoney\
                ,lastbuydate\
                ,price,holding_num\
                 = v20.RunSimul_Kalman(str(code),markettype,name,'realtime','dbpattern',histmode,runcount,srcsite,writedblog,updbpattern\
                            ,appenddb,startdatemode,\
                             dbtradinghist,plotly,stdmode,'none',daych,tangentmode)
                algo_title.append(title)
                algo1_totalaccumgain.append(accumGain)
                algo1_totaltradinggain.append(tradingGain)
                algo1_totaltradingnum.append(tradingNum)        
                algo_stance.append(tradingStance)
                algo_benchmark.append(lastbench)
                totalsum.append(totalmoney)
                algo_lastDate.append(lastbuydate)
                algo_price.append(price)
                algo_holdingnum.append(holding_num)
                algo_codes.append(str(code))
                clear_output()
                if tradingNum != 0:
                    total_cnt += 1
                    total_sum = total_sum + totalmoney
            except:
                errortitles.append(title)
                stcore.PrintException()
                pass   

        tradingresultdf = pd.DataFrame({'title':algo_title,'algo1_accumgain':algo1_totalaccumgain,'algo1_tradinggain':algo1_totaltradinggain\
        ,'algo1_tradingnum':algo1_totaltradingnum\
        ,'totalsum':totalsum\
        ,'lastDate':algo_lastDate\
        ,'benchmark':algo_benchmark\
        # ,'MACDSig':algo_macdsig\
        # ,'SLOWSig':algo_slowsig\
        ,'stance':algo_stance\
        ,'rank':algo_rank\
        ,'code':algo_codes\
        ,'price':algo_price\
        ,'holding_num':algo_holdingnum\
        }\
        )    
        # display(HTML('<font size=1>'+tradingresultdf._repr_html_()+'</font>'))
        display(HTML(tradingresultdf.to_html()))        

   
        initial_capital = 10000000
        tnum = 99
        # print totalsum
        # print 'totalsum',sum(totalsum),(float(sum(totalsum))- float(initial_capital*tnum))/float(initial_capital*tnum)
        print 'totalsum',sum(totalsum),'total_cnt',total_cnt
        print 'totalsum',total_sum,(float(total_sum)- float(initial_capital*total_cnt))/float(initial_capital*total_cnt)
        print '-----error title-------'
        for title in errortitles:
            print title

        import os
        import gc
        import psutil
        gc.collect()
            
        return tradingresultdf   
    except Exception,e:
        stcore.PrintException()
        # con.close()

def screenAutoTest_Kalman_args(listarg):
    try:
        global office
        # dbname = 'screen.sqlite'
        # tablename = 'screen'

        # con = sqlite3.connect("../../data/analysis/"+dbname)

        # dbdf = pd_sql.read_frame("SELECT * from "+tablename, con)

        # dbdf = dbdf[dbdf['allslope'] > 0.0]
        
        # display(HTML(dbdf.to_html()))    

        # print 'dbdf',dbdf
        print 'screenAutoTest_Kalman_args inside'
        # print listarg
        print listarg[0]#,type(listarg[0])#,listarg[1]
        dbdf = listarg[0]
        fetch_date = listarg[1]
        endday_arg = listarg[2]
        run_arg = listarg[3]

        
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

        algo1_totalaccumgain = []
        algo1_totaltradinggain = []
        algo1_totaltradingnum = []        
        
        algo_title = []
        algo_stance = []    
        errortitles = []
        algo_benchmark = []
        totalsum = []
        algo_lastDate= []
        algo_price = []
        algo_holdingnum = []
        algo_rank = []
        algo_codes = []
        total_sum = 0
        total_cnt = 0

        dbg_list = []
        # dbdf = dbdf[:70]
        
        for title in dbdf['title']:
            if ' ' in title:
                title  = title.replace(' ','')
            if '&' in title:
                title  = title.replace('&','and')
            if '-' in title:
                title  = title.replace('-','')    

            rank_found = False   
            rank ='no rank' 
            for cnt in range(sheet_kospi.nrows):
            
                if sheet_kospi.row_values(cnt)[1] == title:
                    
                    code = '{0:06d}'.format(int(sheet_kospi.row_values(cnt)[0]))
                    name = sheet_kospi.row_values(cnt)[1]
                    print code,name
                    markettype = 1
                    rank = sheet_kospi.row_values(cnt)[2]
                    rank = 'kospi '+rank
                    # algo_rank.append(rank)
                    rank_found = True
                    break

            for cnt in range(sheet_kosdaq.nrows):
                
                if sheet_kosdaq.row_values(cnt)[1] == title:
                    
                    code = '{0:06d}'.format(int(sheet_kosdaq.row_values(cnt)[0]))
                    name = sheet_kosdaq.row_values(cnt)[1]
                    print code,name
                    markettype = 2
                    rank = sheet_kosdaq.row_values(cnt)[2]
                    rank = 'kosdaq '+rank
                    # algo_rank.append(rank)
                    rank_found = True
                    break        

            
            for cnt in range(sheet_sp.nrows):
            
                if sheet_sp.row_values(cnt)[0] == title:
                    
                    code = str(sheet_sp.row_values(cnt)[1])
                    name = str(sheet_sp.row_values(cnt)[0])

                    code = code.split('/')[1]
                    print code,name
                    markettype = 1
                    rank = sheet_sp.row_values(cnt)[2]
                    rank = 'no rank'
                    # algo_rank.append(rank)
                    rank_found = True
                    break   

            for cnt in range(sheet_nas.nrows):
            
                if sheet_nas.row_values(cnt)[0] == title:
                    
                    code = str(sheet_nas.row_values(cnt)[1])
                    name = str(sheet_nas.row_values(cnt)[0])

                    code = code.split('/')[1]
                    print code,name
                    markettype = 1
                    rank = sheet_nas.row_values(cnt)[2]
                    rank = 'no rank'
                    # algo_rank.append(rank)
                    rank_found = True
                    break                
            # if rank_found == False:
            #     algo_rank.append('no rank')
            # if not 'kospi' in rank or not 'kosdaq' in rank:
            #     rank ='no rank'


            if run_arg == 'run':
                startdatemode = 3
                tradeday_arg = listarg[4]
                trade_startday = tradeday_arg
                histmode = 'histdb'
            elif run_arg == 'total':
                startdatemode = 2
                trade_startday = ''
                for instartday in dbdf['lastDate']:
                    if instartday != '':
                        trade_startday = instartday.split(' ')[0]
                histmode = 'histdb'
            elif run_arg =='run_today':
                startdatemode = 3
                tradeday_arg = listarg[4]
                trade_startday = tradeday_arg
                histmode = 'none'    


            algomode = listarg[5]                        

            if len(listarg) >= 7:
                seltype =  listarg[6]
            else:
                seltype = 'normal'

            dbtradinghist = 'none'
            # histmode = 'none'
            # histmode = 'histdb'
            plotly = 'plotly'
            stdmode = 'stddb'
            tangentmode = 'tangentdb'        
            daych  =0
            runcount = 0
            srcsite = 1#google
            # srcsite = 2#yahoo
            writedblog = 'nodisplay'
            updbpattern = 'none'
            appenddb = 'none'

            try:
                print 'screenAutoTest_Kalman_args ',seltype
                codevar,namevar\
                ,accumGain\
                ,tradingGain\
                ,tradingNum\
                ,tradingStance\
                ,lastbench\
                ,totalmoney\
                ,lastbuydate\
                ,price,holding_num\
                 = v20.RunSimul_Kalman(str(code),markettype,name,'realtime','dbpattern',histmode,runcount,srcsite,writedblog,updbpattern\
                            ,appenddb,startdatemode,\
                             dbtradinghist,plotly,stdmode,'none',daych,tangentmode,fetch_date,endday_arg,trade_startday,algomode,seltype)
                algo_title.append(title)
                algo1_totalaccumgain.append(accumGain)
                algo1_totaltradinggain.append(tradingGain)
                algo1_totaltradingnum.append(tradingNum)        
                algo_stance.append(tradingStance)
                algo_benchmark.append(lastbench)
                totalsum.append(totalmoney)
                algo_lastDate.append(lastbuydate)
                algo_price.append(price)
                algo_holdingnum.append(holding_num)
                algo_codes.append(str(code))
                algo_rank.append(rank)
                clear_output()

                # if tradingNum != 0:
                #     total_cnt += 1
                #     total_sum = total_sum + totalmoney
                dbgbuf = '%s %s %s' %(title, rank, code)
                dbg_list.append(dbgbuf)

            except:
                errortitles.append(title)
                stcore.PrintException()
                continue   
        try:        
            print 'tradingresultdf len',len(algo_title),len(algo1_totalaccumgain),len(algo1_totaltradinggain)\
            ,len(algo1_totaltradingnum),len(totalsum),len(algo_lastDate)\
            ,len(algo_benchmark),len(algo_stance),len(algo_rank),len(algo_codes)\
            ,len(algo_price),len(algo_holdingnum)
            

            tradingresultdf = pd.DataFrame({'title':algo_title,'algo1_accumgain':algo1_totalaccumgain\
            ,'algo1_tradinggain':algo1_totaltradinggain\
            ,'algo1_tradingnum':algo1_totaltradingnum\
            ,'totalsum':totalsum\
            ,'lastDate':algo_lastDate\
            ,'benchmark':algo_benchmark\
            # ,'MACDSig':algo_macdsig\
            # ,'SLOWSig':algo_slowsig\
            ,'stance':algo_stance\
            ,'rank':algo_rank\
            ,'code':algo_codes\
            ,'price':algo_price\
            ,'holding_num':algo_holdingnum\
            }\
            )    
            # display(HTML('<font size=1>'+tradingresultdf._repr_html_()+'</font>'))
            # display(HTML(traderesult.to_html())) 
            display(HTML(tradingresultdf.to_html()))        

        
            # initial_capital = 10000000
            # tnum = 99
            # print totalsum
            # print 'totalsum',sum(totalsum),(float(sum(totalsum))- float(initial_capital*tnum))/float(initial_capital*tnum)
            # print 'totalsum',sum(totalsum),'total_cnt',total_cnt
            # print 'totalsum',total_sum,(float(total_sum)- float(initial_capital*total_cnt))/float(initial_capital*total_cnt)
            # print '-----error title-------'
            # for title in errortitles:
            #     print title
        except Exception,e:
            print e
            stcore.PrintException()
            pass

         
        try:
            if len(errortitles) > 0:
                if office == False:
                    dbname = 'C:/Users/Administrator/Documents/IPython Notebooks/stock_simul/data/error/runerrortitle'+'_'+title+'.sqlite'
                else:
                    dbname = 'C:/Users/AUTRON/Documents/IPython Notebooks/simul/data/error/runerrortitle'+'_'+title+'.sqlite'

                con = sqlite3.connect(dbname)
                tablename = 'runerrortitle'
                
                con.execute("DROP TABLE IF EXISTS "+tablename)
                errordf = pd.DataFrame({'titles':errortitles})

                pd_sql.write_frame(errordf, tablename, con)


                con.close()

            
        except Exception,e:
            con.close()
            stcore.PrintException()
            pass  
                 

        try:
            if len(tradingresultdf) > 0:
                if office == False:
                    dbname = 'C:/Users/Administrator/Documents/IPython Notebooks/stock_simul/data/error/tradingresultdf'+'_'+title+'.sqlite'
                else:
                    dbname = 'C:/Users/AUTRON/Documents/IPython Notebooks/simul/data/error/tradingresultdf'+'_'+title+'.sqlite'

                con = sqlite3.connect(dbname)
                tablename = 'tradingresultdf'
                
                con.execute("DROP TABLE IF EXISTS "+tablename)

                pd_sql.write_frame(tradingresultdf, tablename, con)

                con.close()

            
        except Exception,e:
            con.close()
            stcore.PrintException()
            pass  
        '''                 
        try:
            if len(dbg_list) > 0:
                if office == False:
                    dbname = 'C:/Users/Administrator/Documents/IPython Notebooks/stock_simul/data/error/dbg_list'+'_'+title+'.sqlite'
                else:
                    dbname = 'C:/Users/AUTRON/Documents/IPython Notebooks/simul/data/error/dbg_list'+'_'+title+'.sqlite'

                con = sqlite3.connect(dbname)
                tablename = 'dbgdf'
                
                con.execute("DROP TABLE IF EXISTS "+tablename)
                dbgdf = pd.DataFrame({'log':dbg_list})
                pd_sql.write_frame(dbgdf, tablename, con)

                con.close()
                
            
        except Exception,e:
            con.close()
            stcore.PrintException()
            pass  
        '''
                                 
        import os
        import gc
        import psutil
        gc.collect()
        

        return tradingresultdf 



    except Exception,e:
        stcore.PrintException()
        pass
        # con.close()

        


def screenAutoTest_Kalman_gain(listarg):
    try:
        global office
        # dbname = 'screen.sqlite'
        # tablename = 'screen'

        # con = sqlite3.connect("../../data/analysis/"+dbname)

        # dbdf = pd_sql.read_frame("SELECT * from "+tablename, con)

        # dbdf = dbdf[dbdf['allslope'] > 0.0]
        
        # display(HTML(dbdf.to_html()))    

        # print 'dbdf',dbdf
        print 'screenAutoTest_Kalman_gain inside'
        # print listarg
        print listarg[0]#,type(listarg[0])#,listarg[1]
        dbdf = listarg[0]
        fetch_date = listarg[1]
        endday_arg = listarg[2]
        run_arg = listarg[3]

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



        algo1_totalaccumgain = []
        algo1_totaltradinggain = []
        algo1_totaltradingnum = []        
        
        algo_title = []
        algo_stance = []    
        errortitles = []
        algo_benchmark = []
        totalsum = []
        algo_lastDate= []
        algo_price = []
        algo_holdingnum = []
        algo_rank = []
        algo_codes = []
        total_sum = 0
        total_cnt = 0


        # dbdf = dbdf[:70]
        for title in dbdf['title']:
            if ' ' in title:
                title  = title.replace(' ','')
            if '&' in title:
                title  = title.replace('&','and')
            if '-' in title:
                title  = title.replace('-','')    

            for cnt in range(sheet_kospi.nrows):
            
                if sheet_kospi.row_values(cnt)[1] == title:
                    
                    code = '{0:06d}'.format(int(sheet_kospi.row_values(cnt)[0]))
                    name = sheet_kospi.row_values(cnt)[1]
                    print code,name
                    markettype = 1
                    rank = sheet_kospi.row_values(cnt)[2]
                    rank = 'kospi '+rank
                    algo_rank.append(rank)
                    break

            for cnt in range(sheet_kosdaq.nrows):
                
                if sheet_kosdaq.row_values(cnt)[1] == title:
                    
                    code = '{0:06d}'.format(int(sheet_kosdaq.row_values(cnt)[0]))
                    name = sheet_kosdaq.row_values(cnt)[1]
                    print code,name
                    markettype = 2
                    rank = sheet_kosdaq.row_values(cnt)[2]
                    rank = 'kosdaq '+rank
                    algo_rank.append(rank)
                    break        

            for cnt in range(sheet_sp.nrows):
            
                if sheet_sp.row_values(cnt)[0] == title:
                    
                    code = str(sheet_sp.row_values(cnt)[1])
                    name = str(sheet_sp.row_values(cnt)[0])

                    code = code.split('/')[1]
                    print code,name
                    markettype = 1
                    rank = sheet_sp.row_values(cnt)[2]
                    rank = 'no rank'
                    algo_rank.append(rank)
                    
                    break        

            for cnt in range(sheet_nas.nrows):
            
                if sheet_nas.row_values(cnt)[0] == title:
                    
                    code = str(sheet_nas.row_values(cnt)[1])
                    name = str(sheet_nas.row_values(cnt)[0])

                    code = code.split('/')[1]
                    print code,name
                    markettype = 1
                    rank = sheet_nas.row_values(cnt)[2]
                    rank = 'no rank'
                    algo_rank.append(rank)
                    
                    break        

        
            if run_arg == 'run':
                startdatemode = 3
                tradeday_arg = listarg[4]
                trade_startday = tradeday_arg
            elif run_arg == 'total':
                startdatemode = 2
                trade_startday = ''
                for instartday in dbdf['lastDate']:
                    if instartday != '':
                        trade_startday = instartday.split(' ')[0]

            algomode = listarg[5]  
            seltype = listarg[6]
            
            dbtradinghist = 'none'
            # histmode = 'none'
            histmode = 'histdb'
            plotly = 'plotly'
            stdmode = 'stddb'
            tangentmode = 'tangentdb'        
            daych  =0
            runcount = 0
            srcsite = 1#google
            # srcsite = 2#yahoo
            writedblog = 'nodisplay'
            updbpattern = 'none'
            appenddb = 'none'

            try:
                codevar,namevar\
                ,accumGain\
                ,tradingGain\
                ,tradingNum\
                ,tradingStance\
                ,lastbench\
                ,totalmoney\
                ,lastbuydate\
                ,price,holding_num\
                ,traderesult = v20.RunSimul_Kalman_gain(str(code),markettype,name,'realtime','dbpattern',histmode,runcount,srcsite,writedblog,updbpattern\
                            ,appenddb,startdatemode,\
                             dbtradinghist,plotly,stdmode,'none',daych,tangentmode,fetch_date,endday_arg,trade_startday,algomode,seltype)
                algo_title.append(title)
                algo1_totalaccumgain.append(accumGain)
                algo1_totaltradinggain.append(tradingGain)
                algo1_totaltradingnum.append(tradingNum)        
                algo_stance.append(tradingStance)
                algo_benchmark.append(lastbench)
                totalsum.append(totalmoney)
                algo_lastDate.append(lastbuydate)
                algo_price.append(price)
                algo_holdingnum.append(holding_num)
                algo_codes.append(str(code))
                clear_output()
                if tradingNum != 0:
                    total_cnt += 1
                    total_sum = total_sum + totalmoney
            except:
                errortitles.append(title)
                stcore.PrintException()
                pass   

        tradingresultdf = pd.DataFrame({'title':algo_title,'algo1_accumgain':algo1_totalaccumgain,'algo1_tradinggain':algo1_totaltradinggain\
        ,'algo1_tradingnum':algo1_totaltradingnum\
        ,'totalsum':totalsum\
        ,'lastDate':algo_lastDate\
        ,'benchmark':algo_benchmark\
        # ,'MACDSig':algo_macdsig\
        # ,'SLOWSig':algo_slowsig\
        ,'stance':algo_stance\
        ,'rank':algo_rank\
        ,'code':algo_codes\
        ,'price':algo_price\
        ,'holding_num':algo_holdingnum\
        }\
        )    
        # display(HTML('<font size=1>'+tradingresultdf._repr_html_()+'</font>'))
        display(HTML(traderesult.to_html())) 
        display(HTML(tradingresultdf.to_html()))        

   
        initial_capital = 10000000
        tnum = 99
        # print totalsum
        # print 'totalsum',sum(totalsum),(float(sum(totalsum))- float(initial_capital*tnum))/float(initial_capital*tnum)
        
        print 'totalsum',sum(totalsum),'total_cnt',total_cnt
        print 'totalsum',total_sum,(float(total_sum)- float(initial_capital*total_cnt))/float(initial_capital*total_cnt)
        print '-----error title-------'
        for title in errortitles:
            print title

        import os
        import gc
        import psutil
        gc.collect()
            
        return tradingresultdf,traderesult 



    except Exception,e:
        stcore.PrintException()
        # con.close()



def screenGetData_fromIndex(startNum,endNum,markettype):
    global office


    if markettype == 1:
        if office == False:
            book_kospi = xlrd.open_workbook("C:/Users/Administrator/Documents/IPython Notebooks/stock_simul/Kospi_symbols.xls")
        else:
            book_kospi = xlrd.open_workbook("C:/Users/AUTRON/Documents/IPython Notebooks/simul/Kospi_symbols.xls")
        sheet_kospi = book_kospi.sheet_by_name('kospi')
    elif markettype == 2:  
        if office == False:  
            book_kosdaq = xlrd.open_workbook("C:/Users/Administrator/Documents/IPython Notebooks/stock_simul/Kosdaq_symbols.xls")
        else:
            book_kosdaq = xlrd.open_workbook("C:/Users/AUTRON/Documents/IPython Notebooks/simul/Kosdaq_symbols.xls")
        sheet_kosdaq = book_kosdaq.sheet_by_name('kosdaq')


    
    srcsite = 1#google
    # srcsite = 2#yahoo
    runcount = 1
    dbtradinghist = 'none'
    # histmode = 'histdb'
    histmode = 'none'
    plotly = 'plotly'
    stdmode = 'stddb'
    tangentmode = 'tangentdb'
    updbpattern = 'none'
    appenddb = 'none'
    writedblog = 'writedblog'
    startdatemode = 2
    daych = 0


    start = time.clock()
    name = u''
    code = 0        

    algo1_totalaccumgain = []
    algo1_totaltradinggain = []
    algo1_totaltradingnum = []        
    
    algo_title = []
    algo_stance = []    
    errortitles = []
    algo_benchmark = []
    totalsum = []
    algo_lastDate= []
    algo_macdsig = []
    algo_slowsig = []
    algo_paramallslope = []
    total_sum = 0
    total_cnt = 0

    

    for cnt in range(startNum,endNum):
        if markettype == 1:
            code = '{0:06d}'.format(int(sheet_kospi.row_values(cnt)[0]))
            title = sheet_kospi.row_values(cnt)[1]
            print title
        elif markettype == 2:
            code = '{0:06d}'.format(int(sheet_kosdaq.row_values(cnt)[0]))
            title = sheet_kosdaq.row_values(cnt)[1]
            print title
        

    

        try:
            lastbench,lastslope = RunSimul(str(code),markettype,name,'realtime','dbpattern',histmode,runcount,srcsite,writedblog,updbpattern\
                        ,appenddb,startdatemode,\
                         dbtradinghist,plotly,stdmode,'none',daych,tangentmode)
            algo_title.append(title)
            algo_benchmark.append(lastbench)
            algo_paramallslope.append(lastslope)
            clear_output()
            
        except Exception,e:
            # print 'error',title
            errortitles.append(title)
            stcore.PrintException()
            pass   

    tradingresultdf = pd.DataFrame({'title':algo_title\
        ,'benchmark':algo_benchmark\
        ,'allslope':algo_paramallslope\
        }\
    )    
    
    display(HTML(tradingresultdf.to_html()))        


    print '-----error title-------'
    for title in errortitles:
        print title

    return tradingresultdf

    

def screenSaveResult(resultdf):
    try:
        dirpath = '../../data/result/'+str(datetime.today()).split(' ')[0].replace('-','')

        stcore.assure_path_exists(dirpath)
        dirpath += '/'
        dbname = dirpath+'result.sqlite'

        con = sqlite3.connect(dbname)

        tablename = 'result'

        con.execute("DROP TABLE IF EXISTS "+tablename)
        
        pd_sql.write_frame(resultdf, tablename, con)
        con.close()

    except Exception,e:
        stcore.PrintException() 
        con.close()   


def sleep_and_return(factor=10):
    import time
    import random
    r = factor * random.random()
    # time.sleep(r)
    
    return r    


def readAllLists(startDate,endDate):
    
    try:
        dirpath = '../../data/analysis/screen/'
        import glob
        screenlists = glob.glob("../../data/analysis/screen/*.sqlite")

        sdate = datetime.strptime(startDate, "%Y%m%d")
        edate = datetime.strptime(endDate, "%Y%m%d")
        # print sdate,edate

        allscreenlistdf = pd.DataFrame()
        for listitem in screenlists:
            # print listitem.split('\\')[-1]
            dbname = listitem
            dbfilename = listitem.split('\\')[-1]
            if 'default' in dbfilename:
                dbfilename = dbfilename.split('_')[0]
                # print 'defalut ',dbfilename
            datename = dbfilename.split('.')[0]
            datename = datetime.strptime(datename,"%Y%m%d")
            if sdate<= datename <= edate:
                print 'selected db:',datename,dbname

                con = sqlite3.connect(dbname)
                tablename = 'screen'
                
                screendf = pd_sql.read_frame("SELECT * from "+tablename, con)    
                con.close()
                allscreenlistdf = pd.concat([allscreenlistdf,screendf])

        allscreenlistdf =  allscreenlistdf.drop_duplicates('title')        
        allscreenlistdf = allscreenlistdf.reset_index()
        resuallscreenlistdflts = allscreenlistdf.drop('index',1)

        print 'total len:',len(allscreenlistdf)
        # display(HTML(allscreenlistdf.to_html()))

        

        return allscreenlistdf
            
    except Exception,e:
        stcore.PrintException() 
        con.close()       

def readResult():
    try:
        # resultdbpath = '20150819'
        resultdbpath = str(datetime.today()).split(' ')[0].replace('-','')
        title = '../../data/result/'+resultdbpath+'/result.sqlite'
        tablename = 'result'

        con = sqlite3.connect(title)
        screendbdf = pd_sql.read_frame("SELECT * from "+tablename, con)
        screendbdf = screendbdf[screendbdf['lastDate'] != 'NA']
        # display(HTML(screendbdf.to_html()))
        con.close()
        return screendbdf
    except Exception,e:
        con.close()
        print e

def extractRemainingScreenLists(allscreenlistdf,screensavedbdf):

    retdf = allscreenlistdf
    for title in screensavedbdf['title']:
        retdf = retdf[retdf.title != title]
    retdf = retdf.reset_index()
    retdf = retdf.drop('index',1)
    # retdf = retdf.drop('level_0',1)
    # display(HTML(retdf.to_html()))
    # retdf = retdf.drop_duplicates('title')
    # print len(retdf)
    return retdf


def screenSaveRemainingResult(resultdf):
    try:
        dirpath = '../../data/result/'+str(datetime.today()).split(' ')[0].replace('-','')

        stcore.assure_path_exists(dirpath)
        dirpath += '/'
        dbname = dirpath+'RemainingResult.sqlite'

        con = sqlite3.connect(dbname)

        tablename = 'result'

        con.execute("DROP TABLE IF EXISTS "+tablename)
        
        pd_sql.write_frame(resultdf, tablename, con)
        con.close()

    except Exception,e:
        stcore.PrintException() 
        con.close()   



