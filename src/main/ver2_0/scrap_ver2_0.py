# -*- coding: UTF-8 -*-
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
# from pytagcloud import create_tag_image, make_tags
# from pytagcloud.lang.counter import get_tag_counts

import webbrowser
import csv
import os
from datetime import datetime, timedelta

import sqlite3
import pandas.io.sql as pd_sql
import pandas as pd

from IPython.html.widgets import interact, interactive, fixed
from IPython.html import widgets
from IPython.display import clear_output, display, HTML
pd.set_option('display.width', 500)

import urllib
import time

import stockcore as stcore
import tradingalgo as talgo
import data_mani as dmani


import xlrd
import xlwt

from stockcore import *
from tradingalgo import *
from data_mani import *
from ver2_0_i import *

global office

def getData(type):
    try:

        # Create a new instance of the Firefox driver
        # driver = webdriver.Firefox()

        # driver = webdriver.Chrome()

        if type == 1:
            # 투자전략 종목진단 paxnet_00
            siteorg = "http://bnb.moneta.co.kr/today_recom/jm/totalIntro.php?"
            enumarg = [1, 2, 3]
        elif type == 2:
            # 종목토론 인기글 paxnet_01
            siteorg = "http://board.moneta.co.kr/cgi-bin/paxBulletin/bulList.cgi?mode=bestlist&code=999998"
            enumarg = [1, 2, 3]
        elif type == 3:
            # 강추이종목 단기강추 paxnet_02
            siteorg = "http://bbs.moneta.co.kr/nbbs/bbs.normal1.lst.screen?p_message_id=&p_bbs_id=N00820"
            enumarg = [1, 2, 3]
        elif type == 4:
            # 스몰캡 타이밍 paxnet_03
            siteorg = 'http://news.moneta.co.kr/Service/stock/ShellList.asp?ModuleID=3484&LinkID=619&'
            enumarg = [1, 2, 3]
        elif type == 5:
            siteorg = 'http://recommend.finance.naver.com/Home/YieldByCompany/naver#Submit?stdt=' + \
                str(datetime.today()).split(' ')[
                    0] + '&cmpcd=&brkcd=0&pfcd=0&orderType=D&orderCol=8'  # naver 추천종목 수익률 naver_00
            enumarg = [1, 2, 3]

        textouts = []
        links = []
        elemtimes = []

        driver = webdriver.Firefox()
        # driver = webdriver.Chrome()
        # driver.implicitly_wait(5)
        # driver.switch_to_default_content()

        for i, page in enumerate(enumarg):

            start = str(page)
        #     site = "http://www.google.com/search?filter=0&start="+start+"&client=firefox-a"
            # sitequerystr = site+"&tbs=cdr:1,"+"cd_min:"+startdate+",cd_max:"+enddate+"&sitesearch="+sitesearch+"&q="+query+"&safe=off&pws=0&num=50&access=a"
        #     sitequerystr = site+"&tbs=cdr:1,"+"cd_min:"+startdate+",cd_max:"+enddate+"&q="+query+"&safe=off&pws=0&num=50&access=a"
        #     driver.get(sitequerystr)
            if type == 1:
                site = siteorg + 'page=' + start
            elif type == 2:
                site = siteorg + '&page=' + start
            elif type == 3:
                site = siteorg + '&p_page_num=' + start + \
                    '&p_current_sequence=zzzzz%7E&p_start_sequence=zzzzz%7E&p_start_page=1&direction=1&p_favor_avoid=&service=stock&menu=menu_debate&depth=1&sub=9&top=1&p_action=&p_tp_board=&total=&p_hot_fg=&cntnum=&p_total=90333&p_beg_item=%B4%DC%B1%E2%B0%AD%C3%DF&p_search_field=&p_search_word=&p_blind_fg=S&p_blind_url=lst&searchMode='
            elif type == 4:
                site = siteorg + 'NavDate=' + \
                    str(datetime.today()).split(' ')[0].replace(
                        '-', '') + '&NavPage=' + start
            elif type == 5:
                site = siteorg + '&curpage=' + start

            try:
                print site
                driver.get(site)

            #     print len(elems)
            #     elems = driver.find_elements_by_xpath('//*/li/div/h3/a')
                if type == 1:
                    elems = driver.find_elements_by_xpath(
                        '//*[@id="content"]/table/tbody/tr/td[1]/table/tbody/tr[2]/td/table/tbody/tr/td[1]')
                elif type == 2:
                    elems = driver.find_elements_by_xpath(
                        '//*[@id="container"]/div/table/tbody/tr/td[2]/a')
                elif type == 3:
                    elems = driver.find_elements_by_xpath(
                        '//*[@id="bbs_frm"]/table/tbody[2]/tr/td[2]/a/span')
                elif type == 4:
                    elems = driver.find_elements_by_xpath(
                        '//*[@id="contant2"]/ul/li/a/strong')
                elif type == 5:
                    time.sleep(2)
                    elems = driver.find_elements_by_xpath(
                        '//*[@id="yieldTable"]/tbody/tr/td[2]/div/a')

                for i, elem in enumerate(elems):
                    #         print elem.get_attribute("href")
                    #         textout.append(elem.text)
                    #         textout = (elem.text).encode('euc-kr')
                    #         writer.writerow([textout])
                    if type == 1:
                        strout = elem.text.replace('[', '')
                        strout = strout.replace(']', '')
                    elif type == 2:
                        strout = elem.text.split('.')[0]
                        strout = strout.replace('[', '')
                        strout = strout.replace(']', '')
                        if i == 0:
                            continue
                    elif type == 3:
                        strout = elem.text.replace('[', '')
                        strout = strout.replace(']', '')
                        strout = strout.split(' ')[0]
                    elif type == 4:
                        strout = elem.text.split(',')[0]
                    elif type == 5:
                        strout = elem.text

                    textouts.append(strout)

                    # linkout = '<a href="{0}">link</a>'.format(elem.get_attribute("href"))
            #         linkout = elem.get_attribute("href")
                    # links.append(linkout)

                time.sleep(10)
            except Exception, e:
                stcore.PrintException()
                pass

        driver.close()

    except Exception, e:
        stcore.PrintException()
        driver.quit()

    return textouts


def saveData(datadf, dbfilename, tableName):
    try:
        dirpath = '../../data/scrap/' + \
            str(datetime.today()).split(' ')[0].replace('-', '')

        assure_path_exists(dirpath)
        dirpath += '/'
        dbname = dirpath + dbfilename + '_' + \
            str(datetime.today()).split(' ')[0].replace('-', '') + '.sqlite'
        # if os.path.isfile(dbname):
        #     os.remove(dbname)
        con = sqlite3.connect(dbname)

        tablename = tableName  # 'analysis_table_0'
        # print 'tablename',tablename,type(tablename)
        con.execute("DROP TABLE IF EXISTS " + tablename)

        pd_sql.write_frame(datadf, tablename, con)
        con.close()

    except Exception, e:
        con.close()
        stcore.PrintException()
        pass


def readLists():
    try:
        import glob
        lists = glob.glob("../../data/scrap/*.sqlite")
        for filename in lists:
            print filename.split('\\')[-1]

        # con = sqlite3.connect("../../data/scrap/"+dbname)
        # dbdf = pd_sql.read_frame("SELECT * from "+tablename, con)
        # con.close()

    except Exception, e:
        # con.close()
        stcore.PrintException()
        pass


def saveSumData_selectLists():
    try:

        lists = [
            'naver_00_20150711.sqlite',
            'paxnet_00_20150712.sqlite',
            'paxnet_01_20150712.sqlite',
            'paxnet_02_20150713.sqlite',
            'paxnet_03_20150713.sqlite'
        ]

        dfsum = pd.DataFrame()
        for filename in lists:
            dbname = filename
            tablename = filename.split('_20')[0]
            print tablename
            con = sqlite3.connect("../../data/scrap/" + dbname)

            dbdf = pd_sql.read_frame("SELECT * from " + tablename, con)
            # print dbdf.tail()
            dfsum = pd.concat([dfsum, dbdf], ignore_index=True)
            con.close()
        # display(HTML(dfsum.to_html()))

        stocknames = []
        for title in dfsum['title']:
            if '\n' in title:
                title = title.replace('\n', '')
            if '&' in title:
                title = title.replace('&', 'and')
            if ' ' in title:
                title = title.replace(' ', '')
            if '-' in title:
                title = title.replace('-', '')
            stocknames.append(title)

        dftotal = pd.DataFrame({'title': stocknames})

        dftotal = dftotal.drop_duplicates('title')
        dftotal = dftotal.reset_index()
        dftotal = dftotal.drop('index', 1)
        # dftotal['duplicated'] = dftotal.duplicated('title')
        # dftotal = dftotal[['title']][dftotal['duplicated'] == True]
        display(HTML(dftotal.to_html()))

        return dftotal
    except Exception, e:
        con.close()
        stcore.PrintException()
        pass


def assure_path_exists(path):
    # dir = os.path.dirname(path)
    if not os.path.exists(path):
        os.makedirs(path)


def saveSumData(folderpath):
    try:
        selectfolder = folderpath  # '20150803'
        import glob
        lists = glob.glob("../../data/scrap/" + selectfolder + "/*.sqlite")
        filelists = []
        for filename in lists:
            tempname = filename.split('\\')[-1]
            print tempname
            filelists.append(tempname)

        dfsum = pd.DataFrame()
        for filename in filelists:
            dbname = filename
            tablename = filename.split('_20')[0]
            print tablename
            con = sqlite3.connect(
                "../../data/scrap/" + selectfolder + "/" + dbname)

            dbdf = pd_sql.read_frame("SELECT * from " + tablename, con)
            # print dbdf.tail()
            dfsum = pd.concat([dfsum, dbdf], ignore_index=True)
            con.close()
        # display(HTML(dfsum.to_html()))

        stocknames = []
        for title in dfsum['title']:
            if '\n' in title:
                title = title.replace('\n', '')
            if '&' in title:
                title = title.replace('&', 'and')
            if ' ' in title:
                title = title.replace(' ', '')
            if '-' in title:
                title = title.replace('-', '')
            stocknames.append(title)

        dftotal = pd.DataFrame({'title': stocknames})

        dftotal = dftotal.drop_duplicates('title')
        dftotal = dftotal.reset_index()
        dftotal = dftotal.drop('index', 1)
        # dftotal['duplicated'] = dftotal.duplicated('title')
        # dftotal = dftotal[['title']][dftotal['duplicated'] == True]
        display(HTML(dftotal.to_html()))

        return dftotal
    except Exception, e:
        con.close()
        stcore.PrintException()
        pass

book_etf = xlrd.open_workbook("../../Kospi_symbols.xls")
sheet_etf = book_etf.sheet_by_name('etf')


def readETFLists():
    etfdf = pd.DataFrame()
    for cnt in range(sheet_etf.nrows):

        code = '{0:06d}'.format(int(sheet_etf.row_values(cnt)[0]))
        name = sheet_etf.row_values(cnt)[1]
        # print 'eft code:',code,name
        tmpdf = pd.DataFrame({'title': [name]})
        etfdf = pd.concat([tmpdf, etfdf])

    etfdf = etfdf.reset_index()
    etfdf = etfdf.drop('index', 1)
    return etfdf


def saveSumETFwithScrapData(folderpath):
    portrecommanddf = scrapRecommandAttributesInHTML()
    dftotal = saveSumData(folderpath)
    etfdf = readETFLists()
    

    dftotal = pd.concat([dftotal, etfdf])
    dftotal = pd.concat([portrecommanddf,dftotal])

    return dftotal


def scrapAnalysis():
    oldscreendf = saveSumData_selectLists()

    newscreendf = saveSumData()

    totaldf = pd.concat([oldscreendf, newscreendf])

    clear_output()

    """ select duplicated lists   
    totallen = len(totaldf)
    print 'total num ',totallen
    totaldf_drop = totaldf.drop_duplicates('title')
    totaldf_drop = totaldf_drop.reset_index()
    totaldf_drop = totaldf_drop.drop('index',1)
    # display(HTML(totaldf_drop.to_html()))
    print 'duplicated',totallen - len(totaldf_drop)
    totaldf['same'] = totaldf.duplicated('title')
    totaldf = totaldf[totaldf['same'] == True]
    totaldf_drop = totaldf.drop_duplicates('title')
    totaldf_drop = totaldf_drop.reset_index()
    totaldf_drop = totaldf_drop.drop('index',1)
    display(HTML(totaldf_drop.to_html()))
    """

    totallen = len(totaldf)
    print 'total num ', totallen
    totaldf['same'] = totaldf.duplicated('title')
    totaldf = totaldf[totaldf['same'] == False]
    totaldf = totaldf.reset_index()
    totaldf = totaldf.drop('index', 1)
    # display(HTML(totaldf.to_html()))

    totaldf_final = totaldf
    return totaldf_final


def stocktitleInScreendb(screendf):
    title = '../../data/port/port_daily.sqlite'
    tablename = 'portfolio_result'
    clear_output()
    if os.path.isfile(title):
        try:
            con = sqlite3.connect(title)
            portdf = pd_sql.read_frame("SELECT * from " + tablename, con)
            con.close()

            for porttitle in portdf['title']:
                found = False
                for dftitle in screendf['title']:
                    if dftitle == porttitle:
                        print 'title is included', porttitle
                        found = True
                        break
                if found == False:
                    print 'title is not included'
        except Exception, e:
            con.close()
            stcore.PrintException()
    # found = False
    # for dftitle in screendf['title']:
    #     if dftitle == title:
    #         print 'title is included',title
    #         found = True
    #         break
    # if found == False:
    #     print 'title is not included'


def scrapRecommandAttributesInHTML():
    print 'scrapRecommandAttributesInHTML'
    try:
        site = 'http://recommend.finance.naver.com/Home/CumulativeYieldByBrokerage/naver'
        driver = webdriver.Firefox()
        driver.get(site)

        elems = driver.find_elements_by_xpath('//*[@id="cumulative"]/table[2]/tbody/tr/td[4]/span[2]')
        brkcds = []
        pfcds = []
        for i, elem in enumerate(elems):

            strout1 = elem.get_attribute('data-brkcd')
            strout2 = elem.get_attribute('data-pfcd')
            brkcds.append(strout1)
            pfcds.append(strout2)
            print 'data-brkcd:', strout1,' pfcds:',strout2

            time.sleep(5)

        driver.close()


        '''
        //*[@id="cumulative"]/table[2]/tbody/tr[1]/td[4]/span[2]
        //*[@id="cumulative"]/table[2]/tbody/tr[2]/td[4]/span[2]
        http://recommend.finance.naver.com/Common/PortfolioLookup/naver#Submit?brkcd=7&pfcd=73&stddt=20150818
        http://recommend.finance.naver.com/Common/PortfolioLookup/naver#Submit?brkcd=7&pfcd=57&stddt=20150818
        '''

        '''
        //*[@id="recommPopTable"]/tbody/tr[1]/td[1]/a/div
        //*[@id="recommPopTable"]/tbody/tr/td[1]/a/div
        //*[@id="recommPopTable"]/tbody/tr[1]/td[1]/a/div
        //*[@id="recommPopTable"]/tbody/tr[2]/td[1]/a/div
        '''    
        stocktitles=  []
        runcnt = 0
        for brkcd,pfcd in zip(brkcds,pfcds):
            if runcnt > 3:
                break
            driver = webdriver.Firefox()
            todaydate = str(datetime.today()).split(' ')[0].replace('-', '')
            site = 'http://recommend.finance.naver.com/Common/PortfolioLookup/naver#Submit?brkcd='+brkcd+'&pfcd='+pfcd+'&stddt='+todaydate
            print site
            
            driver.get(site)
            elems = driver.find_elements_by_xpath('//*[@id="recommPopTable"]/tbody/tr/td[1]/a/div')

            for i, elem in enumerate(elems):
                title = elem.text
                
                if '\n' in title:
                    title = title.replace('\n', '')
                if '&' in title:
                    title = title.replace('&', 'and')
                if ' ' in title:
                    title = title.replace(' ', '')
                if '-' in title:
                    title = title.replace('-', '')
            

                stocktitles.append(title)
                print 'title:', title

            time.sleep(5)
            driver.close()    
            runcnt += 1

        stockdf = pd.DataFrame({'title':stocktitles})
        stockdf = stockdf.drop_duplicates('title')
        stockdf = stockdf.reset_index()
        stockdf = stockdf.drop('index', 1)
        stockdf = stockdf[stockdf['title'] != '']
        
        clear_output()
        # display(HTML(stockdf.to_html()))

    except Exception,e:
        stcore.PrintException()

    return stockdf


    
def getRankfromSite():
    print 'getRankfromSite'    
    # http://finance.naver.com/item/main.nhn?code=005930

    global office

    import random
    
    try:
        if office == False:
            book_kosdaq = xlrd.open_workbook("C:/Users/Administrator/Documents/IPython Notebooks/stock_simul/Kosdaq_symbols.xls")
            book_kospi = xlrd.open_workbook('C:/Users/Administrator/Documents/IPython Notebooks/stock_simul/Kospi_Symbols.xls')
        else:    
            book_kosdaq = xlrd.open_workbook("C:/Users/AUTRON/Documents/IPython Notebooks/simul/Kosdaq_symbols.xls")
            book_kospi = xlrd.open_workbook('C:/Users/AUTRON/Documents/IPython Notebooks/simul/Kospi_Symbols.xls')

        sheet_kosdaq = book_kosdaq.sheet_by_name('kosdaq')
        sheet_kospi = book_kospi.sheet_by_name('kospi')
        
        titles = []
        codelist = []
        ranklist = []

        for cnt in range(sheet_kospi.nrows):
            if 0 < cnt :
                try:
                    code = '{0:06d}'.format(int(sheet_kospi.row_values(cnt)[0]))
                    name = sheet_kospi.row_values(cnt)[1]
                    print code,name
                    site = 'http://finance.naver.com/item/main.nhn?code='+code
                    driver = webdriver.Firefox()
                    driver.get(site)                

                    elems = driver.find_elements_by_xpath('//*[@id="tab_con1"]/div[1]/table/tbody/tr[2]')
                    
                    for i, elem in enumerate(elems):
                        strout = elem.text
                        print strout
                        strout = strout.split(' ')[2]
                        titles.append(name)
                        codelist.append(code)
                        ranklist.append(strout)

                    driver.close()
                    time.sleep(random.randint(1, 5))
                except Exception,e:
                    stcore.PrintException()
                    driver.close()    
                    pass

        for cnt in range(sheet_kosdaq.nrows):
            if 0 < cnt :
                try:
                    code = '{0:06d}'.format(int(sheet_kosdaq.row_values(cnt)[0]))
                    name = sheet_kosdaq.row_values(cnt)[1]
                    print code,name
                    site = 'http://finance.naver.com/item/main.nhn?code='+code
                    driver = webdriver.Firefox()
                    driver.get(site)                

                    elems = driver.find_elements_by_xpath('//*[@id="tab_con1"]/div[1]/table/tbody/tr[2]')
                    
                    for i, elem in enumerate(elems):
                        strout = elem.text
                        print strout
                        strout = strout.split(' ')[2]
                        titles.append(name)
                        codelist.append(code)
                        ranklist.append(strout)

                    driver.close()                
                    time.sleep(random.randint(2, 5))
                except Exception,e:
                    stcore.PrintException()
                    driver.close()    
                    pass
                        
        resultdf = pd.DataFrame({'title':titles,'code':codelist,'rank':ranklist})        
        scrapSaveRank(resultdf)        
        
    except Exception,e:
        stcore.PrintException()

def scrapSaveRank(resultdf):
    try:
        dirpath = '../../data/rank/'+str(datetime.today()).split(' ')[0].replace('-','')

        stcore.assure_path_exists(dirpath)
        dirpath += '/'
        
        dbname = dirpath+'rank.sqlite'

        con = sqlite3.connect(dbname)

        tablename = 'rank'

        con.execute("DROP TABLE IF EXISTS "+tablename)
        
        pd_sql.write_frame(resultdf, tablename, con)
        con.close()

    except Exception,e:
        stcore.PrintException() 
        con.close()   

    
def mergeRankToExcel():
    print 'mergeRankToExcel'    
    from xlutils.copy import copy

    try:
        dbname = '../../data/rank/20160110/rank.sqlite'
        tablename = 'rank'
        con = sqlite3.connect(dbname)
        rankdf = pd_sql.read_frame("SELECT * from "+tablename, con)

        display(HTML(rankdf.tail().to_html()))
    except Exception,e:
        stcore.PrintException()
        con.close()

    global office

    try:
        if office == False:
            book_kosdaq = xlrd.open_workbook("C:/Users/Administrator/Documents/IPython Notebooks/stock_simul/Kosdaq_symbols.xls")
            book_kospi = xlrd.open_workbook('C:/Users/Administrator/Documents/IPython Notebooks/stock_simul/Kospi_Symbols.xls')
        else:    
            book_kosdaq = xlrd.open_workbook("C:/Users/AUTRON/Documents/IPython Notebooks/simul/Kosdaq_symbols.xls")
            book_kospi = xlrd.open_workbook('C:/Users/AUTRON/Documents/IPython Notebooks/simul/Kospi_Symbols.xls')

        sheet_kosdaq = book_kosdaq.sheet_by_name('kosdaq')
        sheet_kospi = book_kospi.sheet_by_name('kospi')
        
        titles = []
        codelist = []
        ranklist = []

        titlecnt = 0

        kospi_wb = copy(book_kospi)
        w_sheet_kospi = kospi_wb.get_sheet(0)

        for cnt in range(sheet_kospi.nrows):
            try:
                for ranktitle,rankval in zip(rankdf['title'],rankdf['rank']):        
                    if  ranktitle == sheet_kospi.row_values(cnt)[1]:
                    
                        # code = '{0:06d}'.format(int(sheet_kospi.row_values(cnt)[0]))
                        # name = sheet_kospi.row_values(cnt)[1]
                        # print code,name
                        w_sheet_kospi.write(cnt,2,rankval)
                        titlecnt = titlecnt +1
                        break

            except Exception,e:
                stcore.PrintException()
                pass
        print 'titlecnt',titlecnt      
        kospi_wb.save('C:/Users/Administrator/Documents/IPython Notebooks/stock_simul/Kospi_symbols_update.xls')   

        titlecnt = 0

        kosdaq_wb = copy(book_kosdaq)
        w_sheet_kosdaq = kosdaq_wb.get_sheet(0)

        for cnt in range(sheet_kosdaq.nrows):
            try:
                for ranktitle,rankval in zip(rankdf['title'],rankdf['rank']):        
                    if  ranktitle == sheet_kosdaq.row_values(cnt)[1]:
                    
                        # code = '{0:06d}'.format(int(sheet_kosdaq.row_values(cnt)[0]))
                        # name = sheet_kosdaq.row_values(cnt)[1]
                        # print code,name
                        w_sheet_kosdaq.write(cnt,2,rankval)
                        titlecnt = titlecnt +1
                        break

            except Exception,e:
                stcore.PrintException()
                pass
        print 'titlecnt',titlecnt      
        kosdaq_wb.save('C:/Users/Administrator/Documents/IPython Notebooks/stock_simul/Kosdaq_symbols_update.xls')   

    except Exception,e:
        stcore.PrintException()
    




















    