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


import sys
sys.path.append('../../src/main/ver2_0/')
sys.path.append('../../lib/')
from IPython import parallel
import screen_ver2_0 as sr

import os.path

global office
office = False



def calcTotalGain(startDate,endDate,fetch_date,current_date,tradestart_day,algo_mode):
    try:
        import datetime
        import glob
        
        print startDate,endDate 
        sDate = dt.datetime(int(startDate.split('-')[0]),int(startDate.split('-')[1]),int(startDate.split('-')[2]))
        eDate = dt.datetime(int(endDate.split('-')[0]),int(endDate.split('-')[1]),int(endDate.split('-')[2]))
        # print sDate,eDate
        total_days = (eDate - sDate).days + 1

        lists = glob.glob("../../data/dayresult/trade/*.sqlite")

        totalbench = 0.0
        totalbench_list = []
        daylistcnt = 0
        totalbench_daylist = []
        totalbench_datelist = []
        traderesult_list = []
        currentgain_list =[]
        title_curgain_list = []
        tradedate_list = []

        for day_number in range(total_days):
            search_date = (sDate + datetime.timedelta(days = day_number))
            curpath = str(search_date).split(' ')[0].replace('-','')
            # print curpath

            titlelist = []
            for dbname in lists:
                if curpath in dbname:
                    print dbname
                    try:
                        tablename = 'trade'
                        con = sqlite3.connect(dbname)
                        tmpdf = pd_sql.read_frame("SELECT * from "+tablename, con)
                        # display(HTML(tmpdf.to_html()))
                        # print 'tmpdf len',len(tmpdf)
                        # totaldf = pd.concat([totaldf,tmpdf])
                        con.close()

                        daylistcnt = 0
                        for title in tmpdf['title']:
                            dbdf = tmpdf[tmpdf['title'] == title]
                            

                            if (title in titlelist) == False:
                                titlelist.append(title)
                                # print 'title',title,dbname
                                
                                # display(HTML(dbdf.to_html()))
                                try:
                                    result,traderesult = sr.screenAutoTest_Kalman_gain([dbdf,fetch_date,current_date,'run'\
                                        ,tradestart_day,algo_mode])
                                except Exception,e:
                                    continue
                                count = 0
                                for tmpindex in traderesult.index:
                                    if tmpindex == search_date:
                                        # print tmpindex,count
                                        break
                                    count = count +1
                                nextidx = count +1
                                print 'len(traderesult)',len(traderesult)
                                if len(traderesult) -1 >= nextidx:
                                    print traderesult['bench'][nextidx]                            
                                    endbench = traderesult['bench'][nextidx]                            
                                else:
                                    print result['benchmark'].values[0]
                                    endbench = result['benchmark'].values[0]

                                totalbench = totalbench + (endbench -  traderesult['bench'][count])
                                print 'current gain',(endbench -  traderesult['bench'][count]),title
                                print 'curpath ',curpath,'totalbench',totalbench

                                daylistcnt = daylistcnt + 1
                                
                                currentgain = (endbench -  traderesult['bench'][count])
                                if currentgain > 0.0:
                                	currentgain_list.append(currentgain)
                                	title_curgain_list.append(title)
                                   	tradedate_list.append(curpath)
                                # print results['traderesult']
                                # display(HTML(traderesult.to_html()))
                                # display(HTML(result.to_html()))
                                traderesult_list.append(traderesult)

                        totalbench_list.append(totalbench)
                        totalbench_daylist.append(daylistcnt)
                        totalbench_datelist.append(dbname)
                        
                    except Exception,e:
                        stcore.PrintException()
                        print e
                        con.close() 

        if len(title_curgain_list) > 0:                
        	gaindf = pd.DataFrame({'title':title_curgain_list,'gain':currentgain_list,'date':tradedate_list})
        else:
        	gaindf = pd.DataFrame()

        return totalbench,totalbench_list,totalbench_daylist,traderesult_list,gaindf
    
    except Exception,e:
        stcore.PrintException()
        print e
