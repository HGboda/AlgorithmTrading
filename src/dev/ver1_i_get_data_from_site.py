import urllib2
import re
from BeautifulSoup import BeautifulSoup
from lxml import etree
import lxml.html as LH

import numpy as np
import pylab as pl
import matplotlib
import csv
import time
import datetime as dt
from datetime import datetime,timedelta
from time import mktime
import scipy as sp
import pandas as pd
import Quandl
from pandas.io.data import DataReader


def fetchData(code):

    financeurl = 'http://finance.naver.com/item/sise_day.nhn?code='+code+'&page=1'

    response = urllib2.urlopen(financeurl)
    content = response.read()
    # response.close()

    soup = BeautifulSoup(content)
#     print soup.tr.td.span
    # dates = soup.findAll("tr")[1]#[0].findAll('span')
    datelist = []
    dates = soup.findAll('td',attrs={'align':'center'})#[0].findAll('span')
    for date in dates:
        # print date.text
        datelist.append(date.text.replace('.','-'))

    closep = []    
    openp = []
    highp = []
    lowp = []
    volume = []        
    colcnt = 0
    contents = soup.findAll('td', attrs={'class':'num'})
    for content in contents:
        content.findAll('span')
        if colcnt == 0:
            closep.append(int(content.text.replace(',','')))
        elif colcnt == 2:
            openp.append(int(content.text.replace(',','')))
        elif colcnt == 3:
            highp.append(int(content.text.replace(',','')))
        elif colcnt == 4:
            lowp.append(int(content.text.replace(',','')))
        elif colcnt == 5:
            volume.append(int(content.text.replace(',','')))
        colcnt += 1
        if colcnt == 6:
            colcnt = 0
        # print content.text,colcnt
        # print content.text
    print 'date',datelist        
    print 'closep',closep
    print 'openp',openp
    print 'highp',highp
    print 'lowp',lowp
    print 'volume',volume
    
    datelen = len(datelist)
    datedflist = []
    for cnt in range(datelen):
        str1 = datelist[cnt].split('-')
        dtdate = datetime(int(str1[0]),int(str1[1]),int(str1[2]),0)
        datedflist.append(dtdate)

    # rtdf = pd.date_range(datedflist[0], datedflist[-1])    
    # print rtdf
    d={'Open':openp, 'High':highp,'Low':lowp,'Close':closep,'Volume':volume }
    # print d
    adjustdf = pd.DataFrame(d,index=datedflist)
    # print adjustdf
    adjustdf.index.name = 'Date'
    # print adjustdf
    return adjustdf