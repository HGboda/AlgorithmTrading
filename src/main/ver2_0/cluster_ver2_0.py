#!/usr/bin/env python
# -*- coding: UTF-8 -*-

print(__doc__)

# Author: Gael Varoquaux gael.varoquaux@normalesup.org
# License: BSD 3 clause
import datetime

import numpy as np
import pylab as pl
from matplotlib import finance
from matplotlib.collections import LineCollection

from sklearn import cluster, covariance, manifold


from pandas.io.data import DataReader
# import screen as sr 
import stockcore as stcore
import tradingalgo as talgo 
import data_mani as dmani
# import ver1_8_i as v18

import xlrd
import xlwt
from xlutils.copy import copy 
from xlrd import open_workbook 
from xlwt import easyxf 

from IPython.html.widgets import interact, interactive, fixed
from IPython.html import widgets
from IPython.display import clear_output, display, HTML


import webbrowser
import csv
import os
from datetime import datetime,timedelta

import sqlite3
import pandas.io.sql as pd_sql
import pandas as pd


def cluster_fetchData(codearg,typearg,namearg,mode,dbmode,histmode,runcount,srcsite,writedblog,updbpattern,appenddb,startdatemode 
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
        
        
    
    # bars = bars.drop(bars.index[-2])    
    today  = datetime.today()
    todayweek = today.weekday()

    bars['week'] = bars.index.weekday
    tailweekdays = bars['week'][-5:]
    
    bars['bench'] = bars['Close'].pct_change().cumsum()
    bars['benchdiff'] = bars['bench'].diff()

   
    return bars

global dflength
dflength = 0
def clusterSymbol(dbdf):
    global dflength
    saveType = False
    try:

        book_kosdaq = xlrd.open_workbook("../../Kosdaq_symbols.xls")
        sheet_kosdaq = book_kosdaq.sheet_by_name('kosdaq')

        book_kospi = xlrd.open_workbook('../../Kospi_Symbols.xls')
        sheet_kospi = book_kospi.sheet_by_name('kospi')

        quotes2 = []
        nametitles = []
        codearrs = []
        titlefound = False
        for title in dbdf['title']:
            if ' ' in title:
                title  = title.replace(' ','')
            if '&' in title:
                title  = title.replace('&','and')
            if '-' in title:
                title  = title.replace('-','')    
            print 'title',title
            for cnt in range(sheet_kospi.nrows):
            
                if sheet_kospi.row_values(cnt)[1] == title:
                    
                    code = '{0:06d}'.format(int(sheet_kospi.row_values(cnt)[0]))
                    name = sheet_kospi.row_values(cnt)[1]
                    print code,name
                    markettype = 1
                    titlefound = True
                    break

            for cnt in range(sheet_kosdaq.nrows):
                
                if sheet_kosdaq.row_values(cnt)[1] == title:
                    
                    code = '{0:06d}'.format(int(sheet_kosdaq.row_values(cnt)[0]))
                    name = sheet_kosdaq.row_values(cnt)[1]
                    print code,name
                    markettype = 2
                    titlefound = True
                    break  

            if titlefound == False:
                continue   
            titlefound = False         
            try:        
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

                print 'found code',code, name
                bars = cluster_fetchData(str(code),markettype,name,'realtime','dbpattern',histmode,runcount,srcsite,writedblog,updbpattern\
                                        ,appenddb,startdatemode,\
                                         dbtradinghist,plotly,stdmode,'none',daych,tangentmode)
                
                # bars = bars[1:]

                if dflength == 0:
                    dflength = len(bars)
                else:
                    if dflength > len(bars):
                        dflength = len(bars)
                
                quotes2.append(bars)
                nametitles.append(name)
                codearrs.append(code)
                clear_output()
            except Exception,e:
                # print 'error title',name
                pass

        npquotesOpen = []  
        npquotesClose = []   
        count = 0
        for q in quotes2:
            # print q.tail()
            # print pd.isnull(q).any().any()
            # if pd.isnull(q).any().any() == True:
            #     print 'NaN'
            #     continue
            q = q.fillna(0)

            if dflength < len(q):
                q = q[:dflength]
                npquotesOpen.append(q['Open'].values)
                npquotesClose.append(q['Close'].values)
                # print q['Close'].values,'count',count,len(q)    
            else:
                npquotesOpen.append(q['Open'].values)
                npquotesClose.append(q['Close'].values)
                # print q['Close'].values,'count',count,len(q)
            count += 1
            # print len(q.values),'dflength',dflength
        open2 = np.array(npquotesOpen).astype(np.float)         
        close2 = np.array(npquotesClose).astype(np.float)         
        # npquotesClose = []        
        # for q in quotes2:
        #     npquotesClose.append(q['Close'].values)
        # npquotesOpen = np.array([q['Open'].values for q in quotes2])
        # open2 =  npquotesOpen
        # npquotesClose = np.array([q['Close'].values for q in quotes2])
        # close2 =  npquotesClose
        # print npquotesOpen
        # print npquotesClose
        
        variation = (close2 - open2)
        
        symbol_dict = dict(zip(codearrs,nametitles))

        symbols, names = np.array(symbol_dict.items()).T

        edge_model = covariance.GraphLassoCV()

        # standardize the time series: using correlations rather than covariance
        # is more efficient for structure recovery
        tempX = variation.T
        # print tempX,'tempX len',len(tempX)
        X = variation.copy().T
        # print 'open len',len(open2),'close len',len(close2),'variation len',len(variation),'X len',len(X)
        print 'type open',type(open2),'type close',type(close2),'type variation',type(variation),'type X',type(X)
        print 'shape open',open2.shape,'shape close',close2.shape,'shape variation',variation.shape,'shape X',X.shape

        
        X /= X.std(axis=0)
        edge_model.fit(X)

        # ###############################################################################
        # # Cluster using affinity propagation

        _, labels = cluster.affinity_propagation(edge_model.covariance_)
        n_labels = labels.max()

        # print names
        # print 'type symbols',type(symbols),'type names',type(names)
        # for name in names:
        #     print 'name',name
        # print names[0],names[1],names[2],names[3]
        # print 'lables',labels,'n_labels',n_labels,'type labels',type(labels)

        randomtitles = pd.DataFrame()
        for i in range(n_labels+1):
            # print labels == i
            print('Cluster %i: %s' % ((i + 1), ', '.join(names[labels == i])))
            if 1 < len(names[labels==i]) <= 3:
                # print 'random cluster ',np.random.choice(names[labels==i],3)
                tmpdf = pd.DataFrame({'title':np.random.choice(names[labels==i],1)})
                randomtitles = pd.concat([tmpdf, randomtitles])
            elif 3 < len(names[labels==i]) <= 5:
                tmpdf = pd.DataFrame({'title':np.random.choice(names[labels==i],2)})
                randomtitles = pd.concat([tmpdf, randomtitles])
            elif 5 < len(names[labels==i]) <= 7:
                tmpdf = pd.DataFrame({'title':np.random.choice(names[labels==i],4)})
                randomtitles = pd.concat([tmpdf, randomtitles])    
            elif 7 < len(names[labels==i]) :
                tmpdf = pd.DataFrame({'title':np.random.choice(names[labels==i],5)})
                randomtitles = pd.concat([tmpdf, randomtitles])        
                # print randomtitles

        # for i in range(n_labels + 1):
        #     print 'Cluster '+str(i + 1)+', '+ names[labels == i]
        
        # ###############################################################################
        # Find a low-dimension embedding for visualization: find the best position of
        # the nodes (the stocks) on a 2D plane

        # We use a dense eigen_solver to achieve reproducibility (arpack is
        # initiated with random vectors that we don't control). In addition, we
        # use a large number of neighbors to capture the large-scale structure.
        node_position_model = manifold.LocallyLinearEmbedding(
            n_components=2, eigen_solver='dense', n_neighbors=6)

        embedding = node_position_model.fit_transform(X.T).T

        # ###############################################################################
        # Visualization
        pl.figure(1, facecolor='w', figsize=(15, 15))
        pl.clf()
        ax = pl.axes([0., 0., 1., 1.])
        pl.axis('off')

        # Display a graph of the partial correlations
        partial_correlations = edge_model.precision_.copy()
        d = 1 / np.sqrt(np.diag(partial_correlations))
        partial_correlations *= d
        partial_correlations *= d[:, np.newaxis]
        non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)

        # Plot the nodes using the coordinates of our embedding
        pl.scatter(embedding[0], embedding[1], s=100 * d ** 2, c=labels,
                   cmap=pl.cm.spectral)

        # Plot the edges
        start_idx, end_idx = np.where(non_zero)
        #a sequence of (*line0*, *line1*, *line2*), where::
        #            linen = (x0, y0), (x1, y1), ... (xm, ym)
        segments = [[embedding[:, start], embedding[:, stop]]
                    for start, stop in zip(start_idx, end_idx)]
        values = np.abs(partial_correlations[non_zero])
        lc = LineCollection(segments,
                            zorder=0, cmap=pl.cm.hot_r,
                            norm=pl.Normalize(0, .7 * values.max()))
        lc.set_array(values)
        lc.set_linewidths(15 * values)
        ax.add_collection(lc)

        # Add a label to each node. The challenge here is that we want to
        # position the labels to avoid overlap with other labels
        for index, (name, label, (x, y)) in enumerate(
                zip(names, labels, embedding.T)):

            dx = x - embedding[0]
            dx[index] = 1
            dy = y - embedding[1]
            dy[index] = 1
            this_dx = dx[np.argmin(np.abs(dy))]
            this_dy = dy[np.argmin(np.abs(dx))]
            if this_dx > 0:
                horizontalalignment = 'left'
                x = x + .002
            else:
                horizontalalignment = 'right'
                x = x - .002
            if this_dy > 0:
                verticalalignment = 'bottom'
                y = y + .002
            else:
                verticalalignment = 'top'
                y = y - .002
            pl.text(x, y, name, size=10,
                    horizontalalignment=horizontalalignment,
                    verticalalignment=verticalalignment,
                    bbox=dict(facecolor='w',
                              edgecolor=pl.cm.spectral(label / float(n_labels)),
                              alpha=.6))

        pl.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
                embedding[0].max() + .10 * embedding[0].ptp(),)
        pl.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
                embedding[1].max() + .03 * embedding[1].ptp())

        pl.show()
        
        return randomtitles
    except Exception,e:
        stcore.PrintException()    


"""
###############################################################################
# Retrieve the data from Internet

# Choose a time period reasonnably calm (not too long ago so that we get
# high-tech firms, and before the 2008 crash)
d1 = datetime.datetime(2013, 01, 01)
d2 = datetime.datetime(2014, 04, 05)

# kraft symbol has now changed from KFT to MDLZ in yahoo
symbol_dict = {
    '005380.KS': 'HyundaiMotors',
    '005930.KS': 'SamsungElec',
    '005490.KS': 'POSCO',
    '012330.KS': 'MOBIS',
    '051910.KS': 'LGChemi',
    '066570.KS': 'LGElec',
    '034220.KS': 'LGDisplay',
    '009540.KS': 'HyundaiHeavy',
    '055550.KS':  'ShihanBank',
    '032830.KS':  'SamsunLife',
    '105560.KS': 'KBBank',
    '000270.KS': 'Kia',
    '015760.KS':  'HankukElec',
    '096770.KS':   'SKInovation',
    '023530.KS':  'LotteShopping',
    '017670.KS':  'SKTelecom',
    '003550.KS':    'LG',
    '000660.KS' : 'SKHynix',
    '053000.KS':   'WooriFinance',
    '030200.KS':   'KT',
    '000830.KS': 'SamsungC&T',#삼성물산
    '004170.KS':  'ShinsegaeCo',
    '034020.KS':  'DoosanHeavyIndustries',# 두산중공업
    '004020.KS':  'HyundaiSteel',#  현대제철
    '000810.KS': 'SamsungFire',#삼성화재
    '033780.KS':  'KT&G',# KT&G
    '035420.KS': 'Naver ',#  NHN
    '009150.KS': 'SamsungElecMech',#   삼성전기
    '024110.KS': 'IndustrialBank',#  기업은행
    '011170.KS': 'LotteChemical ',#  호남석유
    '010950.KS':  'SOIL',# 에스오일
    '000720.KS': 'HyundaiEngineering',#현대건설
    '010060.KS':   'OCI',
    '010140.KS': 'SamsungHeavyIndus ',#   삼성중공업
    '028050.KS': 'SamsungEngineering',#  삼성엔지니어링
    '029780.KS': 'SamsungCard',#  삼성카드
    '088350.KS': 'HanwhaLife',#  대한생명
    '006400.KS':  'SamsungSDI',#  삼성SDI
    '086790.KS': 'HanaFinancial',#  하나금융지주
    '051900.KS': 'LGHousehold&HealthCare',#  LG생활건강
    '086280.KS': 'Globis',#  글로비스
    '090430.KS': 'Amorepacific',#  아모레퍼시픽
    '078930.KS':   'GS',
    '035250.KS': 'KangwonLand',#  강원랜드
    '010130.KS': 'KoreaZincCo',#  고려아연
    '042660.KS': 'DaewooShipbuilding',#  대우조선해양
    '003600.KS':    'SK',
    '036570.KS': 'Ncsoft',#  엔씨소프트
    '012450.KS': 'SamsungTechwin',#  삼성테크윈
    '003490.KS': 'KoreanAir',#   대한항공
    '006360.KS': 'GSEngineering',#    GS건설
    '052690.KS': 'KEPCOEngineering',#   한전기술
    '006800.KS': 'DaewooSecurities',#    대우증권
    '042670.KS': 'DoosanInfracore',#   두산인프라코어
    '000240.KS': 'HankookTireWorldwide',# 한국타이어월드와이드
    '034730.KS': 'SKC&C',#   SK씨엔씨
    '004800.KS': 'Hyosung',#    효성
    '009830.KS': 'HanwhaChemical',#    한화케미칼
    '016360.KS': 'SamsungSecurities',#   삼성증권
    '010620.KS': 'HyundaiMipo',#   현대미포조선
    '000210.KS': 'DaelimIndustrial',# 대림산업
    '000150.KS': 'Doosan ',# 두산
    '032640.KS': 'LGUPlus',#   LG유플러스
    '006260.KS': 'LS',#    LS
    '047050.KS': 'DaewooInternational',#   대우인터내셔널
    '036460.KS': 'KoreaGas',#   한국가스공사
    '021240.KS': 'Coway',#   웅진코웨이
    '000880.KS': 'Hanwha',# 한화
    '005830.KS': 'DongbuInsurance',#    동부화재
    '069960.KS': 'HyundaiDepartment',#   현대백화점
    '117930.KS': 'HanjinShipping',#  한진해운
    '097950.KS': 'CJCheiljedang',#   CJ제일제당
    '001740.KS': 'SKNetworks',#    SK네트웍스
    '010120.KS': 'LSIndustrial',#   LS산전
    '051600.KS': 'KEPCOPlantService',#   한전KPS
    '012630.KS': 'HyundaiDevelopmentCo',#   현대산업
    '005940.KS': 'WooriInvestment',#    우리투자증권
    '028670.KS': 'PanOcean',#   STX팬오션
    '011070.KS': 'LGInnotek',#   LG이노텍
    '037620.KS': 'MiraeAsset',#   미래에셋증권
    '003450.KS': 'HYUNDAISECURITIES',#    현대증권
    '001800.KS': 'Orion',#    오리온
    '018880.KS': 'HallaVisteon',#   한라공조
    '001040.KS':    'CJ',
    '012750.KS': 'S1',#   에스원
    '060980.KS': 'Mando',#   만도
    '001450.KS': 'HyundaiMarine',#    현대해상
    '097230.KS': 'HanjinHeavyIndustries',#   한진중공업
    '067250.KS': 'STXOffshore',#   STX조선해양
    '115390.KS': 'Lock&Lock',#  락앤락
    '011780.KS': 'KumhoPetroChemical',#   금호석유
    '002990.KS': 'KumhoIndustrial',#    금호산업
    '004990.KS': 'LotteConfectionery',#    롯데제과
    '000100.KS': 'YuhanCorporation',# 유한양행

    # 'MSFT': 'Microsoft',
    # 'IBM': 'IBM',
    # 'TWX': 'Time Warner',
    # 'CMCSA': 'Comcast',
    # 'YHOO': 'Yahoo',
    # 'AMZN': 'Amazon',
    # 'YHOO': 'Yahoo',
    # 'AAPL': 'Apple'
    }
    

symbols, names = np.array(symbol_dict.items()).T
# print type(names)
print symbols,names
# quotes = [finance.quotes_historical_yahoo(symbol, d1, d2, asobject=True)
#           for symbol in symbols]
# open = np.array([q.open for q in quotes]).astype(np.float)
# print type(open)
quotes2 = [DataReader(symbol, "yahoo", datetime.datetime(2012,1,1), datetime.datetime(2014,4,5))
          for symbol in symbols]
# print quotes2[0].tail(),quotes2[1].tail()
# npquotesOpen = [q['Open'].tolist() for q in quotes2]

npquotesOpen = np.array([q['Open'].values for q in quotes2])
# print type(npquotesOpen)
open2 =  npquotesOpen
# print type(open2)
# npquotesClose = [q['Close'].tolist() for q in quotes2]
npquotesClose = np.array([q['Close'].values for q in quotes2])
# print npquotes[0]
close2 =  npquotesClose
# print close2
# open =  np.array([q['Open'].values for q in quotes])
# close =  np.array([q['Close'].values for q in quotes])
# print open

# open = np.array([q.open for q in quotes]).astype(np.float)
# close = np.array([q.close for q in quotes]).astype(np.float)

# # The daily variations of the quotes are what carry most information
variation = (close2 - open2)

# print variation
#############################################################################
# Learn a graphical structure from the correlations
edge_model = covariance.GraphLassoCV()

# standardize the time series: using correlations rather than covariance
# is more efficient for structure recovery
X = variation.copy().T
X /= X.std(axis=0)
edge_model.fit(X)

# ###############################################################################
# # Cluster using affinity propagation

_, labels = cluster.affinity_propagation(edge_model.covariance_)
n_labels = labels.max()

for i in range(n_labels + 1):
    print('Cluster %i: %s' % ((i + 1), ', '.join(names[labels == i])))

# ###############################################################################
# Find a low-dimension embedding for visualization: find the best position of
# the nodes (the stocks) on a 2D plane

# We use a dense eigen_solver to achieve reproducibility (arpack is
# initiated with random vectors that we don't control). In addition, we
# use a large number of neighbors to capture the large-scale structure.
node_position_model = manifold.LocallyLinearEmbedding(
    n_components=2, eigen_solver='dense', n_neighbors=6)

embedding = node_position_model.fit_transform(X.T).T

# ###############################################################################
# Visualization
pl.figure(1, facecolor='w', figsize=(15, 15))
pl.clf()
ax = pl.axes([0., 0., 1., 1.])
pl.axis('off')

# Display a graph of the partial correlations
partial_correlations = edge_model.precision_.copy()
d = 1 / np.sqrt(np.diag(partial_correlations))
partial_correlations *= d
partial_correlations *= d[:, np.newaxis]
non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)

# Plot the nodes using the coordinates of our embedding
pl.scatter(embedding[0], embedding[1], s=100 * d ** 2, c=labels,
           cmap=pl.cm.spectral)

# Plot the edges
start_idx, end_idx = np.where(non_zero)
#a sequence of (*line0*, *line1*, *line2*), where::
#            linen = (x0, y0), (x1, y1), ... (xm, ym)
segments = [[embedding[:, start], embedding[:, stop]]
            for start, stop in zip(start_idx, end_idx)]
values = np.abs(partial_correlations[non_zero])
lc = LineCollection(segments,
                    zorder=0, cmap=pl.cm.hot_r,
                    norm=pl.Normalize(0, .7 * values.max()))
lc.set_array(values)
lc.set_linewidths(15 * values)
ax.add_collection(lc)

# Add a label to each node. The challenge here is that we want to
# position the labels to avoid overlap with other labels
for index, (name, label, (x, y)) in enumerate(
        zip(names, labels, embedding.T)):

    dx = x - embedding[0]
    dx[index] = 1
    dy = y - embedding[1]
    dy[index] = 1
    this_dx = dx[np.argmin(np.abs(dy))]
    this_dy = dy[np.argmin(np.abs(dx))]
    if this_dx > 0:
        horizontalalignment = 'left'
        x = x + .002
    else:
        horizontalalignment = 'right'
        x = x - .002
    if this_dy > 0:
        verticalalignment = 'bottom'
        y = y + .002
    else:
        verticalalignment = 'top'
        y = y - .002
    pl.text(x, y, name, size=10,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            bbox=dict(facecolor='w',
                      edgecolor=pl.cm.spectral(label / float(n_labels)),
                      alpha=.6))

pl.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
        embedding[0].max() + .10 * embedding[0].ptp(),)
pl.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
        embedding[1].max() + .03 * embedding[1].ptp())

pl.show()
"""