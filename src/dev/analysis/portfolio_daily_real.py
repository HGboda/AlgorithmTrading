
from IPython.html.widgets import interact, interactive, fixed
from IPython.html import widgets
from IPython.display import clear_output, display, HTML

import csv
import xlrd

import xlwt
from xlutils.copy import copy # http://pypi.python.org/pypi/xlutils
from xlrd import open_workbook # http://pypi.python.org/pypi/xlrd
from xlwt import easyxf # http://pypi.python.org/pypi/xlwt


pd.set_option('display.width',500)

import glob
global lists
lists = glob.glob("../../data/port/buy/*.sqlite")
# print lists[0]
global buydblen
buydblen = len(lists)

global selllists
selllists = glob.glob("../../data/result2/small/*.sqlite")
# print lists[0]
global selldblen
selldblen = len(selllists)


global dbslidechoice
dbslidechoice = 0
global dbsellchoice
dbsellchoice = 0
global dbname
dbname = ''
global buydf
buydf = 0
global selldf
selldf = 0

def openBuyDB(BuyDBchoice = 0):
    print 'open buy db'
    global dbslidechoice
    dbslidechoice = BuyDBchoice
    global dbname
    global lists
    dbname = lists[BuyDBchoice]
    clear_output()
    print 'dbname',dbname
    try:
        con = sqlite3.connect(dbname)
        tablename = 'buy_table'
        global buydf    
        buydf = pd_sql.read_frame("SELECT * from "+tablename, con)
        con.close()
    except Exception,e:
        print 'error ',e
    
    display(HTML(buydf.to_html()))


def openSellDB(SellDBchoice = 0):
    print 'open sell db'
    global dbsellchoice
    dbname = selllists[SellDBchoice]
    print 'dbname',dbname
    dbsellchoice = SellDBchoice
    



global i1
i1 = interactive(openBuyDB,BuyDBchoice=(0,buydblen-1))
display(i1)

global i2
i2 = interactive(openSellDB,SellDBchoice=(0,selldblen-1))
display(i2)


def ImportBuyDB(widget):
    clear_output()
    print 'import buy db'
    global buydf
    

    titles = buydf['title']
    basemoney = 5000000.0
    investmoney = 0.0

    intitles = []
    indates = []
    inprices = []
    instocknum = []
    inmoney = []

    for title in titles:
        buyprice = buydf['Price'][buydf['title'] == title].values[0]
        buydate = buydf['Date'][buydf['title'] == title].values[0]
        
        
        stocknum = int(basemoney)/int(buyprice)
        investmoney = int(stocknum) * int(buyprice)   
        print title,buyprice,buydate,stocknum,investmoney

        intitles.append(title)
        indates.append(buydate)
        inprices.append(buyprice)
        instocknum.append(stocknum)
        inmoney.append(investmoney)

    

    portdf = pd.DataFrame({'title':intitles,'Date':indates,'Money':inmoney,'Price':inprices,'StockNum':instocknum})
    try:
        dbname = '../../data/port/port_daily.sqlite'
        tablename = 'portfolio_result'
        con = sqlite3.connect(dbname)

        try:
            prevportdf = pd_sql.read_frame("SELECT * from "+tablename, con)    
            if pd.isnull(prevportdf) == False:
                portdf = pd.concat([prevportdf,portdf])
        except Exception,e:
            print 'error read prev portdf',e
            pass    
        con.execute("DROP TABLE IF EXISTS "+tablename)

        pd_sql.write_frame(portdf, tablename, con)
        con.close()
    except Exception,e:
        print 'error ',e
        con.close()
    display(HTML(portdf.to_html()))        


def RunSellSimul(widget):
    print 'run sell simul'
    global dbsellchoice
    global selllists
    dbname = selllists[dbsellchoice]
    clear_output()
    print 'sell dbname',dbname
    try:
        con = sqlite3.connect(dbname)
        query = "SELECT * FROM sqlite_master WHERE type='table'"
        # print query
        global selldf
        selldf = pd.io.sql.read_frame(query,con)
        # print df.tail()
        tablelen = len(selldf['name'])
    except Exception,e:
        print 'sell db open error ',e
        con.close()
        return
    # global buydf
    # buytitles = buydf['title']
    

    title = '../../data/port/port_daily.sqlite'
    selltbname = 'portfolio_sell_result'
    buytbname = 'portfolio_result'

    try:
        con1 = sqlite3.connect(title)
        buyportdf = pd_sql.read_frame("SELECT * from "+buytbname, con1)    
        buytitles = buyportdf['title']
        # con.execute("DROP TABLE IF EXISTS "+tablename)
        # pd_sql.write_frame(portdf, tablename, con)
        # con1.close()
    except Exception,e:
        print 'port db error ',e
        con1.close()
        return
    # display(HTML(selldf.to_html()))    
    
    try:        
        # sellportdf = pd.DataFrame().fillna(0.0)
        for cnt in range(tablelen):
            if selldf['name'][cnt].find('signal') != -1:
                title = selldf['name'][cnt].split('_')
                for buytitle in buytitles:
                    if buytitle == title[2]:
                        stocktitle = title[2]
                        tablename = selldf['name'][cnt]
                        # print 'sell table name',tablename
                        tabledf = pd.io.sql.read_frame("SELECT * from "+tablename, con)    
                        # print stocktitle,tabledf[-1:]['BuyorSell'].values
                        
                        if pd.isnull(tabledf[-1:]['BuyorSell'].values) == False:
                            print title[2],str(tabledf[-1:]['BuyorSell'].values[0])    
                            stance = str(tabledf[-1:]['BuyorSell'].values[0])

                            tmpstr = tabledf['Date'][-1:].values
                            selltabledate = datetime.strptime(tmpstr[0], '%Y-%m-%d %H:%M:%S')
                            
                            tmpstr2 = buyportdf['Date'][-1:].values
                            buytabledate = datetime.strptime(tmpstr2[0], '%Y-%m-%d %H:%M:%S')

                            # if tabledate > datetime.today()-timedelta(days=offsetday) - timedelta(days=1):# and \
                            print buytabledate,selltabledate    
                            if stance.find('sellprice') != -1\
                                and selltabledate > buytabledate:

                                stockprice = tabledf[-1:]['Price'].values[0]
                                sellprice = stockprice
                                buyprice = buyportdf['Price'][buyportdf[buyportdf['title'] == stocktitle].index].values[0]
                                gain = (float(stockprice) - float(buyprice))/float(buyprice)
                                stocknum = buyportdf['StockNum'][buyportdf[buyportdf['title'] == stocktitle].index].values[0]
                                totalmoney = float(stocknum) *(float(sellprice) - float(buyprice))

                                # print stocktitle,gain,stocknum,buyprice,sellprice,totalmoney

                                newdf = pd.DataFrame({'title':[stocktitle],'buydate':buyportdf.Date[buyportdf['title'] == stocktitle]\
                                ,'selldate':tabledf[-1:]['Date'].values[0],'Gain':[gain],'totalMoney':[totalmoney],'stockNum':[stocknum]})

                                portresultdf = pd_sql.read_frame(query,con1)
                                selltablelen = len(portresultdf['name'])
                                selltbfound = False
                                
                                # display(HTML(portresultdf.to_html()))    
                                display(HTML(newdf.to_html()))    

                                for row in range(selltablelen):
                                    if portresultdf['name'][row].find(selltbname) != -1:
                                        print 'sell table found !! '
                                        selltbfound = True
                                        sellportdf = pd_sql.read_frame("SELECT * from "+selltbname,con1)
                                        break
                                
                                if selltbfound == True:
                                    con1.execute("DROP TABLE IF EXISTS "+selltbname)
                                    sellportdf = pd.concat([sellportdf,newdf])
                                else:
                                    sellportdf = newdf
                                # display(HTML(sellportdf.to_html()))        
                                pd_sql.write_frame(sellportdf, selltbname, con1)

                                buyportdf = buyportdf.drop(buyportdf[buyportdf['title'] == stocktitle].index)
                                # display(HTML(buyportdf.to_html()))
                                print 'buyportdf len:',len(buyportdf)
                                if len(buyportdf) == 0:
                                    con1.execute("DROP TABLE IF EXISTS "+buytbname)
                                if len(buyportdf)  > 0:    
                                    con1.execute("DROP TABLE IF EXISTS "+buytbname)
                                    pd_sql.write_frame(buyportdf, buytbname, con1)

        display(HTML(buyportdf.to_html()))
        try:
            sellresultdf = pd_sql.read_frame("SELECT * from "+selltbname, con1)
            print 'sell result ',sellresultdf['totalMoney'].sum()
            display(HTML(sellresultdf.to_html()))                                
        except Exception,e:
            pass
    except Exception,e:
        print 'error1 ',e 
        con.close()                        
        con1.close()
        pass
                                      


    con.close()                        
    con1.close()


run_button1 = widgets.ButtonWidget(description="Import Buy DB")        
run_button1.on_click(ImportBuyDB)
display(run_button1)

run_button2 = widgets.ButtonWidget(description="Run Sell Simul")        
run_button2.on_click(RunSellSimul)
display(run_button2)
