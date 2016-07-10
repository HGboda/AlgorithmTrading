


import xlrd
import xlwt
from xlutils.copy import copy 
from xlrd import open_workbook 
from xlwt import easyxf 

from IPython.html.widgets import interact, interactive, fixed
from IPython.html import widgets
from IPython.display import clear_output, display, HTML

from stockcore import *
from tradingalgo import *
from data_mani import *
import stockcore as stcore
import tradingalgo as talgo 
import data_mani as dmani




book_kosdaq = xlrd.open_workbook("../../Kosdaq_symbols.xls")
sheet_kosdaq = book_kosdaq.sheet_by_name('kosdaq')

book_kospi = xlrd.open_workbook('../../Kospi_Symbols.xls')
sheet_kospi = book_kospi.sheet_by_name('kospi')

length = sheet_kospi.nrows + sheet_kosdaq.nrows


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
    global stockprice
    stockprice = Price
    print Price
    

i3 = interact(show_price,Price=u'')        
display(i3)

global investmoney
investmoney = u''
def show_money(Money=u''):
    global investmoney
    investmoney = Money
    print Money
    

i3 = interact(show_money,Money=u'')        
display(i3)

global portdbname
portdbname = u''
def show_portdbname(portName=u''):
    global portdbname
    portdbname = portName
    print portName
    

i4 = interact(show_portdbname,Name=u'portdb')        
display(i4)


global buydate
buydate = u''
def show_buydate(buydateIn=u''):
    global buydate
    buydateIn = buydateIn+' 00:00:00'
    buydate = datetime.strptime(buydateIn, '%Y-%m-%d %H:%M:%S')
    print buydateIn
    
i5 = interact(show_buydate,Name=u'buydateIn')        
display(i5)


global selldate
selldate = u''
def show_selldate(selldateIn=u''):
    global selldate
    selldate = selldateIn
    print selldate
    

i6 = interact(show_selldate,Name=u'selldateIn')        
display(i6)




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
    global upperprice
    global lowerprice

    stocknum = int(investmoney)/int(stockprice)

    investmoney = int(stocknum) * int(stockprice)
    global portdbname
    title = '../../data/port/'+portdbname+'.sqlite'
    tablename = 'portfolio_result'
    clear_output()
    if os.path.isfile(title):
        try:
            con = sqlite3.connect(title)
            portdf = pd_sql.read_frame("SELECT * from "+tablename, con)
            con.close()
            print 'portdf len',len(portdf)
            for porttitle in portdf['title']:
                if porttitle == stocktitle:
                    portdf['Price'][portdf[portdf['title'] == stocktitle].index] = stockprice
                    portdf['StockNum'][portdf[portdf['title'] == stocktitle].index] = stocknum
                    portdf['Money'][portdf[portdf['title'] == stocktitle].index] = investmoney
                    portdf['Date'][portdf[portdf['title'] == stocktitle].index] = buydate
                    # portdf['UpperPrice'][portdf[portdf['title'] == stocktitle].index] = upperprice
                    # portdf['LowerPrice'][portdf[portdf['title'] == stocktitle].index] = lowerprice

                    con = sqlite3.connect(title)
                    con.execute("DROP TABLE IF EXISTS "+tablename)

                    pd_sql.write_frame(portdf, tablename, con)
                    con.close()
                    display(HTML(portdf.to_html()))
                    return
        except Exception,e:
            con.close()
            stcore.PrintException()
            portdf = pd.DataFrame().fillna(0.0)
    else:
        portdf = pd.DataFrame().fillna(0.0)


    # gettime = foundap[foundap['title'] == stocktitle][-1:]['time'].values
    # print gettime[0]
    # print foundap[foundap['title'] == stocktitle][-1:],stockprice
    # tmpdf = deepcopy(foundap[foundap['title'] == stocktitle][-1:])
    # tmpdf['Price'] = pd.Series([stockprice], index=tmpdf.index)
    # tmpdf['StockNum'] = pd.Series([stocknum], index=tmpdf.index)
    tmpdf = pd.DataFrame({'title':[stocktitle],'Price':[stockprice],'StockNum':[stocknum],'Money':[investmoney],'Date':[buydate]})
        # ,'UpperPrice':[upperprice],'LowerPrice':lowerprice})
    portdf = pd.concat([portdf,tmpdf])
    print '------------summary--------------'
    print 'total Money:',portdf['Money'].sum()
    display(HTML(portdf.to_html()))
    try:
        con = sqlite3.connect(title)
        con.execute("DROP TABLE IF EXISTS "+tablename)

        pd_sql.write_frame(portdf, tablename, con)
        con.close()
        print 'write done',title,tablename
    except Exception,e:
        con.close()
        stcore.PrintException()


def portfoliosell(widget):
    print 'portfolio sell'
    
    global stockprice
    global stocktitle
    global selldate
    global portdbname
    title = '../../data/port/'+portdbname+'.sqlite'
    selltbname = 'portfolio_sell_result'
    buytbname = 'portfolio_result'
    sellprice = stockprice  

    today = datetime.now()
    todaydate = today.strftime('%Y-%m-%d')
    # print title
    clear_output()
    if os.path.isfile(title):
        print 'sell start'
        # today = datetime(int(todaydate.year),int(todaydate.month),int(todaydate.day))

            
        try:
            con = sqlite3.connect(title)

            buyportdf = pd_sql.read_frame("SELECT * from "+buytbname, con)

            query = "SELECT * FROM sqlite_master WHERE type='table'"
            df = pd_sql.read_frame(query,con)
            tablelen = len(df['name'])
            selltbfound = False
            for row in range(tablelen):
                if df['name'][row].find(selltbname) != -1:
                    selltbfound = True
                    break
            
            if selltbfound == True:
                sellportdf = pd_sql.read_frame("SELECT * from "+selltbname,con)

            display(HTML(buyportdf.to_html()))
            print 'stocktitle',stocktitle

            for porttitle in buyportdf['title'] :
                if porttitle == stocktitle:
                
                    buyprice = buyportdf['Price'][buyportdf[buyportdf['title'] == stocktitle].index].values[0]
                    gain = (float(stockprice) - float(buyprice))/float(buyprice)
                    # print buyprice,gain
                    stocknum = buyportdf['StockNum'][buyportdf[buyportdf['title'] == stocktitle].index].values[0]
                    print gain,stocknum,buyprice,sellprice
                    totalmoney = float(stocknum) *(float(sellprice) - float(buyprice))
                    newdf = pd.DataFrame({'title':[stocktitle],'buydate':buyportdf.Date[buyportdf['title'] == stocktitle]\
                        ,'selldate':[selldate],'Gain':[gain],'totalMoney':[totalmoney],'stockNum':[stocknum]})

                    if selltbfound == True:
                        con.execute("DROP TABLE IF EXISTS "+selltbname)
                        sellportdf = pd.concat([sellportdf,newdf])
                    else:
                        sellportdf = newdf
                    pd_sql.write_frame(sellportdf, selltbname, con)
                    display(HTML(sellportdf.to_html()))
                    
                    buyportdf = buyportdf.drop(buyportdf[buyportdf['title'] == stocktitle].index)
                    display(HTML(buyportdf.to_html()))
                    con.execute("DROP TABLE IF EXISTS "+buytbname)
                    pd_sql.write_frame(buyportdf, buytbname, con)
            
            sellresultdf = pd_sql.read_frame("SELECT * from "+selltbname, con)
            print 'sell result ',sellresultdf['totalMoney'].sum()
            con.close()            
        except Exception,e:
            con.close()
            stcore.PrintException()

def curStatus(widget):
    global portdbname
    title = '../../data/port/'+portdbname+'.sqlite'
    tablename = 'portfolio_result'
    clear_output()
    if os.path.isfile(title):
        try:
            con = sqlite3.connect(title)
            portdf = pd_sql.read_frame("SELECT * from "+tablename, con)
            con.close()

            display(HTML(portdf.to_html()))    
        except Exception,e:
            con.close()
            stcore.PrintException()
    else:
        portdf = pd.DataFrame().fillna(0.0)

    

run_button2 = widgets.ButtonWidget(description="Portfolio Buy")        
run_button2.on_click(portfoliobuy)
display(run_button2)

run_button3 = widgets.ButtonWidget(description="Portfolio Sell")        
run_button3.on_click(portfoliosell)
display(run_button3)

run_button4 = widgets.ButtonWidget(description="Current Port")        
run_button4.on_click(curStatus)
display(run_button4)
