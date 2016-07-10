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

import glob
global lists
lists = glob.glob("../../data/result2/small/*.sqlite")
# lists = glob.glob("../../data/result2/*.sqlite")
# print lists[0]
global dblength
dblength = len(lists)




def openDB(DBchoice = 0):
    print 'openDB'

    global dbslidechoice
    dbslidechoice = DBchoice
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
    # display(HTML(df.to_html()))
    tablelen = len(df['name'])

    list2d=[[] for i in xrange(20)]





global i1
global length
i1 = interactive(openDB,DBchoice=(0,dblength-1))
display(i1)

global monchoice
monchoice = 1
def openMon(Monchoice = 1):
    print 'openMon',Monchoice
    global monchoice
    monchoice = Monchoice

global i2
monlength = 1
i2 = interactive(openMon,Monchoice=(1,12))
display(i2)


import calendar


def _weekday(year,month,day):
    weekday = calendar.weekday(year, month, day)
    if weekday is 0:
        print "Monday"
    elif weekday == 1:
        print "Tuesday"
    elif weekday == 2:
        print "Wednesday"
    elif weekday == 3:
        print "Thursday"
    elif weekday == 4:
        print "Friday"
    elif weekday == 5:
        print "Saturday"
    elif weekday == 6:
        print "Sunday"
    else:
        print "WTF??"





def ShowResult(widget):
    print 'show result'
    clear_output()

    global dbslidechoice
    global dbname
    global lists
    dbname = lists[dbslidechoice]
    print 'dbname',dbname
    con = sqlite3.connect(dbname)
    query = "SELECT * FROM sqlite_master WHERE type='table'"
    
    global df
    df = pd.io.sql.read_frame(query,con)
    # print df.tail()
    tablelen = len(df['name'])

    curyear = 2014
    global monchoice
    curmonth = monchoice
    
    monrange = calendar.monthrange(curyear,curmonth)
    endmonday = monrange[1]

    list2d=[[] for i in xrange(endmonday)]
    holdlist2d=[[] for i in xrange(endmonday)]

    curgains = 0.0
    basemoney = 5000000.0

    for cnt in range(tablelen):
        if df['name'][cnt].find('signal') != -1:
            tablename = df['name'][cnt]
            titlename = tablename.split('_')[2]
            tabledf = pd.io.sql.read_frame("SELECT * from "+tablename, con)

            for cnt in range(len(tabledf)):
                str1 = tabledf.Date[cnt].split('-')
                dtdate = datetime(int(str1[0]),int(str1[1]),int(str1[2].split(' ')[0]),0)
                tabledf.Date[cnt]= dtdate
    
            tabledf.index = tabledf.Date
            # print 'title',titlename
            
            
            tabledf = tabledf[tabledf.index.year == curyear]
            # display(HTML(tabledf.to_html()))
            # print tabledf.index.month,tabledf.index.day    
            tmptabledf = tabledf[tabledf.index.month == curmonth]
            # print tmptabledf.index.day
            # print tmptabledf[tmptabledf.index.day == tmptabledf.index.day[0]]
        
            tmpgain = 0.0
            indigain = 0.0

            for day in range(endmonday):
                dayix = day+1
                
                if str(tmptabledf[tmptabledf.index.day == dayix]['Stance'].values).find('holding') != -1:
                    list2d[day].append(titlename) 

                if str(tmptabledf[tmptabledf.index.day == dayix]['Stance'].values).find('none') != -1:
                    # print titlename,tmptabledf[tmptabledf.index.day == dayix]['currentgain'].values[0]
                    tmpgain = tmptabledf[tmptabledf.index.day == dayix]['currentgain'].values[0]
                    if tmpgain != 0.0:
                        indigain = indigain + tmpgain

                        # print titlename,'curgains',curgains,'indigain',indigain,'basemoney * indigain',basemoney * indigain

                limitdate = datetime(int(curyear),int(curmonth),int(dayix),0)    
                tmptabledf2 = tmptabledf[:limitdate]
                if str(tmptabledf2[-1:]['BuyorSell'].values).find('buyprice') != -1:
                    holdlist2d[day].append(titlename) 

            curgains = curgains + basemoney * indigain
            
    print '---------------'                    
    print 'total gain money:',curgains
    print '---------------'                    
    for day in range(endmonday):
        dayix = day+1
        print _weekday(curyear,curmonth,dayix),'day :',dayix,'len: ',len(list2d[day]),'hold len: ',len(holdlist2d[day])
        todaydf = pd.DataFrame({'today':list2d[day]})
        holddf = pd.DataFrame({'hold':holdlist2d[day]})
        alldf = pd.concat([todaydf,holddf],axis=1)
        display(HTML(alldf.to_html()))
        # print '--------today title---------'
        # for inday in range(len(list2d[day])):
        #     print list2d[day][inday]
        # print '--------hold title---------'
        # for inday in range(len(holdlist2d[day])):
        #     print holdlist2d[day][inday]

    # for day in range(endmonday):
    #     dayix = day+1
    #     print 'hold day :',dayix,'len: ',len(holdlist2d[day])
    #     for inday in range(len(holdlist2d[day])):
    #         print holdlist2d[day][inday]
                



run_button = widgets.ButtonWidget(description="Show Results")        
run_button.on_click(ShowResult)
display(run_button)
