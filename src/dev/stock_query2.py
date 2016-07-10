import sqlite3
import pandas as pd
from datetime import datetime,timedelta
import csv
import xlrd

import xlwt
from xlutils.copy import copy 
from xlrd import open_workbook 
from xlwt import easyxf 

from IPython.html.widgets import interact, interactive, fixed
from IPython.html import widgets
from IPython.display import clear_output, display, HTML


book = xlrd.open_workbook("symbols.xls")
sheet = book.sheet_by_name('kospi')
# length = sheet.nrows

# code = '{0:06d}'.format(int(sheet.row_values(rowcnt)[1]))
# name = sheet.row_values(rowcnt)[2]


import glob
global lists
lists = glob.glob("./data/*.sqlite")
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
def show_args(**kwargs):
    for k,v in kwargs.items():
        global dbname
        global lists
        dbname = lists[v]
        print 'dbname',dbname
        con = sqlite3.connect(dbname)
        query = "SELECT * FROM sqlite_master WHERE type='table'"
        print query
        global df
        df = pd.io.sql.read_frame(query,con)
#         print df.tail()
        global tablelen
        tablelen = len(df['name'])
        
        if not tablelen == 1:
            tablelen = tablelen -1
        print 'tablelen',tablelen
        global i2
        if not i2 == 0:
            i2.widget.close()
            # i2.visible = False
        i2 = interact(show_tables,Table=(0,tablelen),Name=u'Type here')        
        # i2.visible = True

global i1

i1 = interact(show_args,DB=(0,length-1))
# i2 = interact(show_tables,Table=(0,tablelen))        



global tableindex
tableindex = 0
global nameindex
nameindex = 0
global choicedf
choicedf = 0
def show_tables(**kwargs):
    for k,v in kwargs.items():
        global df
        global tablename
        global dbname
        if k== 'Table':
            tablename = df.name[v]

            print 'dbname',dbname
            print tablename
            global tableindex
            tableindex = v
        elif k == 'Name':
            # print 'Name:',v            
            
            rowlength = sheet.nrows
            for rowcnt in range(3,rowlength):
                if v == sheet.row_values(rowcnt)[2]:
                    print 'Found:',v
                    global nameindex
                    nameindex = rowcnt

                    code = '{0:06d}'.format(int(sheet.row_values(nameindex)[1]))
                    stockname = sheet.row_values(nameindex)[2]
                    searchtbname = u'result_'+code+'_'+stockname
                    print searchtbname
                    indexdf = df[df['name'] == searchtbname]
                    # print int(indexdf.index)
                    global dbname
                    dbcon = sqlite3.connect(dbname)
                    global choicedf
                    choicedf = pd.io.sql.read_frame("SELECT * from "+searchtbname, dbcon)
                    choicedf.index = choicedf['Date']
                    choicedf.index.name = 'Date'
                    choicedf = choicedf.drop('Date',1)
                    print 'sliderindex:',int(indexdf.index)
                    print choicedf[-10:]
                    lastdate = choicedf.index[-1] 
                    lastdate = lastdate.split(' ')[0]
                    lastdate2 = lastdate.split('-')
                    lastdate3 = datetime(int(lastdate2[0]),int(lastdate2[1]),int(lastdate2[2]))
                    # print lastdate3
                    gainstart = lastdate3- timedelta(days = 7)
                    gainend = datetime.today() 
                    print choicedf[str(gainstart):str(gainend)]['totalGain'].pct_change().cumsum()
                    break





# export_button = widgets.ButtonWidget(description="DB Open")

    
def handle_displaytable(widget):
    print 'handle_displaytable'
    global df
    global dbname
    print dbname
    con1 = sqlite3.connect(dbname)
    global tablename
    tabledf = pd.io.sql.read_frame("SELECT * from "+tablename, con1)

    # print tabledf.tail()
    print tabledf[-10:]

# def handle_export(widget):
#     print 'handle_export'

# export_button.on_click(handle_export)
# display(export_button)

run_button = widgets.ButtonWidget(description="Table Open")        
run_button.on_click(handle_displaytable)
display(run_button)
    