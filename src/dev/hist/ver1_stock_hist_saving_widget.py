
from IPython.html.widgets import interact, interactive, fixed
from IPython.html import widgets
from IPython.display import clear_output, display, HTML

import csv
import xlrd
# from stock_simul_function import *

filename = '../../Kospi_Symbols.xls'
# filename = '../symbols.xls'
book = xlrd.open_workbook(filename)
sheet = book.sheet_by_name('kospi')
# sheet = book.sheet_by_name('kosdaq')
length = sheet.nrows


name = u''
code = 0        

global stocktitle
stocktitle = u''

def show_args(**kwargs):
    for k,v in kwargs.items():
        print '{0:06d}'.format(int(sheet.row_values(v)[0])),',',sheet.row_values(v)[1]
        global code
        code = '{0:06d}'.format(int(sheet.row_values(v)[0]))
        global name
        name = sheet.row_values(v)[1]

i = interact(show_args,Code=(1,length-1))

export_button = widgets.ButtonWidget(description="Run")
import xlwt
from xlutils.copy import copy # http://pypi.python.org/pypi/xlutils
from xlrd import open_workbook # http://pypi.python.org/pypi/xlrd
from xlwt import easyxf # http://pypi.python.org/pypi/xlwt

global chbox_click
chbox_click = False
global chbox_click2 
chbox_click2 =  False

srcsite = 1#google
# srcsite = 2#yahoo
def handle_export(widget):
    global code
    global name
    print 'Run:',code,name
    runcount = 1
    # writedblog = 'none'
    writedblog = 'writedblog'
    global chbox_click
    if chbox_click == True:
        updbpattern = 'updbpattern'
    elif chbox_click == False:
        updbpattern = 'none'
    print updbpattern

    if chbox_click2 == True:
        appenddb = 'appenddb'
    elif chbox_click2 == False:
        appenddb = 'none'
    print appenddb
    startdatemode = 1
    # dbtradinghist = 'dbtradinghist'
    dbtradinghist = 'none'

#     RunSimul(str(code),1,name,'realtime','none','none')
#     RunSimul(str(code),1,name,'realtime','none','histdb')
#     RunSimul(str(code),1,name,'realtime','dbpattern','histdb')
#     RunSimul(str(code),1,name,'realtime','none','none',runcount,srcsite)
    RunSimul(str(code),1,name,'realtime','none','none',runcount,srcsite,writedblog,updbpattern,appenddb,startdatemode,
             dbtradinghist)

    import os
    import gc
    import psutil
    gc.collect()
export_button.on_click(handle_export)

display(export_button)


chwid = widgets.CheckboxWidget(description = 'Apply Up Trend', value=False)
display(chwid)

def checkbox_click_handler(name,value):
    global chbox_click 
    if value == True:
        chbox_click = value
        print 'click:',value
    elif value == False:
        chbox_click = value
        print 'click:',value
# help(widgets.CheckboxWidget)    
chwid.on_trait_change(checkbox_click_handler)

chwid2 = widgets.CheckboxWidget(description = 'Apply Append DB', value=False)
display(chwid2)

def checkbox_click_handler2(name,value):
    global chbox_click2 
    if value == True:
        chbox_click2 = value
        print 'click2:',value
    elif value == False:
        chbox_click2 = value
        print 'click2:',value
# help(widgets.CheckboxWidget)    
chwid2.on_trait_change(checkbox_click_handler2)


def show_tables(Name=u'Type'):
    print Name
    global name
    name = Name

    for cnt in range(sheet.nrows):
        if sheet.row_values(cnt)[1] == Name:
            global code
            code = '{0:06d}'.format(int(sheet.row_values(cnt)[0]))
            name = sheet.row_values(cnt)[1]
            print code,name
            break


i2 = interact(show_tables,Name=u'Type')        
display(i2)


