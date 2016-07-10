from IPython.html.widgets import interact, interactive, fixed
from IPython.html import widgets
from IPython.display import clear_output, display, HTML

import csv
import xlrd
from stockcore import *
from tradingalgo import *
from data_mani import *
from ver2_0_i import *

import stockcore as stcore
import tradingalgo as talgo 
import data_mani as dmani
import ver2_0_i as v20

# from stock_simul_function import *
book_kosdaq = xlrd.open_workbook("../../Kosdaq_symbols.xls")
sheet_kosdaq = book_kosdaq.sheet_by_name('kosdaq')

book_kospi = xlrd.open_workbook('../../Kospi_Symbols.xls')
sheet_kospi = book_kospi.sheet_by_name('kospi')

length = sheet_kospi.nrows + sheet_kosdaq.nrows

global name 
name = u''
global code
code = 0        

global stocktitle
stocktitle = u''
global daych 
daych = 10

# def show_args(**kwargs):
#     for k,v in kwargs.items():
#         print '{0:06d}'.format(int(sheet.row_values(v)[0])),',',sheet.row_values(v)[1]
#         global code
#         code = '{0:06d}'.format(int(sheet.row_values(v)[0]))
#         global name
#         name = sheet.row_values(v)[1]

global gfetch_date
global gendday_arg
global gtrade_startday
global galgo_mode
def inputDateArg(fetch_date,endday_arg,trade_startday,algo_mode):
    global gfetch_date
    global gendday_arg
    global gtrade_startday
    global galgo_mode

    gfetch_date = fetch_date
    gendday_arg = endday_arg
    gtrade_startday = trade_startday
    galgo_mode = algo_mode

global markettype
markettype = 0
def show_args(Code= 1):
    global markettype
    v = Code
    for cnt in range(sheet_kospi.nrows):
        
        if sheet_kospi.row_values(cnt)[0] == int(sheet_kospi.row_values(v)[0]):
            global code
            global name
            code = '{0:06d}'.format(int(sheet_kospi.row_values(cnt)[0]))
            name = sheet_kospi.row_values(cnt)[1]
            print code,name
            markettype = 1
            break

    for cnt in range(sheet_kosdaq.nrows):
        
        if sheet_kosdaq.row_values(cnt)[0] == int(sheet_kosdaq.row_values(v)[0]):
            global code
            global name
            code = '{0:06d}'.format(int(sheet_kosdaq.row_values(cnt)[0]))
            name = sheet_kosdaq.row_values(cnt)[1]
            print code,name
            markettype = 2
            break        

    # print '{0:06d}'.format(int(sheet.row_values(v)[0])),',',sheet.row_values(v)[1]
    # global code
    # code = '{0:06d}'.format(int(sheet.row_values(v)[0]))
    # global name
    # name = sheet.row_values(v)[1]    
    print code,name

i = interact(show_args,Code=(1,length-1))


export_button = widgets.ButtonWidget(description="Run")
import xlwt
from xlutils.copy import copy # http://pypi.python.org/pypi/xlutils
from xlrd import open_workbook # http://pypi.python.org/pypi/xlrd
from xlwt import easyxf # http://pypi.python.org/pypi/xlwt

global chbox_click
chbox_click = False
global chbox_click2 
chbox_click2 =  True

srcsite = 1#google
# srcsite = 2#yahoo
def handle_export(widget):
    global code
    global name
    if name is u'':
        name = sheet.row_values(1)[1]

    print 'Run:',code,name
    runcount = 1
    writedblog = 'none'
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
    startdatemode = 2
#     dbtradinghist = 'dbtradinghist'
    dbtradinghist = 'none'
    # histmode = 'histdb'
    histmode = 'none'
    plotly = 'plotly'
    stdmode = 'stddb'
    tangentmode = 'tangentdb'
#     tangentmode = 'tan_gen'
    global daych
    global markettype
    try:
        if name == u'dow':
            v20.RunSimul(str(code),4,name,'realtime','dbpattern','none',runcount,srcsite,writedblog,'none','none',startdatemode,
                     'none',plotly,'generate','none',daych,tangentmode)
        elif name == u'nasdaq':
            v20.RunSimul(str(code),4,name,'realtime','dbpattern','none',runcount,srcsite,writedblog,'none','none',startdatemode,
                     'none',plotly,'generate','none',daych,tangentmode)
        elif name == u'sandp':
            v20.RunSimul(str(code),4,name,'realtime','dbpattern','none',runcount,srcsite,writedblog,'none','none',startdatemode,
                     'none',plotly,'generate','none',daych,tangentmode)
        else:
        #     RunSimul(str(code),1,name,'realtime','none','none')
        #     RunSimul(str(code),1,name,'realtime','none','histdb')
        #     RunSimul(str(code),1,name,'realtime','dbpattern','histdb')
        #     RunSimul(str(code),1,name,'realtime','none','none',runcount,srcsite)
            # print appenddb
            try:
                global gfetch_date
                global gendday_arg
                global gtrade_startday
                global galgo_mode
                startdatemode = 3
                v20.RunSimul_Kalman(str(code),markettype,name,'realtime','dbpattern',histmode,runcount,srcsite,writedblog,updbpattern,appenddb,startdatemode,
                         dbtradinghist,plotly,stdmode,'none',daych,tangentmode\
                         ,gfetch_date,gendday_arg,gtrade_startday,galgo_mode)

                
            except Exception,msg:
                if str(msg).find('week check error') != -1:
                    print msg 
                print msg
                pass
    except Exception,e:
        print 'runsimul error:',e
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

chwid2 = widgets.CheckboxWidget(description = 'Apply Append DB', value=True)
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



def show_tables(Name=u''):
    print Name

    global name
    name = Name
    global code
    global markettype
    for cnt in range(sheet_kospi.nrows):
        if sheet_kospi.row_values(cnt)[1] == Name:
            
            code = '{0:06d}'.format(int(sheet_kospi.row_values(cnt)[0]))
            name = sheet_kospi.row_values(cnt)[1]
            rank = sheet_kospi.row_values(cnt)[2]
            print code,name,'kospi',rank
            markettype = 1
            break
    
    for cnt in range(sheet_kosdaq.nrows):
        if sheet_kosdaq.row_values(cnt)[1] == Name:
            
            code = '{0:06d}'.format(int(sheet_kosdaq.row_values(cnt)[0]))
            name = sheet_kosdaq.row_values(cnt)[1]
            rank = sheet_kosdaq.row_values(cnt)[2]
            print code,name,'kosdaq',rank
            markettype = 2
            break

    if Name == u'dow':
        code = '000000'
        name = u'dow'
    elif Name == u'nasdaq':
        code = '000000'
        name = u'nasdaq'
    elif Name == u'sandp':
        code = '000000'
        name = u'sandp'

i2 = interact(show_tables,Namw=u'Type')        
display(i2)


def dayChoice(daychoice = 0):
    print'day Choice :',daychoice 
    global daych 
    daych = daychoice

i3 = interactive(dayChoice,daychoice=(0,60))
display(i3)
