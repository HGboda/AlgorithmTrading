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

import screen_ver2_0 as sr 



global gStartDate
gStartDate = ''

global gEndDate
gEndDate = ''

def start_tables(StartDate=u''):
    print StartDate
    global gStartDate
    gStartDate = StartDate


i2 = interact(start_tables,StartDate=u'20150727')        
display(i2)

def end_tables(EndDate=u''):
    print EndDate
    global gEndDate
    gEndDate = EndDate


i3 = interact(end_tables,EndDate=u'20151030')        
display(i3)


run_button = widgets.ButtonWidget(description="Read All Lists")

global gAllscreenlistdf
gAllscreenlistdf = 0

def handle_run(widget):
    global gStartDate
    global gEndDate

    print 'Read All Lists '
    allscreenlistdf = sr.readAllLists(gStartDate,gEndDate)
    global gAllscreenlistdf
    gAllscreenlistdf = allscreenlistdf


run_button.on_click(handle_run)

display(run_button)


run_button2 = widgets.ButtonWidget(description="Remaining Lists")
def extract_remaining_lists(widget):
    
    print 'remaining Lists '
    global gAllscreenlistdf

    curscreendf = sr.readResult()
    gAllscreenlistdf = sr.extractRemainingScreenLists(gAllscreenlistdf,curscreendf)

    print 'remaining List len:',len(gAllscreenlistdf)

    sr.screenSaveRemainingResult(gAllscreenlistdf)
    display(HTML(gAllscreenlistdf.to_html()))


run_button2.on_click(extract_remaining_lists)
display(run_button2)

