
from IPython.html.widgets import interact, interactive, fixed
from IPython.html import widgets
from IPython.display import clear_output, display, HTML

import csv
import xlrd

import xlwt
from xlutils.copy import copy # http://pypi.python.org/pypi/xlutils
from xlrd import open_workbook # http://pypi.python.org/pypi/xlrd
from xlwt import easyxf # http://pypi.python.org/pypi/xlwt



def simulAuto():
    book = xlrd.open_workbook("../symbols.xls")
    sheet = book.sheet_by_name('kospi')
    # length = sheet.nrows

    # code = '{0:06d}'.format(int(sheet.row_values(rowcnt)[1]))
    # name = sheet.row_values(rowcnt)[2]

    import glob
    global lists
    lists = glob.glob("../data/pattern/extractid*.sqlite")
    # print lists[0]

    length = len(lists)

    srcsite = 1#google
    # srcsite = 2#yahoo
    runcount = 1
    writedblog = 'writedblog'
    # updbpattern = 'updbpattern'
    updbpattern = 'none'
    for title in lists:
        title = title.split('/')[2]
        title = title.split('.')[0]
        title = title.split('\\')[1]
        code = title.split('_')[2]
        print code
        title = title.split('_')[3]
        title = title.decode('euc-kr')
        print title
        try:
            RunSimul(code,1,title,'realtime','dbpattern','none',runcount,srcsite,writedblog,updbpattern)
        except:
            pass   
        import os
        import gc
        import psutil
        gc.collect()


simulAuto()

