
from IPython.html.widgets import interact, interactive, fixed
from IPython.html import widgets
from IPython.display import clear_output, display, HTML

import csv
import xlrd
# from stock_simul_function import *

book = xlrd.open_workbook("symbols.xls")
sheet = book.sheet_by_name('kospi')
# sheet = book.sheet_by_name('kosdaq')
length = sheet.nrows


for rowcnt in range(3,length):
    code = '{0:06d}'.format(int(sheet.row_values(rowcnt)[1]))
    name = sheet.row_values(rowcnt)[2]
    try:
        RunSimul(str(code),1,name)
    except:
        pass        