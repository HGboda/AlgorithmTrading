
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


name = u''
code = 0        
def show_args(**kwargs):
    for k,v in kwargs.items():
        print '{0:06d}'.format(int(sheet.row_values(v)[1])),',',sheet.row_values(v)[2]
        global code
        code = '{0:06d}'.format(int(sheet.row_values(v)[1]))
        global name
        name = sheet.row_values(v)[2]

i = interact(show_args,Code=(3,length-1),)

export_button = widgets.ButtonWidget(description="Run")
import xlwt
from xlutils.copy import copy # http://pypi.python.org/pypi/xlutils
from xlrd import open_workbook # http://pypi.python.org/pypi/xlrd
from xlwt import easyxf # http://pypi.python.org/pypi/xlwt




def handle_export(widget):
    global code
    global name
    print 'Run:',code
#     rb = open_workbook('result_summary.xls',formatting_info=True)
#     rb = open_workbook('result_summary_kosdaq.xls',formatting_info=True)
#     r_sheet = rb.sheet_by_index(0) # read only copy to introspect the file
#     wb = copy(rb) # a writable copy (I can't read values out of this, only write to it)
#     w_sheet = wb.get_sheet(0) # the sheet to write to within the writable copy

#     print 'row',r_sheet.nrows
#     newrow =  r_sheet.nrows
#     w_sheet.write(newrow, 0, name)
#     w_sheet.write(newrow, 1, code)
# #     wb.save('result_summary.xls')    
#     wb.save('result_summary_kosdaq.xls')    
    
    RunSimul(str(code),1,name)

export_button.on_click(handle_export)

display(export_button)