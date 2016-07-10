import sqlite3
import pandas as pd
from datetime import datetime,timedelta

newdbstr1 = []
newdbstr2 = []
newdbstr3 = []

def QueryResult(codearg,namearg):

    title = "./data/result/result_db.sqlite"
    con = sqlite3.connect(title)
    tablename = 'result_'+codearg+'_'+namearg
    try:
        if pd.io.sql.table_exists(tablename, con,'sqlite'):
            df = pd.io.sql.read_frame("SELECT * from "+tablename, con)
            if df['BuyorSell'][-1:] == 'buyprice1' or df['BuyorSell'][-1:] == 'buyprice2':
                # print 'buy: '+namearg+' price: ',int(df['Price'][-1:].values)
                global newdbstr1
                global newdbstr2
                global newdbstr3
                newdbstr1.append(namearg)
                newdbstr2.append(int(df['Price'][-1:].values))
                datestr = str(df['Date'][-1:].values.astype(str))
                datestr = datestr.replace('[','')
                datestr = datestr.replace(']','')
                datestr = datestr.replace("'",'')
                datestr = datestr.split(' ')[0]
                newdbstr3.append(datestr)
#     except sdb.sql.SQLError, err:
    except : 
        pass

import csv
import xlrd

book = xlrd.open_workbook("symbols.xls")
sheet = book.sheet_by_name('kospi')
length = sheet.nrows

for rowcnt in range(3,length):
    code = '{0:06d}'.format(int(sheet.row_values(rowcnt)[1]))
    name = sheet.row_values(rowcnt)[2]
    QueryResult(code,name)

new_df = pd.DataFrame({'Name':newdbstr1,'Price':newdbstr2,'Date':newdbstr3})
print new_df.head()
todaydate = datetime.today()
title = './data/result_buylist_db_'+str(todaydate.year)+str(todaydate.month)+str(todaydate.day)+'.sqlite'
print title
con = sqlite3.connect(title)
tablename = 'buylists'
con.execute("DROP TABLE IF EXISTS "+tablename)
pd.io.sql.write_frame(new_df, tablename, con)
con.close()
