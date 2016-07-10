from IPython.html.widgets import interact, interactive, fixed
from IPython.html import widgets
from IPython.display import clear_output, display, HTML

import csv
import xlrd
# from stock_simul_function import *
sectorname = u'전기전자'
sectorname2 = u'유통'
filename = '../../Kospi_Symbols.xls'
# filename = '../symbols.xls'
book = xlrd.open_workbook(filename)
sheet = book.sheet_by_name('kospi')
sheet1 = book.sheet_by_name(sectorname)
sheet2 = book.sheet_by_name(sectorname2)
# sheet = book.sheet_by_name('kosdaq')
length1 = sheet1.nrows
length2 = sheet2.nrows


name = u''
code = 0        


for rowcnt in range(0,1):
    code = '{0:06d}'.format(int(sheet1.row_values(rowcnt)[0]))
    name = sheet1.row_values(rowcnt)[1]
    print code,name
    
    histmode = 'histdb'
    srcsite = 1#google
    startdatemode = 2

    bars = RunSimul_realData(code,1,name,'realtime','none',histmode,srcsite,startdatemode)       

    break


dfs_1 = pd.read_excel(filename,sectorname, header=None)
dfs_2 = pd.read_excel(filename,sectorname2, header=None)

dfs_1.columns= ['code','name']
dfs_2.columns= ['code','name']
del dfs_1['code']
del dfs_2['code']
# dfs_elec.head()    
names1 = dfs_1['name'].values
names1 = list(names1)
buydf1 = pd.DataFrame(index = bars.index ,columns=names1).fillna(0)
selldf1 = deepcopy(buydf1)

names2 = dfs_2['name'].values
names2 = list(names2)
buydf2 = pd.DataFrame(index = bars.index ,columns=names2).fillna(0)
selldf2 = deepcopy(buydf2)

# elecbuydf.to_excel(searchsheet+'.xls')

import glob
global lists
lists1 = glob.glob("../../data/result/*.sqlite")

dbname = lists1[-1]
print dbname
try:
    con1 = sqlite3.connect("../../data/result/"+dbname)

    tablename = 'analysis_table'
    tablelist = []

    query = "SELECT * FROM sqlite_master WHERE type='table'"

    tablesdf = pd.io.sql.read_frame(query,con1)
    tablelen = len(tablesdf['name'])

    for rowcnt in range(0,length1):
        code = '{0:06d}'.format(int(sheet1.row_values(rowcnt)[0]))
        name = sheet1.row_values(rowcnt)[1]
        # print 'sheet1',code,name

        buyindex = []
        sellindex = []
        tabledf = []
        tabledates = []
        for cnt in range(tablelen):
            if code in tablesdf['name'][cnt]:
                

                tablename = tablesdf['name'][cnt]
                tabledf = pd.io.sql.read_frame("SELECT * from "+tablename, con1)    
                tabledates = tabledf['Date'].values

                # tabledates =[ datetime.strptime(tdate.split(' ')[0], '%Y-%m-%d')  for tdate in tabledates ]    
                tabledates =[ datetime.strptime(tdate, '%Y-%m-%d %H:%M:%S')  for tdate in tabledates ]
                tabledf.index = tabledates
                tabledf.index.name = 'Date'

                buyindex = tabledf[tabledf['BuyorSell'].str.contains('buyprice')].index
                sellindex = tabledf[tabledf['BuyorSell'].str.contains('sellprice')].index
                # print buyindex
                if len(buyindex) >0:
                    fixbuyindex = buydf1.index.isin(buyindex)
                    buydf1.ix[fixbuyindex,name] = 1
                if len(sellindex) >0:    
                    fixsellindex = selldf1.index.isin(sellindex)
                    selldf1.ix[fixsellindex,name] = 1
                break
        # if rowcnt > 5:
        #     break
except Exception,e:
    print 'exceptoin 1',e
    con1.close()

con1.close()

try:    
    con1 = sqlite3.connect("../../data/result/"+dbname)

    tablename = 'analysis_table'
    tablelist = []

    query = "SELECT * FROM sqlite_master WHERE type='table'"

    tablesdf = pd.io.sql.read_frame(query,con1)
    tablelen = len(tablesdf['name'])

    # print 'length2',length2        
    for rowcnt in range(0,length2):
        code = '{0:06d}'.format(int(sheet2.row_values(rowcnt)[0]))
        name = sheet2.row_values(rowcnt)[1]
        # print 'sheet2',code,name

        buyindex = []
        sellindex = []
        tabledf = []
        tabledates = []
        for cnt in range(tablelen):
            if code in tablesdf['name'][cnt]:
                

                tablename = tablesdf['name'][cnt]
                tabledf = pd.io.sql.read_frame("SELECT * from "+tablename, con1)    
                tabledates = tabledf['Date'].values

                # tabledates =[ datetime.strptime(tdate.split(' ')[0], '%Y-%m-%d')  for tdate in tabledates ]    
                tabledates =[ datetime.strptime(tdate, '%Y-%m-%d %H:%M:%S')  for tdate in tabledates ]
                tabledf.index = tabledates
                tabledf.index.name = 'Date'

                buyindex = tabledf[tabledf['BuyorSell'].str.contains('buyprice')].index
                sellindex = tabledf[tabledf['BuyorSell'].str.contains('sellprice')].index
                # print buyindex
                if len(buyindex) >0:
                    fixbuyindex = buydf2.index.isin(buyindex)
                    buydf2.ix[fixbuyindex,name] = 1
                if len(sellindex) >0:    
                    fixsellindex = selldf2.index.isin(sellindex)
                    selldf2.ix[fixsellindex,name] = 1
                break

except Exception,e:
    print 'exceptoin 2',e
    con1.close()

buydf2.to_excel(sectorname2+'.xls')
con1.close()

buydf1.astype(np.int64)
selldf1.astype(np.int64)
# print 'row sum:',elecbuydf.ix[0].sum()
buydf1['total'] = [buydf1.ix[row].sum() for row in range(len(buydf1))]
selldf1['total'] = [selldf1.ix[row].sum() for row in range(len(selldf1))]

totaldf1 = pd.concat([buydf1['total'],selldf1['total']],axis =1)
totaldf1.columns = ['buy','sell']
totaldf1['diff'] = totaldf1['buy']-totaldf1['sell']

buydf2.astype(np.int64)
selldf2.astype(np.int64)
# print 'row sum:',elecbuydf.ix[0].sum()
buydf2['total'] = [buydf2.ix[row].sum() for row in range(len(buydf2))]
selldf2['total'] = [selldf2.ix[row].sum() for row in range(len(selldf2))]

totaldf2 = pd.concat([buydf2['total'],selldf2['total']],axis =1)
totaldf2.columns = ['buy','sell']
totaldf2['diff'] = totaldf2['buy']-totaldf2['sell']

totaldf3 = pd.concat([totaldf1['buy'],totaldf2['buy']],axis =1)
totaldf3.columns = ['sector1','sector2']
# totaldf3[['sector1','sector2']].plot(kind='bar',figsize=(50,30),stacked=True)
fig = plt.figure(figsize=(50, 30))

ax1 = fig.add_subplot(211,  ylabel=sectorname)
totaldf1['buy'].plot(ax = ax1, kind='bar')
ax2 = fig.add_subplot(212,  ylabel=sectorname2)
totaldf2['buy'].plot(ax = ax2, kind='bar')
plt.show()
# # electotaldf[30:60]
