from IPython.html.widgets import interact, interactive, fixed
from IPython.html import widgets
from IPython.display import clear_output, display, HTML

import csv
import xlrd
# from stock_simul_function import *
sectorname = u'전기전자'
filename = '../../Kospi_Symbols.xls'
# filename = '../symbols.xls'
book = xlrd.open_workbook(filename)
sheet = book.sheet_by_name('kospi')
sheet2 = book.sheet_by_name(sectorname)
# sheet = book.sheet_by_name('kosdaq')
length = sheet2.nrows


name = u''
code = 0        


for rowcnt in range(0,1):
    code = '{0:06d}'.format(int(sheet2.row_values(rowcnt)[0]))
    name = sheet2.row_values(rowcnt)[1]
    print code,name
    
    histmode = 'histdb'
    srcsite = 1#google
    startdatemode = 2

    bars = RunSimul_realData(code,1,name,'realtime','none',histmode,srcsite,startdatemode)       

    break


searchsheet = sectorname
dfs_elec = pd.read_excel(filename,searchsheet, header=None)

dfs_elec.columns= ['code','name']
del dfs_elec['code']
# dfs_elec.head()    
elecnames = dfs_elec['name'].values
elecnames = list(elecnames)

elecbuydf = pd.DataFrame(index = bars.index ,columns=elecnames).fillna(0)
elecselldf = deepcopy(elecbuydf)
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

    for rowcnt in range(0,length):
        code = '{0:06d}'.format(int(sheet2.row_values(rowcnt)[0]))
        name = sheet2.row_values(rowcnt)[1]
        # print code,name

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
                fixbuyindex = elecbuydf.index.isin(buyindex)
                fixsellindex = elecselldf.index.isin(sellindex)
                elecbuydf.ix[fixbuyindex,name] = 1
                elecselldf.ix[fixsellindex,name] = 1
                break
        # if rowcnt > 5:
        #     break

except Exception,e:
    print e
    con1.close()
con1.close()

elecbuydf.astype(np.int64)
elecselldf.astype(np.int64)
# print 'row sum:',elecbuydf.ix[0].sum()
elecbuydf['total'] = [elecbuydf.ix[row].sum() for row in range(len(elecbuydf))]
elecselldf['total'] = [elecselldf.ix[row].sum() for row in range(len(elecselldf))]
# print elecbuydf['total']
# print elecselldf['total']
# print 'counts:',elecbuydf.ix[fixindex][name].value_counts()
# elecbuydf.ix[fixindex][name]

# elecbuydf.to_excel(searchsheet+'.xls')

# elecbuydf['total'].plot(kind='bar',figsize=(20,10))
# elecselldf['total'].plot(kind='bar',figsize=(20,10))
electotaldf = pd.concat([elecbuydf['total'],elecselldf['total']],axis =1)
electotaldf.columns = ['buy','sell']
electotaldf['diff'] = electotaldf['buy']-electotaldf['sell']
electotaldf.astype(np.float64)
electotaldf['pct'] = bars['Close'].pct_change()
electotaldf['sig'] = -1
electotaldf['sig2'] = -1
# electotaldf['sig'][:] = np.where(np.logical_or( np.logical_and(electotaldf['diff'][:] >= 0,electotaldf['pct'][:] >= 0) \
#                             , np.logical_and(electotaldf['diff'][:] < 0,electotaldf['pct'][:] < 0)) ,1,-1)

# elecbuydf.to_excel(searchsheet+'.xls')

print 'buy mean',electotaldf['buy'].mean(),'sell mean',electotaldf['sell'].mean()
buymean = electotaldf['buy'].mean()
sellmean = electotaldf['sell'].mean()
electotaldf['sig'][:] = np.where(np.logical_and(electotaldf['sell'][:] > sellmean,  np.logical_and(electotaldf['diff'][:] < 0,electotaldf['pct'][:] > 0)) \
                             ,1,-1)

# electotaldf['sig2'][:] = np.where(np.logical_and(electotaldf['buy'][:] > buymean, np.logical_and(electotaldf['diff'][:] > 0,electotaldf['pct'][:] < 0)) \
#                              ,1,-1)

# display(HTML(electotaldf.to_html()))
fig = plt.figure(figsize=(30, 20))

ax1 = fig.add_subplot(211,  ylabel='Price in $')
bars['Close'].plot(ax= ax1,color='r')
ax1.plot(electotaldf.ix[electotaldf['sig'] == 1].index,
          bars.Close[electotaldf['sig'] == 1],
          '^', markersize=10, color='r')
# ax1.plot(electotaldf.ix[electotaldf['sig2'] == 1].index,
#           bars.Close[electotaldf['sig2'] == 1],
#           'v', markersize=10, color='b')

ax2 = fig.add_subplot(212,  ylabel='diff')
electotaldf['diff'].plot(ax = ax2, kind='bar')
plt.show()
# electotaldf[30:60]
