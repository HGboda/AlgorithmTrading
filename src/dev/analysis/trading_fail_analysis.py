from IPython.html.widgets import interact, interactive, fixed
from IPython.html import widgets
from IPython.display import clear_output, display, HTML

import csv
import xlrd



code = '005930'    
name = u'삼성전자'
histmode = 'histdb'
srcsite = 1#google
startdatemode = 2

# bars = RunSimul_realData(code,1,name,'realtime','none',histmode,srcsite,startdatemode)       



import glob
global lists
lists1 = glob.glob("../../data/result/*.sqlite")

dbname = lists1[-1]
print dbname
try:
    con1 = sqlite3.connect("../../data/result/"+dbname)

    query = "SELECT * FROM sqlite_master WHERE type='table'"

    tablesdf = pd.io.sql.read_frame(query,con1)
    tablelen = len(tablesdf['name'])
    print 'tablesdf',tablesdf
    print 'table length',tablelen
    tabledf = []
    
    titles = []
    failcounts = []
    benchmarks = []
    for cnt in range(tablelen):
        
        if 'signal' in tablesdf['name'][cnt]:
            # print tablesdf['name'][cnt]
            title =  tablesdf['name'][cnt].split('_')
            title = title[2]
            # print title
            
            tablename = tablesdf['name'][cnt]
            tabledf = pd.io.sql.read_frame("SELECT * from "+tablename, con1)    
            tabledates = tabledf['Date'].values
            tabledf['failsig'] = 0
            tabledf['failsig'] = np.where(tabledf.currentgain < 0.0,tabledf.currentgain,0)

            titles.append(title)
            failcounts.append(len(tabledf[tabledf.failsig < 0]))
            
            summarytable = tablesdf['name'][cnt]
            summarytable = summarytable.replace('signal','summary')
            tabledf = pd.io.sql.read_frame("SELECT * from "+summarytable, con1)    
            benchmarks.append(tabledf['Benchmark'])



    faildfs = pd.DataFrame({'title':titles,'failcounts':failcounts,'benchmark':benchmarks})
    faildfs = faildfs.sort(['failcounts'],ascending= True)
    display(HTML(faildfs.to_html()))
except Exception,e:
    print e
    con1.close()
con1.close()

