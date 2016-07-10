%matplotlib inline


import xlrd
import xlwt
from xlutils.copy import copy 
from xlrd import open_workbook 
from xlwt import easyxf 

from IPython.html.widgets import interact, interactive, fixed
from IPython.html import widgets
from IPython.display import clear_output, display, HTML

def readAnalysisDB():
    print 'readAnalysisDB'
    try:
        ''' ----- read analysis db-----'''
        print ' ----- read analysis db-----'
        dbname = 'analysis_corr_auto.sqlite'
        con = sqlite3.connect("../../data/analysis/"+dbname)

        tablename = 'corr_auto_table'
        tablelist = []
        for cnt in range(4):
            tblname = tablename + '_'+str(cnt)
            print tblname
            tablelist.append(pd_sql.read_frame("SELECT * from "+tblname, con))

        analysisdf = pd.concat(tablelist,ignore_index=True)

        # display(HTML(analysisdf.to_html()))    


    except Exception,e:
        print e    
        con.close()

    con.close()

    print len(analysisdf)
    print '----summary------'
    seldf = analysisdf[analysisdf['closecorr'] > 0.9]
    seldf = seldf[seldf['gain'] > 0.0]
    print 'total len',len(seldf)
    print '-----------------'
    limit = seldf['gain'].quantile(0.8)
    seldf = seldf[seldf['gain'] > limit]
    seldf = seldf.sort(['gain'],ascending = False)
    print 'total len 20%:',len(seldf)
    print '-----------------'
    display(HTML(seldf.to_html()))

readAnalysisDB()