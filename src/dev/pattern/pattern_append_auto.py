%matplotlib inline

import xlrd
import xlwt
from xlutils.copy import copy 
from xlrd import open_workbook 
from xlwt import easyxf 

from IPython.html.widgets import interact, interactive, fixed
from IPython.html import widgets
from IPython.display import clear_output, display, HTML



pd.set_option('display.width',500)

filename = '../../Kospi_Symbols.xls'
book = xlrd.open_workbook(filename)
sheet = book.sheet_by_name('kospi')



import glob
global lists
lists = glob.glob("../../data/result2/*.sqlite")
length = len(lists)


global i2
i2 = 0
global df
df= 0
global dbname
dbname = u''
global tablelen
tablelen = 0
global Tablenum
Tablenum = 0
global i3
i3 = 0
global gselectpattern
gselectpattern = []
global gbarsdf
gbarsdf = 0
global gday 
gday = 0

def openDB(DBchoice = 0):
    
    global dbname
    global lists
    dbname = lists[DBchoice]
    print 'dbname',dbname
    con = sqlite3.connect(dbname)
    query = "SELECT * FROM sqlite_master WHERE type='table'"
    print query
    global df
    df = pd.io.sql.read_frame(query,con)
    # print df.tail()

    global tablelen
    tablelen = len(df['name'])/3
    
    if not tablelen == 1:
        tablelen = tablelen -1
    print 'tablelen',tablelen
    global i2
    if not i2 == 0:
        i2.widget.close()
        # i2.visible = False
    i2 = interact(show_tables,Tablechoice=(0,tablelen))        
    con.close()

global i1
i1 = interactive(openDB,DBchoice=(0,length-1))
display(i1)


def show_tables(Tablechoice):
    global df
    global dbname

    try:
        choicenum = Tablechoice*3
        tablename = df.name[choicenum]
        # print 'choicenum:',choicenum,tablename
        global Tablenum
        Tablenum = choicenum
    except:
        pass
    # con1 = sqlite3.connect(dbname)

    # tabledf = pd.io.sql.read_frame("SELECT * from "+tablename, con1)    
    # print tabledf
    # con1.close()

def patternOpen(widget):
    global df
    global dbname
    global Tablenum
    # print 'Tablenum',Tablenum,dbname
    # print df
    tablename = df.name[Tablenum]

    codearg = tablename.split('_')[1]
    namearg = tablename.split('_')[2]
    typearg = 1
    mode = 'realtime'
    dbmode ='none'
    histmode = 'none'
    srcsite = 1
    updbpattern = 'none'


    con1 = sqlite3.connect(dbname)
    tabledf = pd.io.sql.read_frame("SELECT * from "+tablename, con1)    
    print tabledf
    con1.close()

    startdate = '2014-01-01'
    bars = DataFetch(codearg,typearg,namearg,mode,dbmode,histmode,srcsite,updbpattern,startdate)
    # bars_org = deepcopy(bars)
    # print bars.tail()
    global gbarsdf
    gbarsdf = bars

    tabledf['TrigSig'] = False    
    tabledf['TrigSig'][tabledf[tabledf.currentgain < 0.0 ].index] = True
    print tabledf.tail()
    
    fig = plt.figure(figsize=(10, 5))
    fig.patch.set_facecolor('white')     # Set the outer colour to white

    datelist = []
    dateindex = tabledf.index[tabledf['TrigSig'] == True]

    for date in dateindex:
        dateix = date -1
        # print date,dateix,tabledf['Date'][tabledf.index[dateix]]
        datelist.append(tabledf['Date'][tabledf.index[dateix]])
    # datelist = tabledf['Date'][tabledf['TrigSig'] == True].values
    selectpattern = []

    for date in datelist:
        print date,type(date)

        for day in range(len(bars['Close'])):
            if date == str(bars.index[day]):
                print 'found date in bars:',date,' day:',day
                selectpattern.append(day)

    
    global gselectpattern
    gselectpattern = selectpattern            
    global i3
    if not i3 == 0:
        i3.widget.close()
    patternlen = len(selectpattern)    
    if patternlen > 0:
        i3 = interact(show_pattern,Patternchoice=(0,patternlen))        

    

def show_pattern(Patternchoice):
    global df
    global dbname
    global Tablenum
    tablename = df.name[Tablenum]
    global gselectpattern
    global gbarsdf

    gcurday = gselectpattern[Patternchoice]
    global gday 
    gday = gcurday

    print tablename,'pattern length:',len(gselectpattern),' select pattern num:',Patternchoice,str(gbarsdf.index[gcurday])

    
    curpat = gbarsdf.reset_index()

    corrclosex1 = curpat['Close'][gcurday - 9:gcurday + 1].pct_change().cumsum()
    closex1 = pd.DataFrame(corrclosex1, columns=['ClosePc']).fillna(0.0)
    closex1 = closex1.reset_index()

    corropenx1 = curpat['Open'][gcurday - 9:gcurday + 1].pct_change().cumsum()
    openx1 = pd.DataFrame(corropenx1, columns=['OpenPc']).fillna(0.0)
    openx1 = openx1.reset_index()

    corrhighx1 = curpat['High'][gcurday - 9:gcurday + 1].pct_change().cumsum()
    highx1 = pd.DataFrame(corrhighx1, columns=['HighPc']).fillna(0.0)
    highx1 = highx1.reset_index()

    corrlowx1 = curpat['Low'][gcurday - 9:gcurday + 1].pct_change().cumsum()
    lowx1 = pd.DataFrame(corrlowx1, columns=['LowPc']).fillna(0.0)
    lowx1 = lowx1.reset_index()

    corrvolx1 = curpat['Volume'][gcurday - 9:gcurday + 1].pct_change().cumsum()
    volx1 = pd.DataFrame(corrvolx1, columns=['VolPc']).fillna(0.0)
    volx1 = volx1.reset_index()

    

    #''' plot disable
    fig = plt.figure(figsize=(10, 5))
    fig.patch.set_facecolor('white')     # Set the outer colour to white

    ax1 = fig.add_subplot(221,  ylabel='current pattern')
    closex1['ClosePc'].plot(ax=ax1, color='black', lw=2.,label='ClosePc')
    openx1['OpenPc'].plot(ax=ax1, color='red', lw=2.,label='OpenPc')
    highx1['HighPc'].plot(ax=ax1, color='blue', lw=2.,label='HighPc')
    lowx1['LowPc'].plot(ax=ax1, color='#EF15C3', lw=2.,label='LowPc')

    ax2 = fig.add_subplot(222,  ylabel='volume')
    volx1['VolPc'].plot(ax=ax2, color='#2EFE64', lw=2.,label='VolPc')
    ax1.legend(loc=2, bbox_to_anchor=(0.2, 1.5)).get_frame().set_alpha(0.5)

    plt.show()

    con1 = sqlite3.connect(dbname)
    tabledf = pd.io.sql.read_frame("SELECT * from "+tablename, con1)    
    
    con1.close()
    tabledf['TrigSig'] = False    
    tabledf['TrigSig'][tabledf[tabledf.currentgain < 0.0 ].index] = True
    # print tabledf[-20:]
    display(HTML(tabledf.to_html()))
       
    import os
    import gc
    import psutil
    gc.collect()
    

def patternSave(widget):
    print 'pattern Save'
    global gday 
    gday = gcurday
    _patternSaveAuto(gday)



def _patternSaveAuto(selectday):
    print '_patternSaveAuto'
    global df
    global dbname
    global Tablenum
    tablename = df.name[Tablenum]

    codearg = tablename.split('_')[1]
    namearg = tablename.split('_')[2]

    basepos = u"../../data/pattern/append/"
    

    patname = basepos+u'pattern_db_'+codearg+u'_'+namearg+u'.sqlite'
    con = sqlite3.connect(patname)
    
    readlist = []    
    if os.path.isfile(patname):
        
        # con = sqlite3.connect(patname)
        try:
            query = "SELECT * FROM sqlite_master WHERE type='table'"
            df = pd.io.sql.read_frame(query,con)
        except Exception,e:
            print 'error 1',e
            con.close()

        tablelen = len(df)
        print 'read tablelen:',tablelen    
        tablename_base = 'result_'+codearg+'_'+namearg
        
        
        for cnt in range(tablelen):
            tablename = tablename_base+'_'+str(cnt)
            # print 'readtable:',tablename
            patterndf = pd_sql.read_frame("SELECT * from "+tablename, con)
            readlist.append(PatternData(patterndf))
            readlist[cnt].patterndf.index = readlist[cnt].patterndf['Date']
            readlist[cnt].patterndf = readlist[cnt].patterndf.drop('Date',1)

        # con.close()    
        
    

    tablename_base = 'result_'+codearg+'_'+namearg

    if len(readlist) > 0:
        cnt = len(readlist)
    else:
        cnt = 0

    tablename = tablename_base+'_'+str(cnt)
    # print 'writetable:',tablename
    try:
        con.execute("DROP TABLE IF EXISTS "+tablename)

        global gselectpattern
        global gbarsdf
        # global gday 
        gcurday = selectday

        patterndf = gbarsdf[gcurday-9:gcurday+1].reset_index()
        
        pd_sql.write_frame(patterndf, tablename, con)
        print patterndf
        con.close()      
    except Exception,e:
        print 'error2 ',e
        con.close()

def _patternSaveAuto2(tablename,selectday):
    print '_patternSaveAuto2'
    # global df
    # global dbname
    # global Tablenum
    # tablename = df.name[Tablenum]

    codearg = tablename.split('_')[1]
    namearg = tablename.split('_')[2]

    basepos = u"../../data/pattern/append/"
    

    patname = basepos+u'pattern_db_'+codearg+u'_'+namearg+u'.sqlite'
    con = sqlite3.connect(patname)
    
    readlist = []    
    if os.path.isfile(patname):
        
        # con = sqlite3.connect(patname)
        try:
            query = "SELECT * FROM sqlite_master WHERE type='table'"
            df = pd.io.sql.read_frame(query,con)
        except Exception,e:
            print 'error 1',e
            con.close()

        tablelen = len(df)
        print 'read tablelen:',tablelen    
        tablename_base = 'result_'+codearg+'_'+namearg
        
        
        for cnt in range(tablelen):
            tablename = tablename_base+'_'+str(cnt)
            # print 'readtable:',tablename
            patterndf = pd_sql.read_frame("SELECT * from "+tablename, con)
            readlist.append(PatternData(patterndf))
            readlist[cnt].patterndf.index = readlist[cnt].patterndf['Date']
            readlist[cnt].patterndf = readlist[cnt].patterndf.drop('Date',1)

        # con.close()    
        
    

    tablename_base = 'result_'+codearg+'_'+namearg

    if len(readlist) > 0:
        cnt = len(readlist)
    else:
        cnt = 0

    tablename = tablename_base+'_'+str(cnt)
    # print 'writetable:',tablename
    try:
        con.execute("DROP TABLE IF EXISTS "+tablename)

        global gselectpattern
        global gbarsdf
        # global gday 
        gcurday = selectday

        patterndf = gbarsdf[gcurday-9:gcurday+1].reset_index()
        
        pd_sql.write_frame(patterndf, tablename, con)
        print patterndf
        con.close()      
    except Exception,e:
        print 'error2 ',e
        con.close()






run_button = widgets.ButtonWidget(description="Pattern Open")        
run_button.on_click(patternOpen)
display(run_button)

run_button2 = widgets.ButtonWidget(description="Pattern Save")        
run_button2.on_click(patternSave)
display(run_button2)


global stocktitle
stocktitle = u''

def FindMatching(Name=u''):
    print 'FindMatching',Name
    global stocktitle
    stocktitle = Name

    filename = '../../Kospi_Symbols.xls'
    book = xlrd.open_workbook(filename)
    sheet = book.sheet_by_name('kospi')

    # sheet = book.sheet_by_name('kosdaq')
    length = sheet.nrows

    for rowcnt in range(1,length):
        code = '{0:06d}'.format(int(sheet.row_values(rowcnt)[0]))
        name = sheet.row_values(rowcnt)[1]
        try:
            if name == Name:
                print 'matching found:', name, code
            
                global dbname
                global lists

                
                print 'dbname',dbname
                con = sqlite3.connect(dbname)
                query = "SELECT * FROM sqlite_master WHERE type='table'"
                # print query
                global df
                df = pd.io.sql.read_frame(query,con)
                # print df.tail()

                global tablelen
                tablelen = len(df['name'])/3
                global Tablenum
                tablecnt = 0
                for tablename in df['name']:
                    # print tablename,Name
                    namearg = tablename.split('_')[2]

                    if tablename.find('signal') != -1 and tablename.find(Name) != -1 \
                        and namearg == Name and len(namearg) == len(Name):

                        if not tablelen == 1:
                            tablelen = tablelen -1
                        print 'tablelen',tablelen
                        
                        
                        Tablenum = tablecnt

                        codearg = tablename.split('_')[1]
                        namearg = tablename.split('_')[2]
                        typearg = 1
                        mode = 'realtime'
                        dbmode ='none'
                        histmode = 'none'
                        srcsite = 1
                        updbpattern = 'none'


                        con1 = sqlite3.connect(dbname)
                        tabledf = pd.io.sql.read_frame("SELECT * from "+tablename, con1)    
                        # print tabledf
                        con1.close()

                        startdate = '2014-01-01'
                        bars = DataFetch(codearg,typearg,namearg,mode,dbmode,histmode,srcsite,updbpattern,startdate)
                        # bars_org = deepcopy(bars)
                        # print bars.tail()
                        global gbarsdf
                        gbarsdf = bars

                        tabledf['TrigSig'] = False    
                        tabledf['TrigSig'][tabledf[tabledf.currentgain < 0.0 ].index] = True
                        # print tabledf.tail()
                        
                        fig = plt.figure(figsize=(10, 5))
                        fig.patch.set_facecolor('white')     # Set the outer colour to white

                        datelist = []
                        dateindex = tabledf.index[tabledf['TrigSig'] == True]

                        for date in dateindex:
                            dateix = date -1
                            # print date,dateix,tabledf['Date'][tabledf.index[dateix]]
                            datelist.append(tabledf['Date'][tabledf.index[dateix]])
                        # datelist = tabledf['Date'][tabledf['TrigSig'] == True].values
                        selectpattern = []

                        for date in datelist:
                            print date,type(date)

                            for day in range(len(bars['Close'])):
                                if date == str(bars.index[day]):
                                    print 'found date in bars:',date,' day:',day
                                    selectpattern.append(day)

                        
                        global gselectpattern
                        gselectpattern = selectpattern  


                        # global i3
                        # if not i3 == 0:
                        #     i3.widget.close()
                        # patternlen = len(selectpattern)    
                        # if patternlen > 0:
                        #     i3 = interact(show_pattern,Patternchoice=(0,patternlen))        

                        break   
                    tablecnt +=1 
                con.close()
        except:
            pass


imatching = interact(FindMatching, Name=u'')
display(imatching)


def patternAutoSave(widget):
    print 'patternAutoSave'

    global df
    global dbname
    global Tablenum
    
    # tablename = df.name[Tablenum]

    # codearg = tablename.split('_')[1]
    # namearg = tablename.split('_')[2]
    # length = len(df.name)
    # print length
    listlen = len(df['name'])
    for cnt in range(1,70):
        code = '{0:06d}'.format(int(sheet.row_values(cnt)[0]))
        title = sheet.row_values(cnt)[1]
        print title
        
        for row in range(0,listlen):
            clear_output()
            if 'signal' in df.name[row]:
                tablename = df.name[row]
                print tablename
                codearg = tablename.split('_')[1]
                namearg = tablename.split('_')[2]
                if code == codearg:
                    try:
                        con = sqlite3.connect(dbname)
                        tabledf = pd.io.sql.read_frame("SELECT * from "+tablename, con)    
                        
                        con.close()

                        codearg = tablename.split('_')[1]
                        namearg = tablename.split('_')[2]
                        typearg = 1
                        mode = 'realtime'
                        dbmode ='none'
                        histmode = 'none'
                        srcsite = 1
                        updbpattern = 'none'

                        startdate = '2014-01-01'
                        bars = DataFetch(codearg,typearg,namearg,mode,dbmode,histmode,srcsite,updbpattern,startdate)

                        global gbarsdf
                        gbarsdf = bars

                        tabledf['TrigSig'] = False    
                        tabledf['TrigSig'][tabledf[tabledf.currentgain < 0.0 ].index] = True            
                        # print tabledf

                        datelist = []
                        dateindex = tabledf.index[tabledf['TrigSig'] == True]

                        for date in dateindex:
                            dateix = date -1
                            # print date,dateix,tabledf['Date'][tabledf.index[dateix]]
                            datelist.append(tabledf['Date'][tabledf.index[dateix]])
                        # datelist = tabledf['Date'][tabledf['TrigSig'] == True].values
                        selectpattern = []

                        for date in datelist:
                            # print date,type(date)

                            for day in range(len(bars['Close'])):
                                if date == str(bars.index[day]):
                                    print 'found date in bars:',date,' day:',day
                                    selectpattern.append(day)

                        
                        global gselectpattern
                        gselectpattern = selectpattern  
                        selpatlen = len(gselectpattern)

                        for cnt in range(selpatlen):
                            _patternSaveAuto2(tablename,selectpattern[cnt])

                    except Exception,e:
                        print 'error ',e
                        pass
                        con.close()


    







run_button3 = widgets.ButtonWidget(description="Pattern Auto Save")        
run_button3.on_click(patternAutoSave)
display(run_button3)


# %load ../../lib/stockcore.py
# %run ../../lib/stockcore.py


