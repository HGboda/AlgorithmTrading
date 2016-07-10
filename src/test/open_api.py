%matplotlib inline
# from matplotlib import interactive
# interactive(True)

import urllib2
import re
from BeautifulSoup import BeautifulSoup
from lxml import etree
import lxml.html as LH


import numpy as np
import pylab as pl
import matplotlib
import csv
import time
import datetime as dt
from datetime import datetime,timedelta
from matplotlib.dates import date2num
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.finance import candlestick

import numpy as np
import pylab as pl
import matplotlib
import csv
import time
import datetime as dt
from datetime import datetime,timedelta
from time import mktime
import scipy as sp
import pandas as pd
import Quandl
from pandas.io.data import DataReader


testurl = 'http://research.stlouisfed.org/fred2/data/HSN1F.csv'
response = urllib2.urlopen(testurl)
# content = response.read()
df = pd.read_csv(response,index_col=0)
# df.index.name = 'DATE'

print df
fig = plt.figure(figsize=(10, 5))

fig.patch.set_facecolor('white')     # Set the outer colour to white
ax1 = fig.add_subplot(111,  ylabel='New House Sales')

df['VALUE'].plot(ax=ax1, color='r', lw=2.)

