
startdate = datetime.today() - timedelta(days=60)
enddate = datetime.today()
print str(startdate),enddate
totalgain = returns2['total'][str(startdate):str(enddate)].pct_change().cumsum()
# print returns2['total'][str(startdate):str(enddate)]
closepgain = bars['Close'][str(startdate):str(enddate)].pct_change().cumsum()   

print len(closepgain),len(totalgain)

retcorr = totalgain.corr(closepgain)
retcorr
