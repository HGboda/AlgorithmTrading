from copy import deepcopy

global allselectpattern
global gfoundnumlist

print 'gfoundnumlist:',gfoundnumlist
global gpatternAr
lpatternAr_org = deepcopy(gpatternAr)
allselectpattern_org = deepcopy(allselectpattern)

maxindex = gfoundnumlist.index(max(gfoundnumlist))
maxvalue = gfoundnumlist[maxindex] 
extractid = []
whilecnt = 0

# for patternnum in range(len(allselectpattern)):
#     print 'patternid:',allselectpattern[patternnum].patternid
#     print 'parentid:',allselectpattern[patternnum].parentid
#     print 'targetpatNum:',allselectpattern[patternnum].targetpatNum

while maxvalue != 0:#whilecnt < 2:

    extractid.append(allselectpattern[maxindex].patternid)
    extractid0 = allselectpattern[maxindex].patternid

    # print 'len(foundnumlist):',len(gfoundnumlist),'allselectpattern :',len(allselectpattern),'maxvalue:',maxvalue,'patternid:',extractid0

    deleteitem = []
    parentid = -1
    prevparentid = -1
    
    for cnt in range(maxvalue):
        # print 'loop count:',cnt,'parentid:',parentid,'prevparentid:',prevparentid
        for searchnum in range(len(allselectpattern)):
            parentidFound = False
            if allselectpattern[searchnum].patternid == extractid0:
                parentid = allselectpattern[searchnum].parentid
                # print 'searching parentid:',allselectpattern[searchnum].parentid
                if prevparentid >= parentid:
                    continue
                prevparentid = parentid
                parentidFound = True
                # print 'found parentid:',parentid
                break
        if parentidFound == True:
            for searchnum in range(len(allselectpattern)):
                if allselectpattern[searchnum].parentid == parentid:
                    deleteitem.append(searchnum)
                    # print 'deletitem:',searchnum
                

    print len(deleteitem),deleteitem
    delcnt = 0
    for item in deleteitem:
        # print 'del allselectpattern[item]:',item,'parentid:',allselectpattern[item-delcnt].parentid
        del allselectpattern[item-delcnt]
        delcnt += 1


    for patternnum in range(len(allselectpattern)):    

        for searchnum in range(len(allselectpattern)):
            if allselectpattern[patternnum].parentid != allselectpattern[searchnum].parentid:
                if allselectpattern[patternnum].patternid == allselectpattern[searchnum].patternid:
                    allselectpattern[patternnum].foundnum = allselectpattern[patternnum].getFoundCount() + 1

    foundnumlist = []
    for patternnum in range(len(allselectpattern)):    
        foundnumlist.append(allselectpattern[patternnum].getFoundCount()+1)
    
    gfoundnumlist = foundnumlist

    if len(gfoundnumlist) > 0 :
        maxindex = gfoundnumlist.index(max(gfoundnumlist))
        maxvalue = gfoundnumlist[maxindex] 
        # print 'reorganize len(foundnumlist):',len(foundnumlist),'len(allselectpattern):',len(allselectpattern) ,'maxvalue:',maxvalue
        whilecnt += 1
    else:
        break


print 'extractid:',extractid,'len',len(extractid)    
extractid = list(set(extractid))
print 'extractid:',extractid,'len',len(extractid)    
global gextractid
gextractid = extractid
