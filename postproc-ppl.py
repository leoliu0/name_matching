import pandas as pd
import re
nm = pd.read_csv('__coname__.csv',names=['idx','name','permno','comnam'])
nm['ppl'] = nm.name.str.findall('^\w{4,} \w\. \w{4,}$').str[0]
nm = nm[nm.ppl.isna()]

with open('surname.txt', 'r') as f:
    sur = set([x.lower().strip() for x in f.readlines() if x])

with open('firstname.txt', 'r') as f:
    first = set([x.lower().strip() for x in f.readlines() if x])

allname = sur | first

def pp(s):
    names = re.findall(r'[\w\d]+',s)
    for x in names:
        if x.lower() not in allname:
            return 0
    return 1

nm['ppl'] = nm.name.map(pp)

nm = nm[nm.ppl==0].drop('ppl',axis=1)

nm.to_csv('__coname__.csv',index=False)
