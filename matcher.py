#!/usr/bin/python
import argparse,csv,math,os,re,string,sys,json,pkg_resources,pathlib
from collections import Counter, defaultdict
from multiprocessing import Pool
from datetime import datetime as dt
from itertools import *
from unicodedata import normalize
from tqdm import tqdm

import pandas as pd

from fuzzywuzzy import fuzz
from nltk.tokenize import sent_tokenize

cutoff=80

abbr = [('the',''),('and',''),('of',''),('for',''),('llc','llc'),
        ('Inc', 'incorp'), ('Incorporated','incorp'),
        ('CO', 'company'), ('COS', 'Company'),('companies', 'Company'),
        ('cor', 'incorp'),('CORP', 'incorp'),('corporation', 'incorp'),
        ('coporation', 'incorp'),
        ('corpor', 'incorp'),('corporat', 'incorp'),('corporat', 'incorp'),
        ('corporate', 'incorp'),('corporatin', 'incorp'),
        ('LTD', 'limited'),('limit', 'limited'),('limite', 'limited'),
        ('company incorp', 'incorp'),('incorp incorp', 'incorp'),
        ('company limited', 'limited'),('incorp limited', 'limited'),
        ('Assn', 'Association'),('Assoc', 'Association'),
        ('intl', 'international'), ('gbl','global'),('natl','national'),
        ('nat','national'),
        ('int', 'international'),('univ','university'),
        ('MOR', 'Mortgage'), ('Banc', 'BankCorp'),('bk', 'BankCorp'),
        ('bancshares ', 'bankcorp'),('bankshares ', 'bankcorp'),
        ('stores','store'),('brand','brands'),('gen','general'),
        ('solutions ', 'solution'),('science','sciences'),('sci','sciences'),
        ('work', 'works'),('device','devices'),('operation','operations'),
        ('tool', 'tools'),('network','networks'),('material','materials'),
        ('grp', 'group'),('cap','capital'),('FINL','financial'),
        ('THRU', 'Through'), ('COMM', 'Communication'),('MGMT','Management'),
        ('INVT', 'investments'),('INV', 'investments'),('investment', 'investments'),
        ('PTNR','partner'),('ADVR','advisors'),
        ('laboratory','laboratories'),('lab','laboratories'),('labs','laboratories'),
        ('ins','insurance'),('insur','insurance'),('insure','insurance'),
        ('tech', 'technologies'), ('technology', 'technologies'),
        ('INDS', 'industries'), ('industry', 'industries'), ('indl', 'industries'),
        ('IND', 'industries'),('res','research'),('dev','development'),
        ('IP', ''), ('intellectual property', ''),('intellectual properties', ''),
        ('intellectual', ''),('patents',''),('patent',''),('trademark',''),('trademarks',''),
        ('marketing',''),
        ('property', 'properties'), ('Mort', 'Mortgage'), ('Thr', 'Through'),
        ('Sec', 'Securities'), ('BANCORPORATION', 'BankCorp'),
        ('RESOURCE', 'Resources'), ('Holding', 'Holdings'),
        ('Security', 'Securities'), ('ENTERPRISE', 'Enterprises'),
        ('funding', 'fundings'), ('networks', 'systems'), ('chem', 'chemical'),
        ('SYS', 'systems'), ('MFG', 'manufacturing'), ('Prod', 'products'),
        ('Pharma', 'Pharm'),('Pharmaceu', 'Pharm'),('Pharmaceuti', 'Pharm'),
        ('Pharmace', 'Pharm'),('Pharmaceut', 'Pharm'), ('Pharmaceutical', 'Pharm'),
        ('Product', 'products'), ('svcs','services'),('service','services'),
        ('production','productions'),('saving','savings'),('svgs','savings'),
        ('ln','loan'),('electronic','electronics'),('elect','electronics'),
        ('electrs','electronics'),
        ('inst','institution'),
        ('motors','motor'),
        ('machine','machines'),('machs','machines'),('teleg','telegraph'),
        ('tel','telephone'),('tel','telephone'),('ry','railway'),
        ('american','america'),('AMER','america'),('AMERN','america'),
        ('&', 'and'), ('L\.P','LP'),('L\.L\.P','LLP'),('S\.A','SA'),('S\.p\.A','SPA'),
        ('u s a','usa'), ('usa','usa'), ('u s','usa'), ('us','usa'),
        # Japanese suffix
        ('(?<!^)kk',''),('(?<!^)gk',''),('(?<!^)yk',''),('(?<!^)gmk',''),
        ('(?<!^)gsk',''),('(?<!^)nk',''),('(?<!^)tk',''),
        # Germany suffix
        ('(?<!^)ev',''),('(?<!^)rv',''),('(?<!^)kgaa',''),('gmbh co',''),('(?<!^)ag co',''),('(?<!^)se co',''),
        ('gmbh',''),('(?<!^)ag',''),('(?<!^)se',''),('(?<!^)ug',''),
        # French suffix
        ('(?<!^)sep',''),('(?<!^)snc',''),('(?<!^)scs',''),('(?<!^)sca',''),
        ('(?<!^)sci',''),('(?<!^)sarl',''), ('(?<!^)eurl',''),('(?<!^)sa',''),
        ('(?<!^)scop',''),(r'sas$',''),(r'sasu$',''),
        # Swedish suffix
        (r'ab$',''),(r'lm$','')
        ]


common_abbr = set([x for _,x in abbr if x !=''])
# for some abbreviations, we have to hard code it.
hardcode = [('HP', 'HEWLETT PACKARD'),('IBM','international business machines'),
            (r'DU PONT$','DU PONT EI DE NEMOURS'),
            ('HITACHI','HITACHI matchit'),('exxon','exxon matchit'),
            ('SIEMENS','SIEMENS matchit'),('GTE','GTE matchit'),
            ('north  america philips','philips')]


suffix = set(['incorp', 'llc', 'company', 'limited', 'trust','lp','llp','sa','spa',
          'usa', 'holdings', 'group', 'enterprises', 'international', 'gmbh','kk'
          'and','of','north american',
            # Japanese suffix
            'kk','gk','yk','gmk','gsk','nk','tk',
              ]
             )
suffix_regex = '|'.join(suffix)

removal_regex = re.compile('|'.join(
    [r'\band\b',r'\bof\b',r'\bfor\b',r'\bholdings\b', r'\bholding\b', r'\bgroup\b',
     r'\benterprises\b', r'\binternational\b',r'\bglobal\b']))

def loc(f):
    return os.path.join(pathlib.Path(__file__).parent.absolute(),f)

eng = set(json.load(open('words_dictionary.json')).keys())
eng = eng | set([x.lower().strip() for x in (open(loc('surname.txt')).readlines())])
eng = eng | set([x.lower().strip() for x in (open(loc('firstname.txt')).readlines())])


def abbr_adj(name):  # replace abbr to full
    for string, adj_string in abbr+hardcode:
        name = re.sub('(?<!\w)' + string + '(?!\w)',
                      ' ' + adj_string,
                      name,
                      flags=re.IGNORECASE)
    return name.strip().lower()


def suffix_adj(name):  # Remove suffix
    for string in suffix:
        name = re.sub(
            '(?<!\w)' + string +
            '(?!\w)',  # The string has to be after some punctuations or space.
            '', name, flags=re.IGNORECASE)
    return name.strip()


def name_preprocessing(z):
    z = z.replace('-REDH', '').replace('-OLD', '').replace('-NEW', '')
    z = ' '.join(re.findall(r'[\w\d]+',z))
    z = re.sub('The ', '', z, flags=re.I)
    z = z.lower()
    # combining single words...
    a = ''.join(re.findall(r'\b\w\s\b',z))
    if a:
        b = a.replace(' ','')
        z = z.replace(a,b+' ')

    #TODO: refactor the code to a function
    for string, adj_string in [('i',''),('ii',''),('iii',''),('iv',''),('v',''),
                         ('vi',''),('vii',''),('viii',''),('ix',''),('x','')]:
        z = re.sub('(?<!\w)' + string + '(?!\w)',
                      ' ' + adj_string, z, flags=re.IGNORECASE)
    z = abbr_adj(z)
    return z.strip().lower()


def check_double(a,b):
    ''' account for double ('BALL & BALL CARBURETOR COMPANY','BALL CORP')'''
    for a1,a2 in combinations(a,2):
        if fuzz.ratio(a1,a2)>89:
            for b1,b2 in combinations(b,2):
                if fuzz.ratio(b1,b2)>89:
                    if fuzz.ratio(a1,b1)<=89:
                        return False
                    else:
                        break
            else:
                return False
def remove_meaningless(s):
    return removal_regex.sub('',s).strip()

ban_list = ('organization','organization','academy','university','commission')

def match(a,b):
    for w in ban_list:
        if w in a:
            return False
    good_y = set()
    pos_y = dict()
    # notice that x is CRSP firms (which is more standard) and y is target names
    x,y = remove_meaningless(b).split(),remove_meaningless(a).split()
    if len(x)==0:
        return False
    if len(x)==1:
        if (x[0] in eng) or (len(x[0])<5):
            return False
    for wx in x:
        match_wx = False
        for n,wy in enumerate(y,start=1):
            if wy not in pos_y:
                pos_y[wy] = n
            if (len(x) == len(y)) and (len(x)>4):
                threshold = 75
            else:
                threshold = 89
            if fuzz.ratio(wx,wy)>threshold and wx[0]==wy[0]: # first letter must match
                match_wx = True
                good_y.add(wy)
        if not match_wx and (wx not in suffix): # every word in X must have a match in Y
            return False

    if check_double(x,y) is False:
        return False
    if check_double(y,x) is False:
        return False

    __good_y= set(good_y) - common_abbr - suffix

    if len(__good_y - eng)>0:
        if len(__good_y)*len([w for q in __good_y for w in q if w in string.ascii_letters])>20:
            return True
    bad_y = set(y) - set(good_y) - suffix
    if len(bad_y)==0: # no additional words except for suffix in Y means good match
        return True
    for bad_wy in bad_y:
        if pos_y[bad_wy]<=len(x): # all additional words in Y must appear after X
            return False
    return True

def unpacking(main_row):
    lst = []
    main_index, main_name, main_disamb, main_wosuf = main_row
    for base_index, base_name, base_disamb, base_wosuf in base_.values:
        if fuzz.token_set_ratio(main_wosuf,base_wosuf)>cutoff:
            if match(main_disamb, base_disamb):
                lst.append([main_index, main_name, base_index, base_name,
                            fuzz.token_set_ratio(main_disamb, base_disamb)])
    return lst

def match_test(a,b):
    a,b = name_preprocessing(a),name_preprocessing(b)
    score = fuzz.token_set_ratio(suffix_adj(a),suffix_adj(b))
    print(suffix_adj(a),suffix_adj(b))
    if score>cutoff:
        return match(a,b)
    else:
        print('failed at cutoff',cutoff,' is',score)

def main():
    with Pool() as p:
        with open(output, 'w', newline='') as w:
            wr = csv.writer(w)
            with tqdm(total=len(main_)) as pb:
                for result in p.imap(unpacking,
                                 main_.values,
                                 chunksize=100):
                    if result:
                        wr.writerows(result)
                    pb.update()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("-b")
    parser.add_argument("-o")
    args = parser.parse_args()
    output = args.o if args.o else '__match__.csv'
    filename = args.input
    print('pre-processing... this could take a while...')
    basefile = args.b if args.b else 'stocknames.csv'
    base_ = pd.read_csv(basefile).dropna()
    main_ = pd.read_csv(filename).dropna()
    # disambiguation
    base_['disamb'] = base_[base_.columns[1]].map(name_preprocessing)
    main_['disamb'] = main_[main_.columns[1]].map(name_preprocessing)
    base_['wo_suffix'] = base_['disamb'].map(suffix_adj)
    main_['wo_suffix'] = main_['disamb'].map(suffix_adj)

    wastime = dt.now()
    print(wastime,'start now ...')
    main()
    print(dt.now(), 'finished, takes',
          (dt.now() - wastime).total_seconds() / 60, 'minutes')
