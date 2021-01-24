#!/usr/bin/python
import argparse,csv,math,os,re,string,sys,json
from collections import Counter, defaultdict
from multiprocessing import Pool
from datetime import datetime as dt
from itertools import *
from unicodedata import normalize

import pandas as pd

from fuzzywuzzy import fuzz
from nltk.tokenize import sent_tokenize

cutoff=92

abbr = [('the',''),('and',''),('of',''),('for',''),('llc','llc'),
        ('Inc', 'incorp'), ('Incorporated','incorp'),
        ('CO', 'Company'), ('COS', 'Company'),('companies', 'Company'),
        ('cor', 'incorp'),('CORP', 'incorp'),('corporation', 'incorp'),
        ('corpor', 'incorp'),('corporat', 'incorp'),('corporat', 'incorp'),
        ('corporate', 'incorp'),('corporatin', 'incorp'),
        ('LTD', 'limited'),('limit', 'limited'),('limite', 'limited'),
        ('company incorp', 'incorp'),('incorp incorp', 'incorp'),
        ('company limited', 'limited'),('incorp limited', 'limited'),
        ('Assn', 'Association'),('Assoc', 'Association'),
        ('intl', 'international'), ('gbl','global'),('natl','national'),
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
        ('INDS', 'industries'), ('industry', 'industries'),
        ('IND', 'industries'),('res','research'),('dev','development'),
        ('IP', ''), ('intellectual property', ''),('intellectual properties', ''),
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
        ('ln','loan'), ('electronic','electronics'),('inst','institution'),
        ('IBM','international business machines'),('motors','motor'),
        ('machine','machines'),('machs','machines'),
        ('american','america'),('AMER','america'),('AMERN','america'),
        ('&', 'and'), ('L\.P','LP'),('L\.L\.P','LLP'),('S\.A','SA'),('S\.p\.A','SPA'),
        ('u s a','united states'), ('usa','united states'),
        ('u s','united states'),('i',''),('ii',''),('iii',''),('iv',''),('v',''),
        ('vi',''),('vii',''),('viii',''),('ix',''),('x','')]

suffix = set(['incorp', 'llc', 'company', 'limited', 'trust','lp','llp','sa','spa',
          'usa', 'holdings', 'group', 'enterprises', 'international', 'gmbh',
          'and','of'])
suffix_regex = '|'.join(suffix)

def abbr_adj(name):  # replace abbr to full
    for string, adj_string in abbr:
        name = re.sub('(?<!\w)' + string + '(?!\w)',
                      ' ' + adj_string,
                      name,
                      flags=re.IGNORECASE)
    return name.strip()


def suffix_adj(name):  # Remove suffix
    for string in suffix:
        name = re.sub(
            '(?<!\w)' + string +
            '(?!\w)',  # The string has to be after some punctuations or space.
            '', name, flags=re.IGNORECASE)
    return name.strip()

def first_two_adj(words):
    if len(words) > 2:
        return abbr_adj(''.join(words[:2]) + ' ' + ' '.join(words[2:]))


def first_three_adj(words):
    if len(words) > 3:
        return abbr_adj(''.join(words[:3]) + ' ' + ' '.join(words[3:]))


def name_preprocessing(z):
    z = z.replace('-REDH', '').replace('-OLD', '').replace('-NEW', '')
    z = abbr_adj(z)
    z = ' '.join(re.findall(r'[\w\d]+',z))
    z = re.sub('The ', '', z, flags=re.I)
    z = z.lower()
    # combining single words...
    s = re.findall('(?<!\w)\w\s\w\s\w(?!\w)', z)
    if s:
        z = re.sub('(?<!\w)\w\s\w\s\w(?!\w)', s[0].replace(' ', ''), z)
    s = re.findall('(?<!\w)\w\s\w(?!\w)', z)
    if s:
        z = re.sub('(?<!\w)\w\s\w(?!\w)', s[0].replace(' ', ''), z)

    return z

removal_regex = re.compile('|'.join(
    [r'\band\b',r'\bof\b',r'\bfor\b',r'\bholdings\b', r'\bholding\b', r'\bgroup\b',
     r'\benterprises\b', r'\binternational\b',r'\bglobal\b']))
eng = set(json.load(open('words_dictionary.json')).keys())

def match(a,b):
    good_y = set()
    pos_y = dict()
    # notice that x is CRSP firms (which is more standard) and y is target names
    x,y = removal_regex.sub('',b).strip().split(),removal_regex.sub('',a).strip().split()
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
            if fuzz.ratio(wx,wy)>threshold:
                match_wx = True
                good_y.add(wy)
        if not match_wx and (wx not in suffix): # every word in X must have a match in Y
            return 0
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
                lst.append([main_index, main_name, base_index, base_name])
    return (main_index, lst)

def match_test(a,b):
    a,b = name_preprocessing(abbr_adj(a)),name_preprocessing(abbr_adj(b))
    score = fuzz.token_set_ratio(suffix_adj(a),suffix_adj(b))
    if score>cutoff:
        return match(a,b)
    else:
        print('failed at cutoff',cutoff,' is',score)

def main():
    with Pool() as p:
        with open(output, 'w', newline='') as w:
            wr = csv.writer(w)
            total_number = len(main_)
            seq = 0
            for index,result in p.imap(unpacking,
                                 main_.values,
                                 chunksize=100):
                seq += 1
                print(f'{seq} out of {total_number}, {index}')
                if result:
                    wr.writerows(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("-o")
    args = parser.parse_args()
    output = args.o if args.o else '__matched__.csv'
    filename = args.input
    print('pre-processing... this could take a while...')
    base_ = pd.read_csv('stocknames_mainclass.csv').dropna()
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
