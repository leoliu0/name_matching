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

abbr = [('the',''),('and',''),('of',''),('for',''),('llc','llc'),
        ('Inc', 'incorp'), ('Incorporated','incorp'),
        ('CORP', 'incorp'),('corporation', 'incorp'),
        ('corpor', 'incorp'),('corporat', 'incorp'),
        ('corporate', 'incorp'),('corporatin', 'incorp'),
        ('Assn', 'Association'),('Assoc', 'Association'),
        ('intl', 'international'), ('gbl','global'),('CO', 'Company'),
        ('LTD', 'Limited'),('limit', 'Limited'),('limite', 'Limited'),
        ('MOR', 'Mortgage'), ('Banc', 'BankCorp'),
        ('grp', 'group'),('cap','capital'),('FINL','financial'),
        ('THRU', 'Through'), ('COMM', 'Communication'),('MGMT','Management'),
        ('INVT', 'investments'),('INV', 'investments'),('investment', 'investments'),
        ('PTNR','partner'),('ADVR','advisors'),('laboratory','laboratories'),
        ('tech', 'technologies'), ('technology', 'technologies'),
        ('INDS', 'industries'), ('industry', 'industries'),
        ('COMPANIES', 'Company'), ('Mort', 'Mortgage'), ('Thr', 'Through'),
        ('Sec', 'Securities'), ('BANCORPORATION', 'BankCorp'),
        ('RESOURCE', 'Resources'), ('Holding', 'Holdings'),
        ('Security', 'Securities'), ('ENTERPRISE', 'Enterprises'),
        ('funding', 'fundings'), ('system', 'systems'), ('chem', 'chemical'),
        ('SYS', 'systems'), ('MFG', 'manufacturing'), ('Prod', 'products'),
        ('Pharma', 'Pharm'),('Pharmaceu', 'Pharm'),('Pharmaceuti', 'Pharm'),
        ('Pharmace', 'Pharm'),('Pharmaceut', 'Pharm'), ('Pharmaceutical', 'Pharm'),
        ('Product', 'products'), ('svcs','services'),('service','services'),
        ('production','productions'),
        ('&', 'and'), ('L\.P','LP'),('L\.L\.P','LLP'),('S\.A','SA'),('S\.p\.A','SPA'),
        ('u s a','usa')]

suffix = [
    'Incorporated', 'Corporation', 'LLC', 'Company', 'Limited', 'trust',
    'Company', 'Holdings', 'Holding', 'Group', 'enterprises', 'international',
    'and', 'gmbh'
]
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
    if fuzz.token_set_ratio(a,b)<94:
        return
    good_y = set()
    pos_y = dict()
    # notice that x is CRSP firms (which is more standard) and y is target names
    x,y = removal_regex.sub('',b).strip().split(),removal_regex.sub('',a).strip().split()
    if len(x)==0:
        return
    if len(x)==1:
        if (x[0] in eng) or (len(x[0])<5):
            return
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
        if not match_wx: # every word in X must have a match in Y
            return
    bad_y = set(y) - set(good_y)
    if len(bad_y)==0: # no additional words in Y means good match
        return
    for bad_wy in bad_y:
        if pos_y[bad_wy]<=len(x): # all additional words in Y must appear after X
            return
    return 1

def unpacking(main_row):
    lst = []
    main_index, main_name, main_abbr, main_disamb = main_row
    for base_index, base_name, base_abbr, base_disamb in base_.values:
        if match(main_disamb, base_disamb):
            lst.append([main_index, main_name, base_index, base_name])
    return (main_index, lst)

def main():
    with Pool() as p:
        with open('__coname__.csv', 'w', newline='') as w:
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
    args = parser.parse_args()
    filename = args.input

    base_ = pd.read_csv('stocknames_mainclass.csv').dropna()
    main_ = pd.read_csv(filename).dropna()
    # adjust abbreviations
    base_['abbr_name'] = base_[base_.columns[1]].map(abbr_adj)
    main_['abbr_name'] = main_[main_.columns[1]].map(abbr_adj)
    # disambiguation
    base_['disambiguated'] = base_[base_.columns[1]].map(name_preprocessing)
    main_['disambiguated'] = main_[main_.columns[1]].map(name_preprocessing)

    wastime = dt.now()
    print(wastime)
    main()
    print(dt.now(), (dt.now() - wastime).total_seconds() / 60)
