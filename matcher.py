#!/usr/bin/python
import argparse
import csv
import functools
import json
import math
import os
import pathlib
import re
import string
import sys
from collections import Counter, defaultdict
from datetime import datetime as dt
from itertools import *
from multiprocessing import Pool
from unicodedata import normalize

import pandas as pd
import pkg_resources
from fuzzywuzzy.fuzz import *
from Levenshtein import jaro_winkler
from nltk import ngrams
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

cutoff=50

abbr1 = [
        # corporation related words and some uninformative words
        ('the',''),('and',''),('of',''),('for',''),
        ('llc','llc'), ('incorp\w+', 'inc'),
        ('CO', 'inc'), ('COS', 'inc'),('companies', 'inc'),
        ('comapany', 'inc'),('company', 'inc'),
        ('cor', 'inc'),('CORP', 'inc'),('corpor\w+', 'inc'),
        ('LTD', 'inc'),('limit', 'inc'),('limite', 'inc'),
        ('company incorp', 'inc'),('incorp incorp', 'inc'),
        ('company limited', 'inc'),('incorp limited', 'inc'),
        ('inc\s+inc','inc'),
        ('Assn', 'Association'),('Assoc', 'Association'),
        ('intl', 'international'),('interna\w+', 'international'),
        ('gbl','international'), ('global','international'),
        ('natl','national'), ('nat','national'),
        ('int', 'international'),
        ('&', 'and'), ('L\.P','LP'),('L\.L\.P','LLP'),('S\.A','SA'),('S\.p\.A','SPA'),
        ('u s a','usa'), ('usa','usa'), ('u s','usa'), ('us','usa'),
        # Japanese suffix
        (r'(?!^|\w)kk',''),(r'(?!^|\w)gk',''),(r'(?!^|\w)yk',''),(r'(?!^|\w)gmk',''),
        (r'(?!^|\w)gsk',''),(r'(?!^|\w)nk',''),(r'(?!^|\w)tk',''), (r'kanus\w+ kaisha',''),
        # Germany suffix
        (r'(?!^|\w)ev',''),(r'(?!^|\w)rv',''),(r'(?!^|\w)kgaa',''),('gmbh co',''),
        (r'(?!^|\w)ag co',''),(r'(?!^|\w)se co',''),
        ('gmbh',''),(r'\bag',''),(r'(?!^|\w)se',''),(r'(?!^|\w)ug',''), ('aktieng\w+',''),
        # French suffix
        (r'(?!^|\w)sep',''),(r'(?!^|\w)snc',''),(r'(?!^|\w)scs',''),(r'(?!^|\w)sca',''),
        (r'(?!^|\w)sci',''),(r'(?!^|\w)sarl',''), (r'(?!^|\w)eurl',''),(r'(?!^|\w)sa',''),
        (r'(?!^|\w)s a',''),
        (r'(?!^|\w)scop',''),(r'\bsas$',''),(r'\bsasu$',''),
        # Swedish suffix
        (r'ab$',''),(r'lm$','')]

abbr2 = [ # informative words
        ('univ','university'),('bldg','building'),('buildings','building'),
        ('MOR', 'Mortgage'), ('Banc', 'BankCorp'),('bk', 'BankCorp'),
        ('bancshares ', 'bankcorp'),('bankshares ', 'bankcorp'),
        ('BANC CORP','bankcorp'), ('BANCORPORATION', 'BankCorp'), ('bancorp' , 'BankCorp'),
        ('stores','store'),('brand','brands'),('gen','general'),('geneal','general'),
        ('Gereral','general'),('Gereral','general'),('generel','general'),
        ('solutions ', 'solution'),('science','sciences'),('sci','sciences'),
        ('work', 'works'),('device','devices'),('operation','operations'),
        ('tool', 'tools'),('network','networks'),('material','materials'),
        ('grp', 'group'),('cap','capital'),('FINL','financial'),
        ('THRU', 'Through'), ('COMM', 'Communication'),('MGMT','Management'),
        ('INVT', 'investments'),('INV', 'investments'),('investment', 'investments'),
        ('PTNR','partner'),('ADVR','advisors'),
        ('laboratory','laboratories'),('lab','laboratories'),('labs','laboratories'),
        ('ins','insurance'),('insur','insurance'),('insure','insurance'),
        ('technologies', 'tech' ), ('technology', 'tech'),
        ('INDS', 'industries'), ('industry', 'industries'), ('indl', 'industries'),
        ('IND', 'industries'),('res','research'),('dev','development'),
        ('IP', ''), ('intellectual property', ''),('intellectual properties', ''),
        ('intellectual', ''),(r'(?!^)patents',''),(r'(?!^)patent',''),
        (r'(?!^)trademark',''),(r'(?!^)trademarks',''),(r'(?!^)licensing',''),
        ('marketing',''),('brands$',''),
        ('property', 'properties'), ('Mort', 'Mortgage'), ('Thr', 'Through'),
        ('Sec', 'Securities'),
        ('RESOURCE', 'Resources'), ('Holding', 'Holdings'),
        ('Security', 'Securities'), ('ENTERPRISE', 'enterprises'),
        ('funding', 'fundings'), ('chem', 'chemical'),
        ('SYS', 'systems'), ('MFG', 'manufacturing'), ('Prod', 'products'),
        ('Pharma', 'Pharm'),('Pharmaceu', 'Pharm'),('Pharmaceuti', 'Pharm'),
        ('Pharmace', 'Pharm'),('Pharmaceut', 'Pharm'), ('Pharmaceutical', 'Pharm'),
        ('Product', 'products'), ('svcs','services'),('service','services'),
        ('production','productions'),('saving','savings'),('svgs','savings'),
        ('ln','loan'),('electronic','electronics'),('elect','electronics'),
        ('electrs','electronics'),('elec','electric'),('electrical','electric'),
        ('inst','institution'),
        ('motors','motor'),
        ('machine','machines'),('machs','machines'),('teleg','telegraph'),
        ('tel','telephone'),('tel','telephone'),('ry','railway'),
        ('american','america'),('AMER','america'),('AMERN','america'),
        ('phillip','philip'),(r'north\w* ameri\w+','america'),
        ]

abbr = abbr1 + abbr2

# for some abbreviations, we have to hard code it.
hardcode = [('HP', 'HEWLETT PACKARD'),('IBM','international business machines'),
            ('DE NEMOURS',''),(r'\bE I\b',''), ('NE NEMOURS',''),(r'\bE I\b',''),
            (r'\bEI\b',''),
            (r'DU PONT','DU PONT'),(r'DU POND','DU PONT'),
            (r'DUPONT','DU PONT'),(r'DU PONTE','DU PONT'),
            ('HITACHI','HITACHI matchit'),('exxon','exxon matchit'),
            ('SIEM\w+S','SIEMENS matchit'),('GTE','GTE matchit'),
            ('north  america philips','philips'),
            ('toshiba','toshiba matchit'), ('Tokyo Shibaura','toshiba matchit'),
            ('merck','merck matchit'),('eastm\w+ ko\w+','kodak'),
            ('kodak','kodak matchit'),('canon','canon matchit'),
            ('Aluminum Company of America', 'alcoa'),
            ('alcoa','alcoa matchit'),
            ('hoescht','hoechst'), ('Hoeschst', 'hoechst'),
            ('Hoechet','hoechst'), ('Hoechset', 'hoechst'),
            ('hoechst','hoechst matchit'),
            ('International Telephone and Telegraph','IT'),
            ('rockwell','rockwell matchit'),
            ('nissan','nissan matchit'),
            ('ford','ford matchit'),
            ('xerox','xerox matchit'),
            ('texaco','texaco matchit'),
            ('volvo','volvo matchit'),
            ('caterpillar','caterpillar matchit'),
           ]

suffix = set(['inc', 'llc', 'company', 'limited', 'trust','lp','llp','sa','spa',
          'usa', 'holdings', 'group', 'enterprises', 'gmbh','kk',
          'and','of','north american',
            # Japanese suffix
            'kk','gk','yk','gmk','gsk','nk','tk','Ka\w+ Kaisha','aktieng\w+'
              ])

def loc(f):
    return pathlib.Path(__file__).parent.absolute()/f

common_phrase = ['capital market']
locations = [x.lower().strip() for x in
             #  (open(loc('locations.csv')).readlines()) if len(x.split())>1]
             (open(loc('locations.csv')).readlines())]
common_phrase = [' '.join(sorted(x.split())) for x in common_phrase] + \
            [' '.join(sorted(x.split())) for x in locations]

eng = set(json.load(open('words_dictionary.json')).keys())
eng = eng | set([x.lower().strip() for x in (open(loc('surname.txt')).readlines())])
eng = eng | set([x.lower().strip() for x in (open(loc('firstname.txt')).readlines())])
eng = eng | set(common_phrase) - set([''])

common_abbr = set([x for _,x in abbr1 if x !=''])
common_abbr12 = set([x for _,x in abbr if x !='']) | eng | suffix


def _abbr_adj(name,l):  # replace abbr to full
    for string, adj_string in l:
        if '(?' in string:
            name = re.sub(string + r'(?!\w)',
                      ' ' + adj_string, name,
            flags=re.IGNORECASE).replace('  ',' ').strip()
        else:
            name = re.sub(r'(?<!\w)' + string + r'(?!\w)',
                      ' ' + adj_string, name,
            flags=re.IGNORECASE).replace('  ',' ').strip()
        if adj_string.strip():
            name = re.sub(r'\b'+adj_string+'\s+'+adj_string+r'\b',adj_string,name)
    return name.replace('  ',' ').strip().lower()

abbr_adj = functools.partial(_abbr_adj,l=hardcode+abbr)
abbr_suffix_adj = functools.partial(_abbr_adj,l=hardcode+abbr1)
abbr_extra_adj = functools.partial(_abbr_adj,l=hardcode+abbr2)

__remove_suffix = re.compile(r'\b' +r'\b|\b'.join(suffix) + r'\b')
def remove_suffix(name):  # Remove suffix
    return __remove_suffix.sub('',name).strip()

names = set([x.strip() for x in (open(loc('names_decode.csv')).readlines())])
names = names|set([x[0] for x in hardcode])
# names from https://github.com/philipperemy/name-dataset
__w_3_plus = re.compile('\w{3,}')

def name_preprocessing(z):
    z = z.lower()
    z = z.replace('-redh', '').replace('-old', '').replace('-new', '')
    z = z.split('-pre')[0].split('-adr')[0
            ].split('division of')[-1].split('known as')[-1
            ].split('-consolidated')[0]
    z = re.sub(r'(?=\w+)our\b',r'or',z)
    z = re.sub(r'(?=\w+)tt\b',r't',z)
    #  z = re.sub(r'(?=(\w+))([a-zA-Z])\2?',r'\2',z)
    z = re.sub(r'(?=\w+)er\b',r'ers',z) # to not match e.g. glove vs glover
    z = z.replace('`','').replace('& company','').replace('& companies','')
    z = re.sub(r'\bco\.? inc\b',r'inc',z)
    z = re.sub(r'\bco\.? ltd\b',r'inc',z)
    z = re.sub(r'\bthe\b', '', z)
    z = ' '.join(re.findall(r'[\w\d]+',z))
    # combining single words...
    a = ''.join(re.findall(r'\b\w\s\b',z))
    if a:
        b = a.replace(' ','')
        z = z.replace(a,b+' ')

    #TODO: refactor the code to a function
    for string, adj_string in [('i',''),('ii',''),('iii',''),#('iv',''),('v',''),
                         #  ('vi',''),('vii',''),('viii',''),('ix',''),('x','')
                              ]:
        z = re.sub('(?<!\w)' + string + '(?!\w)',
                      ' ' + adj_string, z, flags=re.IGNORECASE)
    #  z = abbr_suffix_adj(z)
    # remove people's name
    if len(z.split())>1:
        for w in __w_3_plus.findall(z):
            if w not in names:
                break
    else:
        return
    z = abbr_adj(z)
    return z.strip().lower()

def check_double(a,b):
    ''' account for double ('BALL & BALL CARBURETOR COMPANY','BALL CORP')'''
    for a1,a2 in ngrams(a,2):
        if ratio(a1,a2)>89:
            if a1 in suffix or a2 in suffix:
                continue
            for b1,b2 in ngrams(b,2):
                if ratio(b1,b2)>89:
                    if b1 in suffix or b2 in suffix:
                        continue
                    if ratio(a1,b1)<=89:
                        return False
                    else:
                        break
            else:
                return False

location_remove = re.compile(r'\b|\b'.join([x.strip() for x in locations]))

def remove_meaningless(name):
    #  for x in ('and', 'of','for','holdings','holding', 'group',
              #  'enterprises', 'international','global'):
        #  if not name.startswith(x):
            #  name = re.sub(r'\b'+x+r'\b','',name).strip()
    name = location_remove.sub('',name)
    return name.strip()

ban_list = ('organization','organization','academy','university','commission',
            'council','school','community','institute','church', 'league',
           '800')

__w_plus = re.compile('[a-z]+')
intl = ('global', 'international','worldwide','national')
too_general = ('and', 'of','for','holdings','holding', 'group', 'enterprises', 'international','global')
def match(a,b):
    # part 1: high similarity scores treatment
    c,d = remove_suffix(a),remove_suffix(b)
    x,y = remove_meaningless(b).split(),remove_meaningless(a).split()
    if len(y) - len(x) >5:
        return -1
    _a = set(a.split()) - suffix
    _b = set(b.split()) - suffix
    if not set(_a): # if a only has suffix left, bad ...
        return -1
    if not set(_b): # if b only has suffix left, bad ...
        return -2
    for w in ban_list:
        if w in a:
            return -3

    if (token_sort_ratio(a,b)==100) or (ratio(sorted(c),sorted(d))==100):
        if a[:3]==b[:3]:
            if c in too_general and d in too_general:
                return -21
            else:
                return 1
    # notice that x is CRSP firms (which is more standard) and y is target names
    good_x, good_y = set(),set()
    has_bad_x = False
    pos_x, pos_y = dict(),dict()
    score_x = dict()
    for m,wx in enumerate(x,start=1):
        pos_x[wx] = m
        score_x[m] = list()
        for n,wy in enumerate(y,start=1):
            if wy not in pos_y:
                pos_y[wy] = n
            score = ratio(wx,wy)
            score_x[m].append(score)
            if wy in suffix:
                continue
            if len(x)==1 or len(y)==1:
                threshold = 92 # more strict if very short name
            if m==1:# more strict if first word in the name
                #  if len(wx)>len(wy) and (len(c.split())>1 or len(d.split())>1):
                    #  threshold = 91
                #  else:
                threshold = 92 # more strict if very short name
            else:
                threshold = 89
            if ((len(x) == len(y)) and (len(x)>3)):
                threshold = 75
            if ((score>threshold) and (wx[0]==wy[0]) # first letter must match
                    and (wy[-1] not in '1234567890')): # last char is not a number
                good_x.add(wx)
            #  if score>89 and wx[:5]==wy[:5] and len(wx)>7 and len(wy)>7:
            if jaro_winkler(wx,wy)>0.92:
                good_y.add(wy)
        if (wx not in good_x) and (wx not in suffix): # every word in X must have a match in Y
            has_bad_x = True
        if (wx not in good_x) and m==1: # First X word much match
            has_bad_x = True
        score_x[m] = max(score_x[m]) if score_x[m] else 0

    # match on high scores
    h_score= 94
    #  if ((token_sort_ratio(c,d)>h_score) or (token_sort_ratio(a,b)>h_score)):
    if ((token_sort_ratio(c,d)>h_score)):
        if has_bad_x == False:
            return 2

    # once removing meaningless, the remaining are not uninformative words
    if len(x)==0 or len(y)==0:
        return -4
    if len(x)==1:
        if (x[0] in eng) or (len(x[0])<5):
            return -5
    if len(x)==2:
        if ' '.join(x[:2]) in eng:
            return -6
    if len(x)==3:
        if ' '.join(x[:3]) in eng:
            return -13

    # part 2: low simiarity, try more cleaning ...
    if check_double(x,y) is False:
        return False
    if check_double(y,x) is False:
        return False

    if len(set(c.split()) - common_abbr - good_y)==0 or len(set(d.split()) - common_abbr - good_x)==0:
        remain_good_y = set(good_y) - suffix
        if len(remain_good_y)==1:
            good_wy = __w_plus.findall(next(iter(remain_good_y)))
            if len(good_wy)>0:
                if good_wy[0] in common_abbr12:
                    return -8
            else:
                return -18
            for wy in y:
                if wy in good_y or wy in suffix:
                    continue
                if pos_y[wy]<=len(x):
                    return -20
            for m in range(1,min(len(y)+1,len(x)+1)):
                if score_x[m]<80:
                    return -20

        if x[0] in good_x and y[0] in good_y and x[0] not in eng and y[0] not in eng:
            return 10

    __good_y= good_y - common_abbr - suffix
    __good_x= good_x - common_abbr - suffix

    if len(__good_y)*len([w for q in __good_y for w in q if w in
                          string.ascii_letters])>12:
        if ' '.join(sorted(__good_y)) not in eng:
            pos_good_y,pos_good_x= [],[] # the words in __good_y must be together
            if __good_x:
                for w in __good_y:
                    pos_good_y.append(pos_y[w])
                for w in __good_x:
                    pos_good_x.append(pos_x[w])
                if ((len(pos_good_y) == (1 + max(pos_good_y) - min(pos_good_y))) and
                        (len(pos_good_x) == (1 + max(pos_good_x) - min(pos_good_x)))):

                    if (y[0] in good_y and ((score_x[1]>89) and y[0] not in eng)):
                        return 4
                    else: # first y is not matched ... match them if first word is global
                        if y[0] in intl and score_x[1]>89:
                            return 5

    if x[0] not in good_x and x[0] not in intl:
        if len(x[0])>=5 and len(y[0])>=5:
            if len(x[0])==len(y[0]):
                if x[0][:5] != y[0][:5]:
                    if x[0][-5:] == y[0][-5:]:
                        if jaro_winkler(x[0],y[0])>0.93:
                            if x[0][0]==y[0][0]:
                                return 14

    _x= set(x) - suffix
    _y= set(y) - suffix
    if len(_x)>1 and len(_y)>1:
        if token_sort_ratio([x[0],x[1]], [y[0],y[1]])>84:
            #  if x[0]==y[0] and x[0] not in eng and has_bad_x==False:
            if x[0]==y[0] and has_bad_x==False:
                return 8
            if ((' '.join([x[0],x[1]]) not in eng)
                    and (' '.join([y[0],y[1]]) not in eng)):
                if x[0] in eng and x[1] in eng and y[0] in eng and y[1] in eng:
                    if len(_y - good_y - common_abbr)>0 and has_bad_x==True:
                        return -19
                if jaro_winkler(x[0],y[0])>0.97 and jaro_winkler(x[1],y[1])>0.93:
                    if x[0] not in intl and y[1] not in intl:
                        return 9
            else:
                return -9
        else:
            return -99

    if len(_x)==1 and len(_y)==1:
        if jaro_winkler(x[0],y[0])>0.97 and len(x[0])>7:
            if abs(len(x[0]) - len(y[0]))<=1:
                if x[0] in common_abbr not in intl and y[0] not in common_abbr in intl:
                    return 11
                else:
                    return -12
        if len(x[0])>=5 and len(y[0])>=5:
            if x[0][:5] == y[0][:5]:
                if x[0][-3:] == y[0][-3:]:
                    if abs(len(x[0]) - len(y[0]))<=1:
                        return 12
            if x[0][-5:] == y[0][-5:]:
                if x[0][:2] == y[0][:2]:
                    if abs(len(x[0]) - len(y[0]))<=1:
                        return 13

    if len(good_y)==1:
        if __w_plus.findall(next(iter(good_y)))[0] in common_abbr12:
            return -8
    remaining_x = set(x) - common_abbr - suffix
    if not remaining_x: # if nothing left in x
        return -10
    if len(remaining_x)==1: # if after remove things, the x is a letter, bad match
        remaining_wx = next(iter(remaining_x))
        if len(remaining_wx)==1: # or remaining_wx in eng:
            return -11

    return -15

def match_test(x,y):
    a,b = name_preprocessing(x),name_preprocessing(y)
    if a and b:
        c,d = remove_suffix(a),remove_suffix(b)
        score = token_set_ratio(c,d)
        print(a,'  |||||  ',b)
        if score>cutoff:
            return match(a,b)
        else:
            print('failed at cutoff',cutoff,' is',score)

def unpacking(main_row):
    lst = []
    main_index, main_name, main_pre, main_suffix= main_row
    for base_index, base_name, base_pre, base_suffix in base_.values:
        if token_set_ratio(main_suffix,base_suffix)>cutoff:
            if match(main_pre, base_pre)>0:
                lst.append([main_index, main_name, base_index, base_name,
                            token_sort_ratio(main_suffix, base_suffix)])
    return lst

def main():
    with Pool(80) as p:
        with open(output, 'w', newline='') as w:
            wr = csv.writer(w)
            with tqdm(total=len(main_)) as pb:
                for result in p.imap(unpacking, main_.values, chunksize=10):
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
    base_['pre_proc'] = base_[base_.columns[1]].map(name_preprocessing)
    main_['pre_proc'] = main_[main_.columns[1]].map(name_preprocessing)
    base_ = base_.dropna()
    main_ = main_.dropna()
    base_['nosuffix'] = base_['pre_proc'].map(remove_suffix)
    main_['nosuffix'] = main_['pre_proc'].map(remove_suffix)
    base_ = base_.dropna()
    main_ = main_.dropna()

    wastime = dt.now()
    print(wastime,'start now ...')
    main()
    print(dt.now(), 'finished, takes',
          (dt.now() - wastime).total_seconds() / 60, 'minutes')
