#!/usr/bin/python
import argparse,csv,math,os,re,string,sys,json,pkg_resources,pathlib,functools
from collections import Counter, defaultdict
from multiprocessing import Pool
from datetime import datetime as dt
from itertools import *
from unicodedata import normalize
from tqdm import tqdm

import pandas as pd

from fuzzywuzzy.fuzz import *
from nltk.tokenize import sent_tokenize

cutoff=80

abbr1 = [
        # corporation related words and some uninformative words
        ('the',''),('and',''),('of',''),('for',''),
        ('llc','llc'), ('incorp', 'inc'), ('Incorporated','inc'),
        ('CO', 'inc'), ('COS', 'inc'),('companies', 'inc'),
        ('comapany', 'inc'),('company', 'inc'),
        ('cor', 'inc'),('CORP', 'inc'),('corporation', 'inc'),
        ('coporation', 'inc'), ('corpor', 'inc'),
        ('corporat', 'inc'),('corporat', 'inc'),
        ('corporate', 'inc'),('corporatin', 'inc'),
        ('LTD', 'limited'),('limit', 'limited'),('limite', 'limited'),
        ('company incorp', 'inc'),('incorp incorp', 'inc'),
        ('company limited', 'limited'),('incorp limited', 'limited'),
        ('Assn', 'Association'),('Assoc', 'Association'),
        ('intl', 'international'),
        ('gbl','international'), ('global','international'),
        ('natl','national'), ('nat','national'),
        ('int', 'international'),
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
        (r'ab$',''),(r'lm$','')]

abbr2 = [ # informative words
        ('univ','university'),
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
        ('intellectual', ''),(r'(?!^)patents',''),(r'(?!^)patent',''),
        (r'(?!^)trademark',''),(r'(?!^)trademarks',''),
        ('marketing',''),
        ('property', 'properties'), ('Mort', 'Mortgage'), ('Thr', 'Through'),
        ('Sec', 'Securities'), ('BANCORPORATION', 'BankCorp'),
        ('RESOURCE', 'Resources'), ('Holding', 'Holdings'),
        ('Security', 'Securities'), ('ENTERPRISE', 'enterprises'),
        ('funding', 'fundings'), ('chem', 'chemical'),
        ('SYS', 'systems'), ('MFG', 'manufacturing'), ('Prod', 'products'),
        ('Pharma', 'Pharm'),('Pharmaceu', 'Pharm'),('Pharmaceuti', 'Pharm'),
        ('Pharmace', 'Pharm'),('Pharmaceut', 'Pharm'), ('Pharmaceutical', 'Pharm'),
        ('Product', 'products'), ('svcs','services'),('service','services'),
        ('production','productions'),('saving','savings'),('svgs','savings'),
        ('ln','loan'),('electronic','electronics'),('elect','electronics'),
        ('electrs','electronics'), ('inst','institution'),
        ('motors','motor'),
        ('machine','machines'),('machs','machines'),('teleg','telegraph'),
        ('tel','telephone'),('tel','telephone'),('ry','railway'),
        ('american','america'),('AMER','america'),('AMERN','america'),
        ('phillip','philip')
        ]

abbr = abbr1 + abbr2

common_abbr = set([x for _,x in abbr1 if x !=''])
# for some abbreviations, we have to hard code it.
hardcode = [('HP', 'HEWLETT PACKARD'),('IBM','international business machines'),
            (r'DU PONT$','DU PONT EI DE NEMOURS'),
            ('HITACHI','HITACHI matchit'),('exxon','exxon matchit'),
            ('SIEMENS','SIEMENS matchit'),('GTE','GTE matchit'),
            ('north  america philips','philips')]


suffix = set(['inc', 'llc', 'company', 'limited', 'trust','lp','llp','sa','spa',
          'usa', 'holdings', 'group', 'enterprises', 'international', 'gmbh','kk'
          'and','of','north american',
            # Japanese suffix
            'kk','gk','yk','gmk','gsk','nk','tk',
              ]
             )
suffix_regex = '|'.join(suffix)

def loc(f):
    return pathlib.Path(__file__).parent.absolute()/f

common_phrase = ['capital market']
locations = [x.lower().strip() for x in (open(loc('locations.csv')).readlines())]
common_phrase = [' '.join(sorted(x.split())) for x in common_phrase] + \
            [' '.join(sorted(x.split())) for x in locations]

eng = set(json.load(open('words_dictionary.json')).keys())
eng = eng | set([x.lower().strip() for x in (open(loc('surname.txt')).readlines())])
eng = eng | set([x.lower().strip() for x in (open(loc('firstname.txt')).readlines())])
eng = eng | set(common_phrase) - set([''])


def _abbr_adj(name,l):  # replace abbr to full
    for string, adj_string in l:
        name = re.sub('(?<!\w)' + string + '(?!\w)',
                      ' ' + adj_string,
                      name,
                      flags=re.IGNORECASE)
    return name.strip().lower()

abbr_adj = functools.partial(_abbr_adj,l=abbr+hardcode)
abbr_suffix_adj = functools.partial(_abbr_adj,l=abbr1+hardcode)
abbr_extra_adj = functools.partial(_abbr_adj,l=abbr2+hardcode)

def disamb(name):  # Remove suffix
    name = abbr_adj(name)
    for string in suffix:
        if not name.startswith(string):
            name = re.sub(
            r'\b' + string +
            r'\b',  # The string has to be after some punctuations or space.
            '', name, flags=re.IGNORECASE)
    return name.strip()


def name_preprocessing(z):
    z = z.lower()
    z = z.replace('-redh', '').replace('-old', '').replace('-new', '')
    z = re.sub(r'(?=\w+)our\b',r'or',z)
    z = re.sub(r'(?=\w+)tt\b',r't',z)
    #  z = re.sub(r'(?=(\w+))([a-zA-Z])\2?',r'\2',z)
    z = re.sub(r'(?=\w+)er\b',r'ers',z) # to not match e.g. glove vs glover
    z = z.replace('`','').replace('& company','').replace('& companies','')
    z = re.sub('\bthe\b', '', z)
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
    z = abbr_suffix_adj(z)
    return z.strip().lower()

def check_double(a,b):
    ''' account for double ('BALL & BALL CARBURETOR COMPANY','BALL CORP')'''
    for a1,a2 in combinations(a,2):
        if ratio(a1,a2)>89:
            for b1,b2 in combinations(b,2):
                if ratio(b1,b2)>89:
                    if ratio(a1,b1)<=89:
                        return False
                    else:
                        break
            else:
                return False

location_remove = re.compile(r'\b|\b'.join([x.strip() for x in locations]))
def remove_meaningless(name):
    for x in ['and', 'of','for','holdings','holding', 'group',
              'enterprises', 'international','global']:
        if not name.startswith(x):
            name = re.sub(r'\b'+x+r'\b','',name).strip()
    name = location_remove.sub('',name)
    return name.strip()

ban_list = ('organization','organization','academy','university','commission',
            'department','council','school','community','institute')

def match(a,b,c,d):
    # part 1: high similarity scores treatment
    x,y = remove_meaningless(b).split(),remove_meaningless(a).split()
    if not (set(a.split()) - suffix): # if a only has suffix left, bad ...
        return -1
    if not (set(b.split()) - suffix): # if b only has suffix left, bad ...
        return -2
    for w in ban_list:
        if w in a:
            return -3

    if ((token_sort_ratio(c,d)==100) or
            (token_sort_ratio(a,b)==100) or (ratio(sorted(a),sorted(b))==100)):
        return 1
    # notice that x is CRSP firms (which is more standard) and y is target names
    good_x, good_y = set(),set()
    has_bad_x = False
    pos_x, pos_y = dict(),dict()
    for m,wx in enumerate(x,start=1):
        pos_x[wx] = m
        for n,wy in enumerate(y,start=1):
            if wy not in pos_y:
                pos_y[wy] = n
            if len(x)==1 or len(y)==1:
                threshold = 92 # more strict if very short name
            else:
                threshold = 89
            if ((len(x) == len(y)) and (len(x)>3)):
                #  or (token_sort_ratio(c,d)>91)
                        #  or (token_sort_ratio(a,b)>91)
                        #  or (ratio(sorted(a),sorted(b))>98)):
                # good match on entire name lower the threshold ...
                threshold = 75
            if ((ratio(wx,wy)>threshold) and (wx[0]==wy[0]) # first letter must match
                    and (wy[-1] not in '1234567890')): # last char is not a number
                good_y.add(wy)
                good_x.add(wx)
        if (wx not in good_x) and (wx not in suffix): # every word in X must have a match in Y
            has_bad_x = True
        if (wx not in good_x) and m==1: # First X word much match
            has_bad_x = True

    # match on high scores
    h_score= 94
    if ((token_sort_ratio(c,d)>h_score) or (token_sort_ratio(a,b)>h_score)):
        if has_bad_x == False:
            return 2

    # once removing meaningless, the remaining are not uninformative words
    if len(x)==0:
        return -4
    if len(x)==1:
        if (x[0] in eng) or (len(x[0])<5):
            return -5
    if len(x)==2:
        if ' '.join(x[:2]) in eng:
            return -12
    if len(x)==3:
        if ' '.join(x[:3]) in eng:
            return -13

    # part 2: low simiarity, try more cleaning ...


    if check_double(x,y) is False:
        return False
    if check_double(y,x) is False:
        return False

    if x[0] not in good_x and x[0] not in ('global', 'international'):
        return -14
    __good_y= good_y - common_abbr - suffix
    __good_x= good_x - common_abbr - suffix
    if len(__good_y)>1:
        pos_y_nums = [pos_y[good_wy] for good_wy in __good_y]
        pos_x_nums = [pos_x[good_wx] for good_wx in __good_x]
        if min(pos_y_nums)>1 and (min(pos_x_nums)>1):
            return -6
        if (((max(pos_y_nums) - min(pos_y_nums)) > (len(__good_y)-1)) or
                ((max(pos_x_nums) - min(pos_x_nums)) > (len(__good_x)-1))):
            return -7

    #  if len(__good_y - eng)>0:
    if len(__good_y)*len([w for q in __good_y for w in q if w in string.ascii_letters])>20:
        if ' '.join(sorted(__good_y)) not in eng:
            if y[0] in good_y:
                return good_y
            else: # first y is not matched ...
                if y[0] in ('global','international'):
                    return 5
    # match fail if has bad X and did not pass the multi-phrase test above ...
    if has_bad_x is True:
        return -8

    # check unique words in bad_y after removing suffix
    # (always keep first word as it is informative such as 'international' ...)
    bad_y = ((set(y) - suffix) | set([y[0]])) - good_y
    if len(bad_y)==0: # no additional words except for suffix in Y means good match
        return 7
    for bad_wy in bad_y:
        if pos_y[bad_wy]<=len(x): # all additional words in Y must appear after X
            return -9

    remaining_x = set(x) - common_abbr - suffix
    if not remaining_x: # if nothing left in x
        return -10
    if len(remaining_x)==1: # if after remove things, the x is a letter, bad match
        remaining_wx = next(iter(remaining_x))
        if len(remaining_wx)==1: # or remaining_wx in eng:
            return -11
    return 6

def match_test(a,b):
    c,d = name_preprocessing(a),name_preprocessing(b)
    a,b = disamb(c),disamb(d)
    score = token_set_ratio(a,b)
    print(a,'  |||||  ',b)
    if score>cutoff:
        return match(a,b,c,d)
    else:
        print('failed at cutoff',cutoff,' is',score)

def unpacking(main_row):
    lst = []
    main_index, main_name, main_pre, main_disamb= main_row
    for base_index, base_name, base_pre, base_disamb in base_.values:
        if token_set_ratio(main_disamb,base_disamb)>cutoff:
            if match(main_disamb, base_disamb, main_pre, base_pre)>0:
                lst.append([main_index, main_name, base_index, base_name,
                            token_sort_ratio(main_disamb, base_disamb)])
    return lst

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
    base_['pre_proc'] = base_[base_.columns[1]].map(name_preprocessing)
    main_['pre_proc'] = main_[main_.columns[1]].map(name_preprocessing)
    base_['disamb'] = base_['pre_proc'].map(disamb)
    main_['disamb'] = main_['pre_proc'].map(disamb)

    wastime = dt.now()
    print(wastime,'start now ...')
    main()
    print(dt.now(), 'finished, takes',
          (dt.now() - wastime).total_seconds() / 60, 'minutes')
