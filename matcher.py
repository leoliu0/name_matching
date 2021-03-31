#!/usr/bin/python
import argparse,csv,math,os,re,string,sys,json,pkg_resources,pathlib,functools
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

abbr1 = [
        # corporation related words and some uninformative words
        ('the',''),('and',''),('of',''),('for',''),
        ('llc','llc'), ('Inc', 'incorp'), ('Incorporated','incorp'),
        ('CO', 'company'), ('COS', 'Company'),('companies', 'Company'),
        ('comapany', 'company'),
        ('cor', 'incorp'),('CORP', 'incorp'),('corporation', 'incorp'),
        ('coporation', 'incorp'), ('corpor', 'incorp'),
        ('corporat', 'incorp'),('corporat', 'incorp'),
        ('corporate', 'incorp'),('corporatin', 'incorp'),
        ('LTD', 'limited'),('limit', 'limited'),('limite', 'limited'),
        ('company incorp', 'incorp'),('incorp incorp', 'incorp'),
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
        ('Security', 'Securities'), ('ENTERPRISE', 'Enterprises'),
        ('funding', 'fundings'), ('networks', 'systems'), ('chem', 'chemical'),
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


suffix = set(['incorp', 'llc', 'company', 'limited', 'trust','lp','llp','sa','spa',
          'usa', 'holdings', 'group', 'enterprises', 'international', 'gmbh','kk'
          'and','of','north american',
            # Japanese suffix
            'kk','gk','yk','gmk','gsk','nk','tk',
              ]
             )
suffix_regex = '|'.join(suffix)

def loc(f):
    return os.path.join(pathlib.Path(__file__).parent.absolute(),f)

eng = set(json.load(open('words_dictionary.json')).keys())
eng = eng | set([x.lower().strip() for x in (open(loc('surname.txt')).readlines())])
eng = eng | set([x.lower().strip() for x in (open(loc('firstname.txt')).readlines())])

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
    for string, adj_string in [('i',''),('ii',''),('iii',''),('iv',''),('v',''),
                         ('vi',''),('vii',''),('viii',''),('ix',''),('x','')]:
        z = re.sub('(?<!\w)' + string + '(?!\w)',
                      ' ' + adj_string, z, flags=re.IGNORECASE)
    z = abbr_suffix_adj(z)
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
def remove_meaningless(name):
    for x in ['and', 'of','for','holdings','holding', 'group',
              'enterprises', 'international','global']:
        if not name.startswith(x):
            name = re.sub(r'\b'+x+r'\b','',name).strip()
    return name.strip()

ban_list = ('organization','organization','academy','university','commission')

def match(a,b,c,d):
    # part 1: high similarity scores treatment
    if len(c.split()) > 2 or len(c.split()) > 2: # long names more forgiving
        if fuzz.token_sort_ratio(c,d)>91:
            return True
    else:
        if fuzz.token_sort_ratio(c,d)>96:
            return True
    if not (set(a.split()) - suffix): # if a only has suffix left, bad ...
        return False
    if not (set(b.split()) - suffix): # if b only has suffix left, bad ...
        return False
    if (fuzz.token_sort_ratio(a,b)>96) or (fuzz.ratio(sorted(a),sorted(b))==100):
        return True
    for w in ban_list:
        if w in a:
            return False
    # part 2: low simiarity, try more cleaning ...
    good_y = set()
    has_bad_x = False
    pos_y = dict()
    # notice that x is CRSP firms (which is more standard) and y is target names
    x,y = remove_meaningless(b).split(),remove_meaningless(a).split()
    if len(x)==0:
        return False
    if len(x)==1:
        if (x[0] in eng) or (len(x[0])<5):
            return False

    for m,wx in enumerate(x,start=1):
        match_wx = False
        for n,wy in enumerate(y,start=1):
            if wy not in pos_y:
                pos_y[wy] = n
            if n==1:
                threshold = 92 # more strict if first word (letter 5 vs 6)
            else:
                threshold = 89
            if (len(x) == len(y)) and (len(x)>4):
                threshold = 75
            if ((fuzz.ratio(wx,wy)>threshold) and (wx[0]==wy[0]) # first letter must match
                    and (wy[-1] not in '1234567890')): # last char is not a number
                match_wx = True
                good_y.add(wy)
        if not match_wx and (wx not in suffix): # every word in X must have a match in Y
            has_bad_x = True
        if not match_wx and m==1: # First X word much match no matter what
            has_bad_x = True

    if check_double(x,y) is False:
        return False
    if check_double(y,x) is False:
        return False

    __good_y= good_y - common_abbr - suffix
    #  if len(__good_y - eng)>0:
    if len(__good_y)*len([w for q in __good_y for w in q if w in string.ascii_letters])>20:
        return True
    # match fail if has bad X and did not pass the multi-phrase test above ...
    if has_bad_x is True:
        return False

    # check unique words in bad_y after removing suffix
    # (always keep first word as it is informative such as 'international' ...)
    bad_y = ((set(y) - suffix) | set([y[0]])) - good_y
    if len(bad_y)==0: # no additional words except for suffix in Y means good match
        return 1
    for bad_wy in bad_y:
        if pos_y[bad_wy]<=len(x): # all additional words in Y must appear after X
            return False

    remaining_x = set(x) - common_abbr - suffix
    if not remaining_x: # if nothing left in x
        return False
    if len(remaining_x)==1: # if after remove things, the x is a letter, bad match
        remaining_wx = next(iter(remaining_x))
        if len(remaining_wx)==1: # or remaining_wx in eng:
            return False
    return True

def match_test(a,b):
    c,d = name_preprocessing(a),name_preprocessing(b)
    a,b = disamb(c),disamb(d)
    score = fuzz.token_set_ratio(a,b)
    print(a,'  |||||  ',b)
    if score>cutoff:
        return match(a,b,c,d)
    else:
        print('failed at cutoff',cutoff,' is',score)

def unpacking(main_row):
    lst = []
    main_index, main_name, main_pre, main_disamb= main_row
    for base_index, base_name, base_pre, base_disamb in base_.values:
        if fuzz.token_set_ratio(main_disamb,base_disamb)>cutoff:
            if match(main_disamb, base_disamb, main_pre, base_pre):
                lst.append([main_index, main_name, base_index, base_name,
                            fuzz.token_set_ratio(main_disamb, base_disamb)])
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
