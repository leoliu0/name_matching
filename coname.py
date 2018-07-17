import argparse
import csv
import math
import os
import re
import string
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime as dt
from itertools import *
from unicodedata import normalize

import pandas as pd
from fuzzywuzzy import fuzz
from nltk.tokenize import sent_tokenize

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("subset")
filename = parser.parse_args().input
subset = int(parser.parse_args().subset)

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2,s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def abbr_adj(name): # replace abbr to full
    for string, adj_string in abbr:
        name = re.sub('(?<!\w)'+string+'(?!\w)', 
                        ' ' +adj_string, name, flags=re.IGNORECASE)
    return name.strip()                    

def suffix_adj(name): # Remove suffix
    for string in suffix:
        name = re.sub('(?<!\w)'+string+'(?!\w)', # The string has to be after some punctuations or space.
                        '', name, flags=re.IGNORECASE)
    return name.strip()
def remove_punc(name):
    name = name.replace('&',' ').replace('-',' ').replace('.',' ').replace(',',' ').replace('/',' ').replace("'",' ')
    return re.sub(r'[^\w\s]','',name).strip()

def first_two_adj(words):
    if len(words)>2:
        return abbr_adj(''.join(words[:2])+ ' ' + ' '.join(words[2:]))

def first_three_adj(words):
    if len(words)>3:
        return abbr_adj(''.join(words[:3])+ ' ' + ' '.join(words[3:]))

def name_preprocessing(z):
    z = z.replace('-REDH','').replace('-OLD','').replace('-NEW','')
    z = abbr_adj(z)
    z = remove_punc(z)
    z = re.sub('The ','', z, flags=re.I)
    z = z.lower()
    # combining single words...
    s = re.findall('(?<!\w)\w\s\w\s\w(?!\w)',z)
    if s:
        z = re.sub('(?<!\w)\w\s\w\s\w(?!\w)',s[0].replace(' ',''),z)
    s = re.findall('(?<!\w)\w\s\w(?!\w)',z)
    if s:
        z = re.sub('(?<!\w)\w\s\w(?!\w)',s[0].replace(' ',''),z)
    words = re.split('\s+',remove_punc(z))
    without_suffix = [x for x in re.split('\s+',suffix_adj(z)) if x]
    two_ = first_two_adj(words)
    three_ = first_three_adj(words)
    if two_:
        two_words = re.split('\s+',remove_punc(two_))
        two_ws = [x for x in re.split('\s+',suffix_adj(two_)) if x]
    else:
        two_words, two_ws = None, None
    if three_:
        three_words = re.split('\s+',remove_punc(three_))
        three_ws = [x for x in re.split('\s+',suffix_adj(three_)) if x]
    else:
        three_words, three_ws = None, None

    return z,words,without_suffix,two_,two_words,two_ws,three_,three_words,three_ws

abbr = [('Inc','Incorporated'),('Incorp','Incorporated'), ('Assn','Association'),('intl', 'international'),
        ('CORP', 'Corporation'), ('CO', 'Company'), ('LTD', 'Limited'), ('MOR', 'Mortgage'), 
        ('Banc', 'Banking Corporation'), ('THRU', 'Through'), ('COMM', 'Communication'),
        ('COMPANIES', 'Company'), ('Mort', 'Mortgage'), ('Thr','Through'), ('Sec', 'Securities'),
        ('BANCORPORATION', 'Banking Corporation'), ('RESOURCE', 'Resources'), ('Holding', 'Holdings'), ('Security', 'Securities'),
        ('ENTERPRISE','Enterprises'),('funding','fundings'),('system','systems'),
        ('SYS','systems'),('MFG','manufacturing'),('Prod','products'),('Product','products')]
suffix = ['Incorporated', 'Corporation', 'LLC', 'Company', 'Limited', 'trust', 'Company', 'Holdings', 
        'Holding', 'Securities', 'Security', 'Group', 'ENTERPRISES', 'international', 'Bank', 'fund', 'funds','university']
suffix_regex = '|'.join(suffix)

base_ = pd.read_csv('base_name.csv').dropna()
main_ = pd.read_csv(filename).dropna()
# adjust abbreviations
base_['abbr_name'] = base_[base_.columns[1]].map(abbr_adj)
main_['abbr_name'] = main_[main_.columns[1]].map(abbr_adj)
# disambiguation
base_['disambiguated'] = base_[base_.columns[1]].map(name_preprocessing)
main_['disambiguated'] = main_[main_.columns[1]].map(name_preprocessing)

# construct unique words list and unique pair words list
gvkey_single_dict = dict()
gvkey_pair_dict = dict()

for gvkey, name, abbrev, disamb in base_.values:
    x = re.split('\s+',remove_punc(abbrev.lower()))
    if gvkey in gvkey_single_dict:
        for x in name:
            gvkey_single_dict[gvkey].add(x)
            gvkey_pair_dict[gvkey] = set(pairwise(x)) | gvkey_pair_dict[gvkey] # Adding the set
    else:
        gvkey_single_dict[gvkey] = set(x)
        gvkey_pair_dict[gvkey] = set(pairwise(x))
        
        
single_list = []
pair_list = []
for v in gvkey_single_dict.values():
    single_list.extend(list(v))
for v in gvkey_pair_dict.values():
    pair_list.extend(v)
unique_word = [word for word,n in Counter(single_list).most_common() if n<=2]
pair_word = [word for word,n in Counter(pair_list).most_common() if n<=2]

def permutation(x,y):
    x,x_words,without_suffix_x,two_x,two_words_x,two_ws_x,three_x,three_words_x,three_ws_x = x
    y,y_words,without_suffix_y,two_y,two_words_y,two_ws_y,three_y,three_words_y,three_ws_y = y
    if len(x)>7 and len(y)>7:
        if len(set(x)&set(y))<4:
            return
    if fuzz.token_set_ratio(x,y) < 55:
        return
    if match(x,y,x_words,y_words, without_suffix_x, without_suffix_y):
        return True
    if two_x:
        if match(two_x, y, two_words_x, y_words, two_ws_x, without_suffix_y):
            return True
        if three_x:
            if match(three_x, y, three_words_x, y_words, three_ws_x, without_suffix_y):
                return True
    if two_y:
        if match(x, two_y, x_words, two_words_y, without_suffix_x, two_ws_y):
            return True
        if three_x:
            if match(x, three_y, x_words, three_words_y, without_suffix_x, three_ws_y):
                return True    

def match(x, y, x_words, y_words, without_suffix_x, without_suffix_y):
    score = fuzz.token_set_ratio(x,y)
    if score < 70: 
        return #low score discarded
    if fuzz.token_set_ratio(' '.join(without_suffix_x),' '.join(without_suffix_y)) > 90:
        first_word_x, first_word_y = x_words[0], y_words[0]
        first_score = fuzz.ratio(first_word_x, first_word_y)
        if len(without_suffix_x) == len(without_suffix_y):
            if first_score>90:
                return True
            else:
                xyset = (set(without_suffix_x) & set(without_suffix_y))
                xyset.discard('s')
                if xyset == set(without_suffix_x):
                    return True
        if first_score>90 and (first_word_y in unique_word):
            return True
            
    if len(y_words)>1 and len(x_words)>1: # paired words must are first two of either names
        y1,y2 = y_words[:2]
        if (y1,y2) in pair_word and 'of' not in (y1,y2) and 's' not in (y1,y2):
            for x1,x2 in pairwise(x_words):
                if fuzz.ratio(x1,y1)> 90 and fuzz.ratio(x2,y2)> 90:
                    return True
        x1,x2 = x_words[:2]
        for y1,y2 in pairwise(y_words):
            if (y1,y2) in pair_word and 'of' not in (y1,y2) and 's' not in (y1,y2):
                if fuzz.ratio(x1,y1)> 90 and fuzz.ratio(x2,y2)> 90:
                    return True

def unpacking(main_row):
    lst = []
    main_index, main_name, main_abbr, main_disamb = main_row
    for base_index, base_name, base_abbr, base_disamb in base_.values:
        if permutation(main_disamb, base_disamb):
            lst.append([main_index, main_name, base_index, base_name])
    return (main_index, lst)

wastime = dt.now()
print(wastime)
def main():
    with ProcessPoolExecutor(max_workers=None) as e:
        with open('__coname__.csv','w',newline='') as w:
            wr = csv.writer(w)
            seq = 0 + subset
            total_number = len(main_)
            for index, result in e.map(unpacking, main_.loc[subset:,:].values):
                seq += 1
                print(f'{seq} out of {total_number}, {index}')
                if result:
                    for matched in result:
                        wr.writerow(matched)
if __name__ == '__main__':
    main()

print(dt.now(), (dt.now() - wastime).total_seconds()/60)
