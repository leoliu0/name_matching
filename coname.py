import csv
import math
import os
import re
import string
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime as dt
from unicodedata import normalize
from itertools import *

import pandas as pd
from fuzzywuzzy import fuzz
from nltk.tokenize import sent_tokenize

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2,s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def abbr_adj(name): # replace abbr to full
    for string, adj_string in abbr:
        name = re.sub('(?:\s|\.|\,)'+string+'(?!\w)', 
                        ' ' +adj_string, name, flags=re.IGNORECASE)
    return name.strip()                    

def suffix_adj(name): # Remove suffix
    for string in suffix:
        name = re.sub('(?:\s|\.|\,)'+string+'(?!\w)', # The string has to be after some punctuations or space.
                        '', name, flags=re.IGNORECASE)
    return name.strip()

def capital_letters(name):
    for string in suffix:
        found_suffix = re.findall('(?:\s|\.|\,)'+string+'(?!\w)',name,re.IGNORECASE)
        if found_suffix:
            name = re.sub('(?:\s|\.|\,)'+string+'(?!\w)', 
                        '', name, flags=re.IGNORECASE)
            break
    if found_suffix and len(name)>1: # If the length of name longer than 1, such as HP, return HP Inc
        ls = re.findall('[A-Z]', name)
        if len(ls)<2:
            return
        # Then we return all the captial letters with suffix.
        # We add a space between HP and Inc because sometimes, let's say it's .inc got captured, then we strip all additional spaces.
        return re.sub('\s{2,}',' ',''.join(ls+[' ']+found_suffix)).strip()

def first_letters(name): # get the first letter of firm names in order to match their abbr...
    for string in suffix:
        found_suffix = re.findall('(?:\s|\.|\,)'+string+'(?!\w)',name,re.IGNORECASE)
        if found_suffix:
            name = re.sub('(?:\s|\.|\,)'+string+'(?!\w)', 
                        '', name, flags=re.IGNORECASE)
            break
    if found_suffix and len(name)>1: # If the length of name longer than 1, such as HP, return HP Inc
        ls = [x[0] for x in re.split('\s', name) if x]
        if len(ls)<2:
            return
        return ''.join(ls) + found_suffix[0]

def remove_punc(name):
    name = name.replace('&',' ').replace('-',' ').replace('.',' ').replace(',',' ').replace('/',' ').replace("'",' ')
    return re.sub(r'[^\w\s]','',name).strip()

abbr = [('Inc','Incorporated'),('Incorp','Incorporated'), ('Assn','Association'),
        ('CORP', 'Corporation'), ('CO', 'Company'), ('LTD', 'Limited'), ('MOR', 'Mortgage'), 
        ('Banc', 'Banking Corporation'), ('THRU', 'Through'), ('COMM', 'Communication'),
        ('COMPANIES', 'Company'), ('Mort', 'Mortgage'), ('Thr','Through'), ('Sec', 'Securities'),
        ('BANCORPORATION', 'Banking Corporation'), ('RESOURCE', 'Resources'), ('Holding', 'Holdings'), ('Security', 'Securities'),
        ('ENTERPRISE','Enterprises'),('funding','fundings'),
        ('SYS','system'),('MFG','manufacturing'),('Prod','products')]
suffix = ['Incorporated', 'Corporation', 'LLC', 'Company', 'Limited', 'trust', 'Company', 'Holdings', 
        'Holding', 'Securities', 'Security', 'Group', 'ENTERPRISES', 'international', 'Bank', 'fund', 'funds','university']
suffix_regex = '|'.join(suffix)

base_ = pd.read_csv('base_name.csv').dropna()
main_ = pd.read_csv('US_assignees.csv').dropna()
# construct unique words list
name_set = dict()
for gvkey, name in base_.values:
    x = re.split('\s+',name)
    if gvkey in name_set:
        for x in name:
            name_set[gvkey].add(x.lower())
    else:
        name_set[gvkey] = set([x.lower() for x in x])
word_list = []
for v in name_set.values():
    word_list.extend(list(v))
unique_word = [word for word,n in Counter(word_list).most_common() if n==1]   

# adjust abbreviations
base_['abbr_name'] = base_[base_.columns[1]].map(abbr_adj)
main_['abbr_name'] = main_[main_.columns[1]].map(abbr_adj)

# Construct location list
with open('locations.csv','r') as f:
    location_name = [r[0] for r in csv.reader(f)]
    location_name.append('USA')
    
# construct unique words list and unique pair words list
gvkey_single_dict = dict()
gvkey_pair_dict = dict()

for gvkey, name, abbr in base_.values:
    x = re.split('\s+',remove_punc(abbr.lower()))
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

def name_preprocessing(z):
    z = z.replace('-REDH','').replace('-OLD','').replace('-NEW','')
    z = abbr_adj(z)
    z = remove_punc(z)
    z = re.sub('The ','', z, flags=re.I)
    # combining single words...
    s = re.findall('(?<!\w)\w\s\w\s\w(?!\w)',z)
    if s:
        z = re.sub('(?<!\w)\w\s\w\s\w(?!\w)',s[0].replace(' ',''),z)
    s = re.findall('(?<!\w)\w\s\w(?!\w)',z)
    if s:
        z = re.sub('(?<!\w)\w\s\w(?!\w)',s[0].replace(' ',''),z)
    words = re.split('\s+',remove_punc(z))
    return z,words

def permutation(x,y):
    x,x_words = name_preprocessing(x)
    y,y_words = name_preprocessing(y)
    if match(x, y, x_words, y_words):
        return True
    if len(x_words)>2:
        if match(''.join(x_words[:2])+ ' ' + ' '.join(x_words[2:]), y, [''.join(x_words[:2])]+ x_words[2:], y_words):
            return True
    if len(y_words)>2:
        if match(x, ''.join(y_words[:2])+ ' ' + ' '.join(y_words[2:]),x_words,[''.join(y_words[:2])]+ y_words[2:]):
            return True
    if len(x_words)>3:
        if match(''.join(x_words[:3])+ ' ' + ' '.join(x_words[3:]), y, [''.join(x_words[:3])]+ x_words[3:], y_words):
            return True
    if len(y_words)>3:
        if match(x, ''.join(y_words[:3])+ ' ' + ' '.join(y_words[3:]),x_words,[''.join(y_words[:3])]+ y_words[3:]):
            return True
def match(x,y,x_words,y_words):
    
    without_suffix_x, without_suffix_y = re.split('\s+',suffix_adj(x)), \
                                            re.split('\s+',suffix_adj(y))
    score = fuzz.token_set_ratio(x,y)
    if score < 55: 
        return #low score discarded
    first_word_x, first_word_y = x_words[0].lower(), y_words[0].lower()
    
    if first_word_x == first_word_y and first_word_x in unique_word:
        return True
    if fuzz.token_set_ratio(x,y) > 97:
        if len(without_suffix_x) == len(without_suffix_y):
            return True
        if (first_word_x in unique_word) or (first_word_y in unique_word):
            return True
    
    for y1,y2 in pairwise(without_suffix_y): #TODO, change to y when finish testing!
        if (y1.lower(),y2.lower()) in pair_word:
            for x1,x2 in pairwise(without_suffix_x):
                if fuzz.token_set_ratio(x1,y1)> 94 and fuzz.token_set_ratio(x2,y2)> 94:
                    return True
def unpacking(main_row):
    lst = []
    main_index, main_name, main_abbr = main_row
    for base_index, base_name, base_abbr in base_.values:
        if permutation(main_abbr, base_abbr):
            lst.append([main_index, main_name, base_index, base_name])
    return (main_index, lst)

wastime = dt.now()
print(wastime)
def main():
    with ProcessPoolExecutor(max_workers=6) as e:
        with open('__coname__.csv','w',newline='') as w:
            wr = csv.writer(w)
            seq = 0
            total_number = len(main_)
            for index, result in e.map(unpacking, main_.values):
                seq += 1
                print(f'{seq} out of {total_number}, {index}')
                if result:
                    for matched in result:
                        wr.writerow(matched)
if __name__ == '__main__':
    main()

print(dt.now(), (dt.now() - wastime).total_seconds()/60)
