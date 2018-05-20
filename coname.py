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

import pandas as pd
from fuzzywuzzy import fuzz
from nltk.tokenize import sent_tokenize

def abbr_adj(name): # replace abbr to full
    for string, adj_string in abbr:
        name = re.sub('(?:\s|\.|\,)'+string+'(?!\w)', 
                        ' ' +adj_string, name, flags=re.IGNORECASE)
    return name                    

def suffix_adj(name): # Remove suffix
    for string in suffix:
        name = re.sub('(?:\s|\.|\,)'+string+'(?!\w)', # The string has to be after some punctuations or space.
                        '', name, flags=re.IGNORECASE)
    return name

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
        return re.sub('\s{2,}',' ',''.join(ls+[' ']+found_suffix)) 

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
    return re.sub(r'[^\w\s]','',name)

abbr = [('Inc','Incorporated'),('Incorp','Incorporated'), ('Assn','Association'),
        ('CORP', 'Corporation'), ('CO', 'Company'), ('LTD', 'Limited'), ('BANCORP', 'Banking Corporation'),
        ('MOR',	'Mortgage'), ('Banc', 'Banking Corporation'), ('THRU', 'Through'), ('COMM',	'Communication'),
        ('COMPANIES', 'Company'), ('Mort', 'Mortgage'), ('Thr','Through'), ('Sec', 'Securities'),
        ('BANCORPORATION', 'Banking Corporation'), ('RESOURCE', 'Resources'), ('Holding', 'Holdings'), ('Security', 'Securities'),
        ('ENTERPRISE','Enterprises'),('funding','fundings')]
suffix = ['Incorporated', 'Corporation', 'LLC', 'Company', 'Limited', 'trust', 'Banking Corporation', 'Company', 'Holdings', 
        'Holding', 'Securities', 'Security', 'Group', 'ENTERPRISES', 'international', 'Bank', 'fund', 'funds','university']
suffix_regex = '|'.join(suffix)

base_ = pd.read_csv('base_name.csv').dropna()
main_ = pd.read_csv('uspto.csv').dropna()
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

def not_has_adpos(word):
    # return False/None means it does have adpos and it leads keep of the matching pair.
    if re.search('\s+(in|against|name|to|of)(?!\w)',word, re.I) or \
        re.search('^(in|against|name|to|of)(?!\w)',word, re.I):
        if re.search(suffix_regex+' of', word, re.I):
            return True
    else:        
        return True
def match(x, y):
    ''' matching between two names and write to file...
    '''
    if fuzz.token_set_ratio(x,y) < 55:
        return
    x, y = abbr_adj(x), abbr_adj(y)
    x_words, y_words = re.split('\s+',remove_punc(x)), re.split('\s+',remove_punc(y))
    ''' Scenario 1 : the neighbour of matched words must be matched, unless there is no neighbours:
    '''
    matching_pos_x, matching_pos_y = None, None
    threshold = 94
    identified = False
    # create dict to index sequences
    x_seq = {n:x for n,x in enumerate(x_words)} 
    y_seq = {n:y for n,y in enumerate(y_words)}
    match_count = 0
    for id_x, x_word in x_seq.items():
        for id_y, y_word in y_seq.items():
            if fuzz.token_set_ratio(x_word, y_word) > threshold:
                match_count += 1
                identified = True
                if match_count == 1:
                    if id_x != 0 and id_y != 0:
                        return # If the first match, it must be the first letter of either x or y

                    if len(x_words) == 1 or len(y_words) == 1:
                        if y_word.lower() not in unique_word:# if the only word in the name is not unique, exclude
                            return
                    elif (len(x_words) == id_x + 1) or (len(y_words) == id_y + 1):# if head matches tail, exclude
                        return

                    if id_x > 0:
                        if not_has_adpos(' '.join(x_words[:id_x])):
                            return # The unmatched words before the matched must contain adpos (it may be extraction error)                   
                    if id_y > 0:
                        if not_has_adpos(' '.join(y_words[:id_y])):
                            return # The unmatched words before the matched must contain adpos (it may be extraction error)
                if (id_x+1 in x_seq) and (id_y+1 in y_seq):
                    if fuzz.token_set_ratio(x_seq[id_x+1], y_seq[id_y+1]) < threshold:
                        remaining_x = ' '.join(x_words[id_x+1:]).lower()
                        remaining_y = ' '.join(y_words[id_y+1:]).lower()
                        for location in location_name:
                            if (location.lower() in remaining_x) or (location.lower() in remaining_y):
                                break
                        else:
                            return
                        for remain in y_words[id_y+2:]:
                            if fuzz.token_set_ratio(x_words[id_x+1], remain) > threshold:
                                break
                        else:
                            for remain in x_words[id_x+2:]: # if could not find any matching in remaining words, exclude ...
                                if fuzz.token_set_ratio(y_words[id_y+1], remain) > threshold:
                                    break
                            else:
                                return
    if identified:
        return True
def unpacking(main_row):
    lst = []
    main_index, main_name, main_abbr = main_row
    for base_index, base_name, base_abbr in base_.values:
        if match(main_abbr, base_abbr):
            lst.append([main_index, main_name, base_index, base_name])
    return (main_index, lst)

wastime = dt.now()
print(wastime)
def main():
    with ProcessPoolExecutor() as e:
        with open('__coname__.csv','w',newline='') as w:
            wr = csv.writer(w)
            for index, result in e.map(unpacking, main_.values):
                print(index)
                if result:
                    for matched in result:
                        wr.writerow(matched)
if __name__ == '__main__':
    main()

print(dt.now(), (dt.now() - wastime).total_seconds()/60)
