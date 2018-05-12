import csv
import math
import os
import re
import string
import sys
from collections import Counter, defaultdict
from datetime import datetime as dt
from unicodedata import normalize
import pandas as pd
from fuzzywuzzy import fuzz
from nltk.tokenize import sent_tokenize
from concurrent.futures import ProcessPoolExecutor

abbr = [('Inc','Incorporated'),('Incorp','Incorporated'), ('Assn','Association'),
        ('CORP', 'Corporation'), ('CO', 'Company'), ('LTD', 'Limited'), ('BANCORP', 'Banking Corporation'),
        ('MOR',	'Mortgage'), ('Banc', 'Banking Corporation'), ('THRU', 'Through'), ('COMM',	'Communication'),
        ('COMPANIES', 'Company'), ('Mort', 'Mortgage'), ('Thr','Through'), ('Sec', 'Securities'),
        ('BANCORPORATION', 'Banking Corporation'), ('RESOURCE', 'Resources'), ('Holding', 'Holdings'), ('Security', 'Securities'),
        ('ENTERPRISE','Enterprises'),('funding','fundings')]
suffix = ['Incorporated', 'Corporation', 'LLC', 'Company', 'Limited', 'trust', 'Banking Corporation', 'Company', 'Holdings', 
        'Holding', 'Securities', 'Security', 'Group', 'ENTERPRISES', 'international', 'Bank', 'fund', 'funds']

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
def not_has_adpos(word):
    if (' a ' in word) or ('In ' in word) or (' in ' in word) or ( ' against ' in word):
        return False
    else:
        return True

def match(x, y):
    ''' matching between two names and write to file...
    '''
    x, y = abbr_adj(x), abbr_adj(y)
    x_words, y_words = re.split('\s+',remove_punc(x)), re.split('\s+',remove_punc(y))
    ''' Scenario 1 : the neighbour of matched words must be matched, unless there is no neighbours:
    '''
    matching_pos_x,matching_pos_y = None, None
    threshold = 94
    identified = False
    # create dict to index sequences
    x_seq = {n:x for n,x in enumerate(x_words)} 
    y_seq = {n:y for n,y in enumerate(y_words)}
    for id_x, x_word in x_seq.items():
        for id_y, y_word in y_seq.items():
            if fuzz.token_set_ratio(x_word, y_word) > threshold and len(x_word)>1: # allow one mistake per five letters
                identified = True
                if x_word == x_words[0] and y_word == y_words[-1] and len(x_words)>1 and not_has_adpos(y_word):
                    return
                if y_word == y_words[0] and x_word == x_words[-1] and len(y_words)>1 and not_has_adpos(x_word):
                    return
                if (id_x-1 in x_seq) and (id_y-1 in y_seq): 
                    if fuzz.token_set_ratio(x_seq[id_x-1], y_seq[id_y-1]) < threshold: # The previous words must match
                        if not (matching_pos_x and matching_pos_y): # if no previous matched words, exclude...
                            return
                        if matching_pos_x - id_x != 1 and matching_pos_y - id_y != 1:
                            return # if previous word does not match, and last match is not close, exclude it ...
                if (id_x+1 in x_seq) and (id_y+1 in y_seq):
                    if fuzz.token_set_ratio(x_seq[id_x+1], y_seq[id_y+1]) < threshold:
                        if (id_x+2 in x_seq) and (id_y+2 in y_seq):
                            if fuzz.token_set_ratio(x_seq[id_x+2], y_seq[id_y+2]) > threshold:
                                continue # if next word does not match, only the next next word match the test can continue ...
                            else:
                                return
                        else:
                            return                    
                matching_pos_x,matching_pos_y = id_x, id_y
    if identified:
        return True
def unpacking(mp_file):
    lst = []
    with open(f'.\\match_pool\\{mp_file}') as f:
        rd = csv.reader(f)
        for m_index, m_name, s_index, s_name, m_adj, s_adj, type_ in rd:
            if type_ == 'full':
                if match(m_name, s_name):
                    lst.append([m_index, m_name, s_index, s_name, type_])
    return lst


mp = os.listdir('.\\match_pool\\')
wastime = dt.now()
print(wastime)
def main():
    with ProcessPoolExecutor() as e:
        with open('__coname__.csv','w',newline='') as w:
            wr = csv.writer(w)
            for mp_file, result in zip(mp, e.map(unpacking, mp,chunksize=1000)):
                print(mp_file)
                if result is not None:
                    for matched in result:
                        wr.writerow(matched)
if __name__ == '__main__':
    main()

print(dt.now(), (dt.now() - wastime).total_seconds()/60)