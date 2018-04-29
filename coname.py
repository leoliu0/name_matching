import csv
import math
import os
import re
import string
import sys
from collections import Counter, defaultdict
from datetime import datetime as dt
from unicodedata import normalize
from zipfile import ZipFile

from bs4 import BeautifulSoup as bs
from fuzzywuzzy import fuzz
from nltk.tokenize import sent_tokenize


money_str = '\$[\s,.\d]+(?:thousands|thousand|million|billion){0,1}'
mths = '(?:[Jj]anuary|[Ff]ebruary|[Mm]arch|[Aa]pril|[Mm]ay|[Jj]une|[Jj]uly|[Aa]ugust|[Ss]eptember|[Oo]ctober|[Nn]ovember|[Dd]ecember|[Jj]an|[Ff]eb|[Mm]ar|[Aa]pr|[Mm]ay|[Jj]un|[Jj]ul|[Aa]ug|[Ss]ep|[Oo]ct|[Nn]ov|[Dd]ec)'
date_str = '\d{0,2}\,{0,1}\s{0,1}'+mths+'\s+\d{0,2}\s{0,1},{0,1}\s{0,2}\d{4}'
abbr = [('CORP', 'Corporation'), ('CO', 'Company'), ('LTD', 'Limited'), ('BANCORP', 'Banking Corporation'),
        ('MOR',	'Mortgage'), ('Banc', 'Banking Corporation'), ('THRU', 'Through'), ('COMM',	'Communication'),
        ('COMPANIES', 'Company'), ('Mort', 'Mortgage'), ('Thr','Through'), ('Sec', 'Securities'),
        ('BANCORPORATION', 'Banking Corporation'), ('RESOURCE', 'Resources'), ('Holding', 'Holdings'), ('Security', 'Securities'),
        ('ENTERPRISE','Enterprises'),('funding','fundings')]
suffix = ['Corporation', 'Company', 'Limited', 'trust', 'Banking Corporation', 'Company', 'Holdings', 'Holding', 
            'Securities', 'Security', 'Group', 'ENTERPRISES', 'international', 'Bank', 'fund', 'funds']

# List of key words to identify cases
# legal, District Court, Judgment, Infring*, lawsuit, plaintiff, defendant, jury, verdict
kws = '[Ll]egal|(?:[Dd]istrict|[Ss]uperior)\s[Cc]ourt|[Ii]nfring|[Ll]awsuit|[Pp]laintiff|[Dd]efendant|[Jj]ury|[Vv]erdict|alleg(?:e|ed|ing)\s'

def space_clean(name):
    return normalize('NFKC', name).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()

def space_punc_clean(name):
    return ' '.join(re.findall('\w+', name))

def find_money(input_txt):
    return re.findall(money_str, input_txt.strip())

def abbr_adj(name):
    for string, adj_string in abbr:
        name = re.sub('\s+'+string+'(?!\w)', 
                        ' ' +adj_string, name, flags=re.IGNORECASE)
    return name                    
    
def suffix_adj(name):
    for string in suffix:
        name = re.sub('\s+'+string+'(?!\w)', 
                        '', name, flags=re.IGNORECASE)
    return name

def parse_date(date):
    date = space_punc_clean(date)
    for str_format in ['%B %d %Y','%Y%m%d','%B %Y', '%d %B %Y']:
        try:
            date = dt.strptime(date, str_format)
        except:
            continue
    if isinstance(date,dt):
        return date
    else:
        return None

def first_letter(name): # get the first letter of firm names in order to match their abbr
    name = abbr_adj(name)
    for string in suffix:
        found_suffix = re.findall('\s+'+string+'(?!\w)',name,re.IGNORECASE)
        if found_suffix:
            name = re.sub('\s+'+string+'(?!\w)', 
                        '', name, flags=re.IGNORECASE)
            break
    if found_suffix and len(name)>1: # If the length of name longer than 1, such as HP, return HP Inc
        ls = [x[0] for x in re.split('\s', name) if x]
        return ''.join(ls) + found_suffix[0]
        

def calc_adj_score(coname, party_name):
    ''' Calculate matching score between firm name of 10-K and legal party in 10-K text
    '''
    party_name = abbr_adj(party_name)
    score = fuzz.token_set_ratio(coname, party_name)  # Calculate raw score
    if score < 50:
        return score
    coname_words, party_words = coname.split(' '), party_name.split(' ')
    ''' Scenario 1 : matched words are not in order :
        If the sequence of matched words are not in order, they are false positives, return 0 unless perfect match.
    '''
    pos_x, pos_y, x_1= None, None, None
    for idx, x in enumerate(coname_words):
        for idy, y in enumerate(party_words):
            if fuzz.token_set_ratio(x, y) > 90:
                if pos_x and pos_y:
                    if x != x_1:
                        if idy+1 < pos_y:
                            if len(coname_words) == len(party_words) and len(party_words)>2:
                                return score - 20
                            else:
                                return 0

                x_1, y_1 = x, y
                pos_x, pos_y = idx+1, idy+1
    ''' Scenario 2: Unmatched words present in names
        Deduct scores when unmatched words present (if perfect match TODO)
    '''
    for L1, L2 in [(coname_words, party_words),(party_words, coname_words)]: # looping them twice for each
        for x in L1:
            for y in L2:
                if fuzz.token_set_ratio(x, y) > 90:
                    break # if the current word from party has a match, start processing next word
            else:  # if current word from party does not have a match in firm, deduct score
                score = score - 1

    x_1, y_1 = None, None
    for x in coname_words:
        y_1 = None
        for y in party_words:
            # When found the first matching word, if the previous word is not matched
            # it is almost sure they are not the same firm so that deduct 20 points (failed the matching)
            if fuzz.token_set_ratio(x, y) > 90:
                if len(x) == 1: # If the matched word has only one letter, it is not informative...
                    y_1 = y
                    continue
                if x_1 and y_1 and fuzz.token_set_ratio(x_1, y_1) < 90:
                    score = score - 20
                    return score  # Once deduct, return the score
                # If found matching word without the previous word being different,
                # Adjust the score upwards (by helping to pass the 80 points bar) and return it.
                score = score + max(0, (8 - len(x)))*4
                return score
            y_1 = y
        x_1 = x
    return score
theshold = 80
