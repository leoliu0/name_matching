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

def set_remove_punc(name):
    return set(re.split('\s+',re.sub(r'[^\w\s]','',name)))