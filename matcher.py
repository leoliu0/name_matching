#!/usr/bin/python
import argparse
import csv
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
from _abbr import *
from _name_pre import name_preprocessing

cutoff = 50


def loc(f):
    return pathlib.Path(__file__).parent.absolute() / f


common_phrase = ['capital market']
locations = [
    x.lower().strip() for x in
    #  (open(loc('locations.csv')).readlines()) if len(x.split())>1]
    (open(loc('locations.csv')).readlines())
]
common_phrase = [' '.join(sorted(x.split())) for x in common_phrase] + \
            [' '.join(sorted(x.split())) for x in locations]

eng = set(json.load(open('words_dictionary.json')).keys())
eng = eng | set(
    [x.lower().strip() for x in (open(loc('surname.txt')).readlines())])
eng = eng | set(
    [x.lower().strip() for x in (open(loc('firstname.txt')).readlines())])
eng = eng | set(common_phrase) - set([''])

common_abbr = set([x for _, x in abbr1 if x != ''])
common_abbr12 = set([x for _, x in abbr if x != '']) | eng | suffix

__remove_suffix = re.compile(r'\b' + r'\b|\b'.join(suffix) + r'\b')


def remove_suffix(name):  # Remove suffix
    return __remove_suffix.sub('', name).strip()


def check_double(a, b):
    ''' account for double ('BALL & BALL CARBURETOR COMPANY','BALL CORP')'''
    for a1, a2 in ngrams(a, 2):
        if ratio(a1, a2) > 89:
            if a1 in suffix or a2 in suffix:
                continue
            for b1, b2 in ngrams(b, 2):
                if ratio(b1, b2) > 89:
                    if b1 in suffix or b2 in suffix:
                        continue
                    if ratio(a1, b1) <= 89:
                        return False
                    else:
                        break
            else:
                return False


location_remove = re.compile(r'\b|\b'.join([x.strip() for x in locations]))


def _has_location(name):
    #  for x in ('and', 'of','for','holdings','holding', 'group',
    #  'enterprises', 'international','global'):
    #  if not name.startswith(x):
    #  name = re.sub(r'\b'+x+r'\b','',name).strip()
    return location_remove.search(name)
    #  return name.strip()


ban_list = ('organization', 'organization', 'academy', 'university', 'agency',
            'union', '21st', 'commission', 'council', 'school', 'community',
            'institute', 'federation', 'nations', 'association', 'church',
            'society', 'league', '800', '24', 'great america')

__w_plus = re.compile('[a-z]+')
intl = ('global', 'international', 'worldwide', 'national')
too_general = ('and', 'of', 'for', 'holdings', 'holding', 'group',
               'enterprises', 'international', 'global')

na = set(['north', 'america', 'great']) | set(intl) | set(too_general)


def match(a, b):
    # part 1: high similarity scores treatment
    c, d = remove_suffix(a), remove_suffix(b)
    #  x, y = remove_meaningless(b).split(), remove_meaningless(a).split()
    x, y = b.split(), a.split()

    if len(y) - len(x) > 5:
        return -23
    _a = set(a.split()) - suffix
    _b = set(b.split()) - suffix
    if not (set(_a) - na):  # if a only has suffix left, bad ...
        return -1
    if not (set(_a) - na):  # if b only has suffix left, bad ...
        return -2

    if _has_location(a) and _has_location(b):
        if token_sort_ratio(a, b) > 95:
            return 21
        else:
            return -22

    if (token_sort_ratio(a, b) == 100) or (ratio(sorted(c), sorted(d)) == 100):
        if a[:3] == b[:3]:
            if c in too_general and d in too_general:
                return -21
            else:
                if len(x) >= 2 and len(y) >= 2:
                    if x[1][:3] == y[1][:3]:
                        return 1
                else:
                    return 1

    for w in ban_list:
        if w in a:
            return -3
    # notice that x is CRSP firms (which is more standard) and y is target names
    good_x, good_y = set(), set()
    has_bad_x = False
    pos_x, pos_y = dict(), dict()
    score_x = dict()
    for m, wx in enumerate(x, start=1):
        pos_x[wx] = m
        score_x[m] = list()
        for n, wy in enumerate(y, start=1):
            if wy not in pos_y:
                pos_y[wy] = n
            score = ratio(wx, wy)
            score_x[m].append(score)
            if wy in suffix:
                continue
            if len(x) == 1 or len(y) == 1:
                threshold = 92  # more strict if very short name
            if m == 1:  # more strict if first word in the name
                #  if len(wx)>len(wy) and (len(c.split())>1 or len(d.split())>1):
                #  threshold = 91
                #  else:
                threshold = 92  # more strict if very short name
            else:
                threshold = 89
            if ((len(x) == len(y)) and (len(x) > 3)):
                threshold = 75
            if ((score > threshold)
                    and (wx[0] == wy[0])  # first letter must match
                    and
                (wy[-1] not in '1234567890')):  # last char is not a number
                good_x.add(wx)
            #  if score>89 and wx[:5]==wy[:5] and len(wx)>7 and len(wy)>7:
            if jaro_winkler(wx, wy) > 0.92:
                good_y.add(wy)
        if (wx not in good_x) and (
                wx not in suffix):  # every word in X must have a match in Y
            has_bad_x = True
        if (wx not in good_x) and m == 1:  # First X word much match
            has_bad_x = True
        score_x[m] = max(score_x[m]) if score_x[m] else 0

    # match on high scores
    h_score = 94
    #  if ((token_sort_ratio(c,d)>h_score) or (token_sort_ratio(a,b)>h_score)):
    if ((token_sort_ratio(c, d) > h_score)):
        if has_bad_x == False:
            if a[0] == b[0]:
                return 2

    # once removing meaningless, the remaining are not uninformative words
    if len(x) == 0 or len(y) == 0:
        return -4
    if len(x) == 1:
        if (x[0] in eng) or (len(x[0]) < 5):
            return -5
    if len(x) == 2:
        if ' '.join(x[:2]) in eng:
            return -6
    if len(x) == 3:
        if ' '.join(x[:3]) in eng:
            return -13

    # part 2: low simiarity, try more cleaning ...
    if check_double(x, y) is False:
        return False
    if check_double(y, x) is False:
        return False

    if len(set(c.split()) - common_abbr -
           good_y) == 0 or len(set(d.split()) - common_abbr - good_x) == 0:
        remain_good_y = set(good_y) - suffix
        if len(remain_good_y) == 1:
            good_wy = __w_plus.findall(next(iter(remain_good_y)))
            if len(good_wy) > 0:
                if good_wy[0] in common_abbr12:
                    return -8
            else:
                return -18
            for wy in y:
                if wy in good_y or wy in suffix:
                    continue
                if pos_y[wy] <= len(x):
                    return -20
            for m in range(1, min(len(y) + 1, len(x) + 1)):
                if score_x[m] < 80:
                    return -20

        if x[0] in good_x and y[0] in good_y and x[0] not in eng and y[
                0] not in eng:
            if a[:3] == b[:3]:
                return 10

    __good_y = good_y - common_abbr - suffix
    __good_x = good_x - common_abbr - suffix

    if len(__good_y) * len(
        [w for q in __good_y for w in q if w in string.ascii_letters]) > 12:
        if ' '.join(sorted(__good_y)) not in eng:
            pos_good_y, pos_good_x = [], [
            ]  # the words in __good_y must be together
            if __good_x:
                for w in __good_y:
                    pos_good_y.append(pos_y[w])
                for w in __good_x:
                    pos_good_x.append(pos_x[w])
                if ((len(pos_good_y)
                     == (1 + max(pos_good_y) - min(pos_good_y)))
                        and (len(pos_good_x)
                             == (1 + max(pos_good_x) - min(pos_good_x)))):

                    if (y[0] in good_y
                            and ((score_x[1] > 89) and
                                 (score_x[2] > 89) and y[0] not in eng)):
                        return 4
                    else:  # first y is not matched ... match them if first word is global
                        if y[0] in intl and score_x[1] > 93:
                            if has_bad_x == True:
                                if y[1][:3] == x[0][:3]:
                                    return 5
                            else:
                                return 55

    _x = set(x) - suffix
    _y = set(y) - suffix
    if len(_x) > 1 and len(_y) > 1:
        if token_sort_ratio([x[0], x[1]], [y[0], y[1]]) > 84:
            #  if x[0]==y[0] and x[0] not in eng and has_bad_x==False:
            if x[0] == y[0] and has_bad_x == False:
                return 8
            if ((' '.join([x[0], x[1]]) not in eng)
                    and (' '.join([y[0], y[1]]) not in eng)):
                if x[0] in eng and x[1] in eng and y[0] in eng and y[1] in eng:
                    if len(_y - good_y -
                           common_abbr) > 0 and has_bad_x == True:
                        return -19
                if jaro_winkler(x[0], y[0]) > 0.97 and jaro_winkler(
                        x[1], y[1]) > 0.94:
                    if x[0] not in intl and y[1] not in intl:
                        if has_bad_x == True:
                            if len(_y - good_y) == 0:
                                return 91
                        else:
                            return 9
            else:
                return -9
        else:
            return -99

    if len(_x) == 1 and len(_y) == 1:
        if jaro_winkler(x[0], y[0]) > 0.97 and len(x[0]) > 7:
            if abs(len(x[0]) - len(y[0])) <= 1:
                if x[0] in common_abbr not in intl and y[
                        0] not in common_abbr in intl:
                    return 11
                else:
                    return -12
        if len(x[0]) >= 5 and len(y[0]) >= 5:
            if x[0][:5] == y[0][:5]:
                if x[0][-3:] == y[0][-3:]:
                    if abs(len(x[0]) - len(y[0])) <= 1:
                        return 12
            if x[0][-5:] == y[0][-5:]:
                if x[0][:3] == y[0][:3]:
                    if abs(len(x[0]) - len(y[0])) <= 1:
                        return 13

    if len(good_y) == 1:
        good_wy = __w_plus.findall(next(iter(good_y)))
        if len(good_wy) > 0:
            if good_wy[0] in common_abbr12:
                return -8
            else:
                return -18
    remaining_x = set(x) - common_abbr - suffix
    if not remaining_x:  # if nothing left in x
        return -10
    if len(remaining_x
           ) == 1:  # if after remove things, the x is a letter, bad match
        remaining_wx = next(iter(remaining_x))
        if len(remaining_wx) == 1:  # or remaining_wx in eng:
            return -11

    return -15


def match_test(x, y):
    a, b = name_preprocessing(x), name_preprocessing(y)
    if a and b:
        c, d = remove_suffix(a), remove_suffix(b)
        score = token_set_ratio(c, d)
        #  print(a, '  |||||  ', b)
        if score > cutoff:
            return match(a, b)
        #  else:
        #  print('failed at cutoff', cutoff, ' is', score)


def unpacking(main_row):
    lst = []
    main_index, main_name, main_pre, main_suffix = main_row
    for base_index, base_name, base_pre, base_suffix in base_.values:
        if token_set_ratio(main_suffix, base_suffix) > cutoff:
            if match(main_pre, base_pre) > 0:
                lst.append([
                    main_index, main_name, base_index, base_name,
                    token_sort_ratio(main_suffix, base_suffix)
                ])
    return lst


def main():
    with Pool(70) as p:
        with open(output, 'w', newline='') as w:
            wr = csv.writer(w)
            with tqdm(total=len(main_)) as pb:
                for result in p.imap(unpacking, main_.values, chunksize=100):
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
    print(wastime, 'start now ...')
    main()
    print(dt.now(), 'finished, takes',
          (dt.now() - wastime).total_seconds() / 60, 'minutes')
