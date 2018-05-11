import pandas as pd
import csv
import re
from fuzzywuzzy import fuzz
from concurrent.futures import *
from datetime import datetime
from itertools import product

pd.options.display.max_rows = 100
pd.set_option('display.width', 100)

main_csv = 'uspto_name.csv'
supp_csv = 'comp_name.csv'
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

main_df = pd.read_csv(main_csv, encoding='latin1')
supp_df = pd.read_csv(supp_csv, encoding='latin1')
main_df['abbr_name'] = main_df[main_df.columns[1]].map(abbr_adj)
supp_df['abbr_name'] = supp_df[supp_df.columns[1]].map(abbr_adj)
main_df['suff_name'] = main_df[main_df.columns[2]].map(suffix_adj)
supp_df['suff_name'] = supp_df[supp_df.columns[2]].map(suffix_adj)
main_df['capital_name'] = main_df[main_df.columns[2]].map(capital_letters)
supp_df['capital_name'] = supp_df[supp_df.columns[2]].map(capital_letters)
main_df['first_letter_name'] = main_df[main_df.columns[2]].map(first_letters)
supp_df['first_letter_name'] = supp_df[supp_df.columns[2]].map(first_letters)
main_df['splitted_set'] = main_df[main_df.columns[3]].map(set_remove_punc)
supp_df['splitted_set'] = supp_df[supp_df.columns[3]].map(set_remove_punc)

def _func(main_row):
    m_index, m_name, m_abbr, m_suff, m_capital, m_first_letter, m_split = main_row
    with open(f'./match_pool/{m_index}.csv','w',newline='') as mp:
        wr = csv.writer(mp)
        for s_index, s_name, s_abbr, s_suff, s_capital, s_first_letter, s_split in supp_df.values:
            
            # write those adjusted abbr and with raw score > 60
            if m_split & s_split:
                wr.writerow([m_index,m_name,s_index,s_name,m_suff,s_suff,'full'])
            else:
                for m,s in product(m_split, s_split):
                    if fuzz.token_set_ratio(m,s) >= 80:
                            wr.writerow([m_index,m_name,s_index,s_name,m_suff,s_suff,'full'])
                            break

            # Then capital letter approach to capture name abbr
            score = fuzz.token_set_ratio(m_capital,s_capital)
            if score > 90: # This threshold allow matching 'AB' and 'ABC' and better pair...
                wr.writerow([m_index,m_name,s_index,s_name,m_capital,s_capital,'capital'])
            
            # The first letter approach to capture those wrong cases from capturing capital letter.
            score = fuzz.token_set_ratio(m_first_letter,s_first_letter)
            if score > 90: # This threshold allow matching 'AB' and 'ABC' and better pair...
                wr.writerow([m_index,m_name,s_index,s_name,m_first_letter,s_first_letter,'first'])
    return m_index

def main():
    with ProcessPoolExecutor() as executor:
        for index in executor.map(_func, main_df.values,chunksize=1000):
            print(index)

wasnow = datetime.now()
print('started at ', wasnow)
if __name__ == '__main__':
    main()
    
print('used ',(datetime.now() - wasnow ).total_seconds()/60)

# wasnow = datetime.now()
# for m_index, m_name, m_abbr, m_suff, m_capital, m_first_letter, m_split in main_df.values[:100]:
#     with open(f'./match_pool/{m_index}.csv','w',newline='') as mp:
#         wr = csv.writer(mp)

#         for s_index, s_name, s_abbr, s_suff, s_capital, s_first_letter, s_split in supp_df.values:
            
#             # write those adjusted abbr and with raw score > 60
#             if m_split & s_split:
#                 wr.writerow([m_index,m_name,s_index,s_name,m_suff,s_suff,'full'])
#             else:
#                 for m,s in product(m_split, s_split):
#                     if fuzz.token_set_ratio(m,s) >= 80:
#                             wr.writerow([m_index,m_name,s_index,s_name,m_suff,s_suff,'full'])
#                             break

#             # Then capital letter approach to capture name abbr
#             score = fuzz.token_set_ratio(m_capital,s_capital)
#             if score > 90: # This threshold allow matching 'AB' and 'ABC' and better pair...
#                 wr.writerow([m_index,m_name,s_index,s_name,m_capital,s_capital,'capital'])
            
#             # The first letter approach to capture those wrong cases from capturing capital letter.
#             score = fuzz.token_set_ratio(m_first_letter,s_first_letter)
#             if score > 90: # This threshold allow matching 'AB' and 'ABC' and better pair...
#                 wr.writerow([m_index,m_name,s_index,s_name,m_first_letter,s_first_letter,'first'])
# print((datetime.now() - wasnow ).total_seconds()/60)