import pandas as pd
import csv
import re
from fuzzywuzzy import fuzz
from concurrent.futures import ProcessPoolExecutor

main_csv = 'uspto_name.csv'
supp_csv = 'coname.csv'
abbr = [('CORP', 'Corporation'), ('CO', 'Company'), ('LTD', 'Limited'), ('BANCORP', 'Banking Corporation'),
        ('MOR',	'Mortgage'), ('Banc', 'Banking Corporation'), ('THRU', 'Through'), ('COMM',	'Communication'),
        ('COMPANIES', 'Company'), ('Mort', 'Mortgage'), ('Thr','Through'), ('Sec', 'Securities'),
        ('BANCORPORATION', 'Banking Corporation'), ('RESOURCE', 'Resources'), ('Holding', 'Holdings'), ('Security', 'Securities'),
        ('ENTERPRISE','Enterprises'),('funding','fundings')]
suffix = ['Corporation', 'Company', 'Limited', 'trust', 'Banking Corporation', 'Company', 'Holdings', 'Holding', 
            'Securities', 'Security', 'Group', 'ENTERPRISES', 'international', 'Bank', 'fund', 'funds']

def abbr_adj(name): # replace abbr to full
    for string, adj_string in abbr:
        name = re.sub('\s+'+string+'(?!\w)', 
                        ' ' +adj_string, name, flags=re.IGNORECASE)
    return name                    

def suffix_adj(name): # Remove suffix
    for string in suffix:
        name = re.sub('\s+'+string+'(?!\w)', 
                        '', name, flags=re.IGNORECASE)
    return name

main = pd.read_csv(main_csv, encoding='latin1')
supp = pd.read_csv(supp_csv, encoding='latin1')

main['abbr_name'] = main[main.columns[1]].map(abbr_adj)
supp['abbr_name'] = supp[supp.columns[1]].map(abbr_adj)
main['suff_name'] = main[main.columns[1]].map(suffix_adj)
supp['suff_name'] = supp[supp.columns[1]].map(suffix_adj)


for m_index, m_name, m_abbr, m_suff in main.values:
    with open(f'./match_pool/{m_index}.csv','w',newline='') as mp:
        wr = csv.writer(mp)

        for sup_row in supp.values:
            sup_index = sup_row[0]
            sup_name = sup_row[1]

            score = fuzz.token_set_ratio(main_name,sup_name)
            if score < 60:
                continue
            wr.writerow([main_index,main_name,sup_index,sup_name,score, 'full'])

          
