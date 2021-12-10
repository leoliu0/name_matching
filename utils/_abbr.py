import functools
import re


def _abbr_adj(name, l):  # replace abbr to full
    for string, adj_string in l:
        if '(?' in string:
            name = re.sub(string + r'(?!\w)',
                          ' ' + adj_string,
                          name,
                          flags=re.IGNORECASE).replace('  ', ' ').strip()
        else:
            name = re.sub(r'(?<!\w)' + string + r'(?!\w)',
                          ' ' + adj_string,
                          name,
                          flags=re.IGNORECASE).replace('  ', ' ').strip()
        if adj_string.strip():
            name = re.sub(r'\b' + adj_string + r'\s+' + adj_string + r'\b',
                          adj_string, name)
    return name.replace('  ', ' ').strip().lower()


abbr1 = [
    # corporation related words and some uninformative words
    ('the', ''),
    ('and', ''),
    ('of', ''),
    ('for', ''),
    ('llc', 'llc'),
    ('ll c', 'llc'),
    (r'incorp\w+', 'inc'),
    ('CO', 'inc'),
    ('COS', 'inc'),
    ('companies', 'inc'),
    ('comapany', 'inc'),
    ('company', 'inc'),
    ('cor', 'inc'),
    ('CORP', 'inc'),
    (r'corpor\w+', 'inc'),
    ('LTD', 'inc'),
    ('limit', 'inc'),
    ('limite', 'inc'),
    ('limited', 'inc'),
    ('company incorp', 'inc'),
    ('incorp incorp', 'inc'),
    ('company limited', 'inc'),
    ('incorp limited', 'inc'),
    (r'inc\s+inc', 'inc'),
    ('Assn', 'Association'),
    ('Assoc', 'Association'),
    ('intl', 'international'),
    (r'interna\w+', 'international'),
    ('gbl', 'international'),
    ('global', 'international'),
    ('natl', 'national'),
    ('nat', 'national'),
    ('int', 'international'),
    ('&', 'and'),
    (r'L\.P', 'LP'),
    (r'L\.L\.P', 'LLP'),
    (r'S\.A', 'sa'),
    (r'(?!^) sa$', 'sa'),
    (r'(?!^) s a$', 'sa'),
    (r'(?!^) b v$', 'bv'),
    (r'S\.p\.A', 'SPA'),
    ('u s a', 'usa'),
    ('usa', 'usa'),
    ('u s', 'usa'),
    ('us', 'usa'),
    # Japanese suffix
    (r'(?!^) kk\b', 'inc'),
    (r'(?!^) gk\b', ''),
    (r'(?!^) yk\b', ''),
    (r'(?!^) gmk\b', ''),
    (r'(?!^) gsk\b', ''),
    (r'(?!^) nk\b', ''),
    (r'(?!^) tk\b', ''),
    (r'^kabus\w+ kaisha', ''),
    (r'kanus\w+ kaisha', 'inc'),
    (r'kabus\w+ kaisha', 'inc'),
    # Germany suffix
    (r'(?!^|\w)ev', ''),
    (r'(?!^|\w)rv', ''),
    (r'(?!^|\w)kgaa', ''),
    ('gmbh co', 'inc'),
    (r'(?!^|\w)ag co', 'inc'),
    (r'(?!^|\w)ag$', 'inc'),
    (r'(?!^|\w)a g$', 'inc'),
    (r'(?!^|\w)se co', ''),
    ('gmbh$', 'inc'),
    (r'\bag$', 'inc'),
    (r'(?!^|\w)se', ''),
    (r'(?!^|\w)ug', ''),
    (r'aktieng\w+', 'inc'),
    # French suffix
    (r'(?!^|\w)sep', ''),
    (r'(?!^|\w)snc', ''),
    (r'(?!^|\w)scs', ''),
    (r'(?!^|\w)sca', ''),
    (r'(?!^|\w)sci', ''),
    (r'(?!^|\w)sarl', ''),
    (r'(?!^|\w)eurl', ''),
    (r'(?!^|\w)sa', ''),
    (r'(?!^|\w)s a', ''),
    (r'(?!^|\w)scop', ''),
    (r'\bsas$', ''),
    (r'\bsasu$', ''),
    # Swedish suffix
    (r'ab$', ''),
    (r'lm$', ''),
    # Dutch suffix
    (r'n\s+v$', 'inc'),
]

abbr2 = [  # informative words
    ('univ', 'university'),
    ('bldg', 'building'),
    ('buildings', 'building'),
    ('MOR', 'Mortgage'),
    ('Banc', 'BankCorp'),
    ('bk', 'BankCorp'),
    ('bancshares ', 'bankcorp'),
    ('bankshares ', 'bankcorp'),
    ('BANC CORP', 'bankcorp'),
    ('BANCORPORATION', 'BankCorp'),
    ('bancorp', 'BankCorp'),
    ('stores', 'store'),
    ('brand', 'brands'),
    ('gen', 'general'),
    ('geneal', 'general'),
    ('Gereral', 'general'),
    ('Gereral', 'general'),
    ('generel', 'general'),
    ('solutions ', 'solution'),
    ('science', 'sciences'),
    ('sci', 'sciences'),
    ('work', 'works'),
    ('device', 'devices'),
    ('operation', 'operations'),
    ('tool', 'tools'),
    ('network', 'networks'),
    ('material', 'materials'),
    ('grp', 'group'),
    ('cap', 'capital'),
    ('FINL', 'financial'),
    ('THRU', 'Through'),
    ('COMM', 'Communication'),
    ('MGMT', 'Management'),
    ('INVT', 'investments'),
    ('INV', 'investments'),
    ('investment', 'investments'),
    ('PTNR', 'partner'),
    ('ADVR', 'advisors'),
    ('laboratory', 'laboratories'),
    ('lab', 'laboratories'),
    ('labs', 'laboratories'),
    ('ins', 'insurance'),
    ('insur', 'insurance'),
    ('insure', 'insurance'),
    ('technologies', 'tech'),
    ('technology', 'tech'),
    ('INDS', 'industries'),
    ('industry', 'industries'),
    ('industrial', 'industries'),
    ('indl', 'industries'),
    ('IND', 'industries'),
    ('res', 'research'),
    ('dev', 'development'),
    ('IP', ''),
    ('intellectual property', ''),
    ('intellectual properties', ''),
    ('intellectual', ''),
    (r'(?!^)patents', ''),
    (r'(?!^)patent', ''),
    (r'(?!^)trademark', ''),
    (r'(?!^)trademarks', ''),
    (r'(?!^)licensing', ''),
    #  ('marketing', ''),
    ('brands$', ''),
    ('property', 'properties'),
    ('Mort', 'Mortgage'),
    ('Thr', 'Through'),
    ('Sec', 'Securities'),
    ('RESOURCE', 'Resources'),
    ('Holding', 'Holdings'),
    ('Security', 'Securities'),
    ('ENTERPRISE', 'enterprises'),
    ('funding', 'fundings'),
    ('chem', 'chemical'),
    ('SYS', 'systems'),
    ('MFG', 'manufacturing'),
    ('Prod', 'products'),
    ('Pharma', 'Pharm'),
    ('Pharmaceu', 'Pharm'),
    ('Pharmaceuti', 'Pharm'),
    ('Pharmace', 'Pharm'),
    ('Pharmaceut', 'Pharm'),
    ('Pharmaceutical', 'Pharm'),
    ('Product', 'products'),
    ('svcs', 'services'),
    ('service', 'services'),
    ('production', 'productions'),
    ('saving', 'savings'),
    ('svgs', 'savings'),
    ('ln', 'loan'),
    ('electronic', 'electronics'),
    ('elect', 'electronics'),
    ('electrs', 'electronics'),
    ('elec', 'electric'),
    ('electrical', 'electric'),
    ('inst', 'institution'),
    ('motors', 'motor'),
    ('machine', 'machines'),
    ('machs', 'machines'),
    ('teleg', 'telegraph'),
    ('tel', 'telephone'),
    ('tel', 'telephone'),
    ('ry', 'railway'),
    ('american', 'america'),
    ('AMER', 'america'),
    ('AMERN', 'america'),
    ('phillip', 'philip'),
    (r'north\w* ameri\w+', 'america'),
]

# for some abbreviations, we have to hard code it.
hardcode = [
    ('hp hood', ''),
    ('hp pelzers?', ''),
    ('HP', 'HEWLETT PACKARD'),
    ('IBM', 'international business machines'),
    ('DE NEMOURS', ''),
    (r'\bE I\b', ''),
    ('NE NEMOURS', ''),
    (r'\bE I\b', ''),
    (r'\bEI\b', ''),
    (r'DU PONT', 'DU PONT'),
    (r'DU POND', 'DU PONT'),
    (r'DUPONT', 'DU PONT'),
    (r'DU PONTE', 'DU PONT'),
    ('HITACHI', 'HITACHI matchit'),
    ('exxon', 'exxon matchit'),
    ('exxonmobil', 'exxon matchit'),
    (r'\blg\b', 'lg matchit'),
    (r'\bl g\b', 'lg matchit'),
    (r'SIEM\w+S', 'SIEMENS matchit'),
    ('GTE', 'GTE matchit'),
    ('north  america philips', 'philips'),
    ('toshiba', 'toshiba matchit'),
    ('Tokyo Shibaura', 'toshiba matchit'),
    ('toyota', 'toyota matchit'),
    (r'\bhonda\b', 'honda matchit'),
    ('schlumbergers', 'schlumbergers matchit'),
    ('microsoft', 'microsoft matchit'),
    ('verizon', 'verizon matchit'),
    ('chevron', 'chevron matchit'),
    ('cisco', 'cisco matchit'),
    ('ericsson', 'ericsson matchit'),
    (r'\b3m\b', '3m matchit'),
    (r'\boracle\b', 'oracle matchit'),
    (r'\bgm\b', 'general motor'),
    (r'\bat t\b', 'at t matchit'),
    (r'\bnokia\b', 'nokia matchit'),
    ('merck', 'merck matchit'),
    (r'eastm\w+ ko\w+', 'kodak'),
    ('kodak', 'kodak matchit'),
    ('canon', 'canon matchit'),
    ('Aluminum Company of America', 'alcoa'),
    ('alcoa', 'alcoa matchit'),
    ('hoescht', 'hoechst'),
    ('Hoeschst', 'hoechst'),
    ('Hoechet', 'hoechst'),
    ('Hoechset', 'hoechst'),
    ('hoechst', 'hoechst matchit'),
    ('International Telephone and Telegraph', 'IT'),
    #  ('rockwell','rockwell matchit'),
    ('nissan', 'nissan matchit'),
    ('ford meter box', ''),
    ('ford', 'ford matchit'),
    ('xerox', 'xerox matchit'),
    ('texaco', 'texaco matchit'),
    ('volvo', 'volvo matchit'),
    ('caterpillar', 'caterpillar matchit'),
]

suffix = set([
    'inc',
    'llc',
    'company',
    'limited',
    'trust',
    'lp',
    'llp',
    'sa',
    'spa',
    'usa',
    'holdings',
    'group',
    'enterprises',
    'gmbh',
    'kk',
    'and',
    'of',
    'north american',
    # Japanese suffix
    'kk',
    'gk',
    'yk',
    'gmk',
    'gsk',
    'nk',
    'tk',
    r'Ka\w+ Kaisha',
    r'aktieng\w+'
])

abbr = abbr1 + abbr2

abbr_adj = functools.partial(_abbr_adj, l=hardcode + abbr)
abbr_suffix_adj = functools.partial(_abbr_adj, l=hardcode + abbr1)
abbr_extra_adj = functools.partial(_abbr_adj, l=hardcode + abbr2)
