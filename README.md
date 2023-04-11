# Fuzzy Name-matching

Name matching algorithm for company to CRSP permnos (US. public firms)

Please use matcher.py as it reflects a new wave of disambiguation efforts. 

To help with this project, add name pair that should be matched to the white list and bad matches in the black list. I will periodically look at those problematic ones and further improve this algo


## how to use this
git clone this repo. If you do not know how to use git. Download this repo as zip and unzip it.

Make sure you have Python 3.6+ installed and install any missing packages it tells you to install

```pip install pandas rapidfuzz nltk loguru```

Place your name file in to the unzipped folder. The name file has to be in the following csv format:

```csv
1,apple inc
2,microsoft corp
3,whatever inc
...
```

where you have an index column and the name colume to match.

After having this file, run

```bash
./matcher.py name.csv 
```

This will result in __result__.csv file that contains the matched results like the following:

```
1,apple inc, 12345, APPLE INC, 100
...
```

The result columns are: your_index, your_name, permno, name_in_CRSP, matching_score

All results are only those the program thinks they are good matches. A high score only indicate they are textually similar, but the matched one can have low score, indicating good matches but texually unsimilar. It is only used for further processing if needed. You can safely ignore it.

I pull the latest CRSP stocknames once for a while, if you want to match on your own file, pass -b option and your file have to have the same specs as the stocknames:

```bash
./matcher.py name.csv -b your_supplied_file.csv 
```
