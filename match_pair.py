import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

from matcher import match

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("-o","--output",default='match_result.csv')
    parser.add_argument("-c","--cpu",default=1,type=float)
    args = parser.parse_args()

    def do(row):
        if match(row[0],row[1])>0:
            return row[0],row[1]

    with ProcessPoolExecutor(int(cpu_count()*args.cpu)) as p:
        with open(args.file) as f:
            rd = csv.reader(f)
            futures = as_completed([p.submit(do,row) for row in rd])
        with open(args.output,'w') as wf:
            wr = csv.writer(wf)
            wr.writerow(['a','b'])
            for f in futures:
                res = f.result()
                if res:
                    wr.writerow(res)
