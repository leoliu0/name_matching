import argparse
import pandas as pd
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count,Pool
from loguru import logger
from tqdm.auto import tqdm

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

    def process_chunk(wr,chunk,p):
        for res in tqdm(p.imap(do,chunk,chunksize=100),total=len(chunk),desc="Chunk Progress"):
            if res:
                wr.writerow(res)

    logger.info(f"running on {int(cpu_count()*args.cpu)} of CPUs")

    num_lines = sum(1 for _ in open(args.file))
    logger.info(f"the file has {num_lines} lines")
    with Pool(int(cpu_count()*args.cpu)) as p:
        with open(args.output,'w') as wf:
            wr = csv.writer(wf)
            wr.writerow(['a','b'])
            with open(args.file) as f:
                rd = csv.reader(f)
                chunk = []
                for i,row in enumerate(tqdm(rd,total=num_lines,desc='Total Progress')): 
                    chunk.append(row)
                    if (i %10_000_000==0) and (i>0):
                        process_chunk(wr,chunk,p)
                        chunk = []
                if chunk:
                    process_chunk(wr,chunk,p)
