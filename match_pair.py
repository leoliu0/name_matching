#!/bin/python
import argparse
import pandas as pd
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor as Pool
from loguru import logger
from tqdm.auto import tqdm
from icecream import ic

from matcher import match, name_preprocessing

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("-o", "--output", default='match_result.csv')
    parser.add_argument("-c", "--cpu", default=1, type=float)
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()

    def do(row):
        a, b = row
        if a[1] and b[1]:
            if match(a[1], b[1]) > 0:
                return a[0], b[0]

    def process_chunk(wr, chunk, pbar):
        with Pool(int(cpu_count()*args.cpu)) as p:
            d = set([x[0] for x in chunk]) | set([x[1] for x in chunk])
            d = {x: name_preprocessing(x) for x in d}
            a = [(x[0], d[x[0]]) for x in chunk]
            b = [(x[1], d[x[1]]) for x in chunk]

            for res in p.map(do, zip(a, b), chunksize=1000):
                pbar.update(1)
                if res:
                    wr.writerow(res)

    logger.info(f"running on {int(cpu_count()*args.cpu)} of CPUs")

    num_lines = sum(1 for _ in open(args.file))
    logger.info(f"the file has {num_lines} lines")
    pbar = tqdm(total=num_lines)
    with open(args.output, 'w') as wf:
        wr = csv.writer(wf)
        wr.writerow(['a', 'b'])
        with open(args.file) as f:
            rd = csv.reader(f)
            chunk = []
            for i, row in enumerate(rd):
                chunk.append(row)
                # if i % 1_000_000 == 0:
                #     ic(len(chunk))
                if (i % 10_000_000 == 0) and (i > 0):
                    process_chunk(wr, chunk, pbar)
                    chunk = []
            if chunk:
                process_chunk(wr, chunk, pbar)
