'''-------------------------------------------------------------------------
This file is part of Semi-CL-WMPD, a Python library for wafer map pattern
detection using semi-supervised constrastive learning with domain-specific
transformation.

Copyright (C) 2020-2021 Hanbin Hu <hanbinhu@ucsb.edu>
                        Peng Li <lip@ucsb.edu>
              University of California, Santa Barbara
-------------------------------------------------------------------------'''

import warnings
import argparse
import os
from pathlib import Path

def extract_result(file):
    with open(file, 'r') as f:
        lines=f.readlines()
        test_perf_lines = lines[-5:]
        test_perf_strip_lines = [line.strip() for line in test_perf_lines]
        if not all(['Best' in line for line in test_perf_strip_lines]):
            raise RuntimeError
        test_perf_str = [line.split()[-1] for line in test_perf_strip_lines]
    return test_perf_str

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-rd', '--resdir', required=True, type=str)
    args = parser.parse_args()

    if not os.path.isdir(args.resdir):
        raise RuntimeError(f"Path {args.resdir} doesn't exist.")
    logfiles = [os.path.join(args.resdir, f) for f in os.listdir(args.resdir) if f.endswith('.log')]
    results = []
    for file in logfiles:
        test_name = Path(file).stem
        try:
            result_file = extract_result(file)
        except:
            continue

        results.append([test_name]+result_file)
    
    results.sort(key=lambda l: l[0])

    with open('res.csv', 'w') as f:
        for result in results:
            f.write(','.join(result))
            f.write('\n')

if __name__=='__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    main()