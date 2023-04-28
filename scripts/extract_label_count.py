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
import numpy as np
import pandas as pd
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--data_path', required=True, type=str)
    args = parser.parse_args()

    df = pd.read_pickle(args.data_path)
    label = df.failureType.to_numpy().astype(int)
    label_tensor = torch.from_numpy(label)
    _, class_count = torch.unique(label_tensor, return_counts=True)
    pattern_type=['none', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring',
                  'Loc', 'Random', 'Scratch', 'Near-full']
    for pattern, cnt in zip(pattern_type, class_count):
        print(f'{pattern},{cnt}')

if __name__=='__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    main()