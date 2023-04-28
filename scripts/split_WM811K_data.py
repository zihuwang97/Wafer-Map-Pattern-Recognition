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
import sys, os, inspect
import argparse
from typing import List, Iterable
import pandas as pd
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from datasets import get_WM811K_dataframe

def save_dataframe(df: pd.DataFrame, filename: str):
    df.to_pickle(filename)

def load_dataframe(filename: str) -> pd.DataFrame:
    return pd.read_pickle(filename)

def required_percentage(vmin: int, vmax: int):
    class RequiredPercentage(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            ratios = sorted(list(set([int(v) for v in values])))
            if not all([vmin<=r<=vmax for r in ratios]):
                msg = f'argument "{self.dest}" requires a list of values between {vmin} and {vmax}'
                raise argparse.ArgumentTypeError(msg)
            setattr(args, self.dest, ratios)
    return RequiredPercentage

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--percentages', required=True, nargs='+',
                        action=required_percentage(1, 99))
    args = parser.parse_args()

    # Get dataframe for training
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    train_data_file = os.path.join(data_dir, 'train.pkl')
    if not os.path.isfile(train_data_file):
        train_ratio = 0.9
        data_file = os.path.join(data_dir, 'LSWMD.pkl')
        dataframes = get_WM811K_dataframe(data_file, train_ratio=train_ratio)
        save_dataframe(dataframes['train'], train_data_file)
        save_dataframe(dataframes['val'], os.path.join(data_dir, 'test.pkl'))
        df_train = dataframes['train']
    else:
        df_train = load_dataframe(train_data_file)

    # Split dataframe
    print(f'Complete distribution: {list(df_train.failureType.value_counts())}')
    already_sampled_ratio = 0
    df_remain = df_train
    df_sample = df_train.sample(0)
    for ratio in args.percentages:
        df = df_remain.sample(frac=(ratio-already_sampled_ratio)/(100.0-already_sampled_ratio))
        df_remain = df_remain.drop(df.index)
        df_sample = pd.concat([df_sample, df])
        print(f'Split ratio = {ratio:2d}/100: {list(df_sample.failureType.value_counts())}')
        split_data_file = os.path.join(data_dir, f'train_{ratio}.pkl')
        save_dataframe(df, split_data_file)
        already_sampled_ratio = ratio

if __name__=='__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    main()