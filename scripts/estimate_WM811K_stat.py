'''-------------------------------------------------------------------------
This file is part of Semi-CL-WMPD, a Python library for wafer map pattern
detection using semi-supervised constrastive learning with domain-specific
transformation.

Copyright (C) 2020-2021 Hanbin Hu <hanbinhu@ucsb.edu>
                        Peng Li <lip@ucsb.edu>
              University of California, Santa Barbara
-------------------------------------------------------------------------'''

import warnings
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms

def read_train_wm():
    wafer_map_df = pd.read_pickle('./data/LSWMD.pkl')
    wafer_map_df = wafer_map_df.drop(['waferIndex'], axis = 1)
    wafer_map_df = wafer_map_df.drop(['lotName'], axis = 1)
    wafer_map_df = wafer_map_df.drop(['dieSize'], axis = 1)
    mapping_type={'Center':0,'Donut':1,'Edge-Loc':2,'Edge-Ring':3,'Loc':4,'Random':5,'Scratch':6,'Near-full':7,'none':8}
    mapping_traintest={'Training':0,'Test|':1}
    wafer_map_df=wafer_map_df.replace({'failureType':mapping_type, 'trianTestLabel':mapping_traintest})
    labelled_wm_df = wafer_map_df[(wafer_map_df['failureType']>=0) &
                                  (wafer_map_df['failureType']<=8)]
    train_wm_df = labelled_wm_df[labelled_wm_df['trianTestLabel']==0]
    train_wm_df = train_wm_df.drop(['trianTestLabel'], axis = 1)
    np_wm=train_wm_df.waferMap.to_numpy()
    norm_wm=[Image.fromarray(np.uint8(wm.astype(np.single)/2*255)) for wm in np_wm]
    return norm_wm

def estimate_mean_std(norm_wm, transform):
    mean=0
    for wm in norm_wm:
        wm_tensor=transform(wm)
        mean += wm_tensor.mean()
    mean /= len(norm_wm)
    std=0
    for wm in norm_wm:
        wm_tensor=transform(wm)-mean
        std += (wm_tensor**2).mean() 
    std /= len(norm_wm)
    std = torch.sqrt(std)
    return mean, std

def main():
    parser = ArgumentParser()
    parser.add_argument('-s', '--size', type=int)
    args = parser.parse_args()

    norm_wm = read_train_wm()
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.size, args.size))
    ])
    mean, std = estimate_mean_std(norm_wm, transform)

    print(f'Mean: {mean:.6f}')
    print(f'Std:  {std:.6f}')

if __name__=='__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    main()