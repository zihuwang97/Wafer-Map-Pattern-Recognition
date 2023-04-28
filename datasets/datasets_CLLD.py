from typing import Tuple, List, Dict, Union, Sequence
import numpy as np
import pandas as pd
from PIL import Image
from omegaconf import DictConfig

import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler, Sampler
from torchvision import transforms
from utils import get_logger, get_hydra_path
from transform import RandomTwistTransform, RandomWMNoise, RandomRotateTransform

class WM811KDataset(Dataset):
    def __init__(self, df:pd.DataFrame, transform:nn.Module=None, num_views:int=1, data_idx:bool=False, include_raw=False, include_no_crop=False):
        self.id = df.index.to_numpy()
        np_wm = df.waferMap.to_numpy()
        # normalization accounted
        self.wm = [Image.fromarray(np.uint8(wm.astype(np.single)/2*255)) for wm in np_wm]
        self.label = df.failureType.to_numpy().astype(np.int)
        self._transform = transform
        self._view = num_views
        self._data_idx = data_idx
        assert self._view >= 1

        self.include_raw = include_raw
        self.include_no_crop = include_no_crop
        self.basic_trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((128, 128))])
        
        self.no_crop_trans = transforms.Compose([
                transforms.ToTensor(),
                RandomTwistTransform(3),
                RandomWMNoise(0.05),
                transforms.Resize((128, 128)),
                transforms.Normalize((0.4463,), (0.2564,)),
                transforms.RandomHorizontalFlip(),
                RandomRotateTransform([0, 90, 180, 270]),
                ])
       

    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx:int):
        ori_wm = self.wm[idx]
        if self._transform:
            wm = self._transform(ori_wm)
        if self._view > 1:
            views = [wm]
            for _ in range(self._view-1):
                views.append(self._transform(ori_wm))
            wm = torch.cat(views,dim=0)
        if self.include_no_crop:
            views = [wm]
            views.append(self.no_crop_trans(ori_wm))
            wm = torch.cat(views,dim=0)
        if self.include_raw:
            views = [wm]
            views.append(self.basic_trans(ori_wm))
            wm = torch.cat(views,dim=0)
        if not self._data_idx:
            return self.id[idx], wm, self.label[idx]
        else:
            return self.id[idx], wm, self.label[idx], idx
    
def get_WM811K_dataframe(data_path:str,
                         train_ratio:float) -> Dict[str, pd.DataFrame]:
    wm_df = pd.read_pickle(data_path)
    wm_df = wm_df.drop(['waferIndex'], axis = 1)
    wm_df = wm_df.drop(['lotName'], axis = 1)
    wm_df = wm_df.drop(['dieSize'], axis = 1)
    mapping_type={'none':0,
                  'Center':1,
                  'Donut':2,
                  'Edge-Loc':3,
                  'Edge-Ring':4,
                  'Loc':5,
                  'Random':6,
                  'Scratch':7,
                  'Near-full':8}
    mapping_traintest={'Training':0,'Test':1}
    wm_df=wm_df.replace({'failureType':mapping_type,
                         'trianTestLabel':mapping_traintest})
    labelled_wm_df = wm_df[(wm_df['failureType']>=0) &
                           (wm_df['failureType']<=8)]

    full_wm_df = labelled_wm_df[labelled_wm_df['trianTestLabel']==0]
    full_wm_df = full_wm_df.drop(['trianTestLabel'], axis=1)

    train_wm_df = full_wm_df.sample(frac=train_ratio)
    val_wm_df = full_wm_df.drop(train_wm_df.index)

    test_wm_df = labelled_wm_df[labelled_wm_df['trianTestLabel']==1]
    test_wm_df = test_wm_df.drop(['trianTestLabel'], axis=1)
    return {'train': train_wm_df,
            'val': val_wm_df,
            'test': test_wm_df}

def get_sampler(cfg:DictConfig,
                label:torch.Tensor) -> Union[Dict[str,Sampler[int]],Dict[str,bool]]:
    if cfg.enabled:
        logger = get_logger()
        target_list = torch.from_numpy(label)
        _, class_count = torch.unique(target_list, return_counts=True)
        assert len(class_count) == 9
        float_class_count = class_count.float()
        class_weights = torch.mean(float_class_count)/float_class_count
        if hasattr(cfg, 'major_class_weight'):
            major_class_weight = cfg.major_class_weight
            assert 0 <= major_class_weight <= 1
            minor_class_cnt = len(class_count)-1
            minor_class_weight = (1-major_class_weight)/minor_class_cnt
            class_weights *= torch.tensor([major_class_weight]+[minor_class_weight]*minor_class_cnt)
        logger.info(f'Use weighted batch sampling with {class_count}')
        logger.info(f'Class weights: {class_weights}')
        class_weights_all = class_weights[target_list]
        weighted_sampler = WeightedRandomSampler(weights=class_weights_all,
                                                 num_samples=len(class_weights_all),
                                                 replacement=True)
        sampling_kwargs = {'sampler': weighted_sampler}
    else:
        sampling_kwargs = {'shuffle': True}
    return sampling_kwargs

def setup_CLLD_dataloaders(cfg: DictConfig,
                             train_transform:nn.Module,
                             finetune_transform:nn.Module,
                             test_transform:nn.Module) -> Dict[str, DataLoader]:
    df_train = pd.read_pickle(get_hydra_path(cfg.train.data))
    df_finetune = pd.read_pickle(get_hydra_path(cfg.finetune.data))
    df_test = pd.read_pickle(get_hydra_path(cfg.test.data))
    wm_train = WM811KDataset(df_train, transform=train_transform, num_views=2, include_raw=True, include_no_crop=True)
    wm_finetune = WM811KDataset(df_finetune, transform=finetune_transform)
    wm_finetune_eval = WM811KDataset(df_finetune, transform=test_transform)
    wm_test = WM811KDataset(df_test, transform=test_transform)
    wm_scl = WM811KDataset(df_finetune, transform=train_transform, num_views=1, include_no_crop=True) # scl data = label data + train trans

    dataloader_kwargs = {
        'num_workers': cfg.workers,
        'pin_memory': True
    }
    # Training Sampler
    train_sampling_kwargs = get_sampler(cfg.train.weight_sampler, wm_train.label)
    finetune_sampling_kwargs = get_sampler(cfg.finetune.weight_sampler, wm_finetune.label)
    train_dataloader = DataLoader(wm_train, batch_size=cfg.train.batch_size, drop_last=True,
                                     **train_sampling_kwargs, **dataloader_kwargs)
    finetune_dataloader = DataLoader(wm_finetune, batch_size=cfg.finetune.batch_size,
                                     **finetune_sampling_kwargs, **dataloader_kwargs)
    finetune_eval_dataloader = DataLoader(wm_finetune_eval, shuffle=False,
                                          batch_size=cfg.test.batch_size, **dataloader_kwargs)
    test_dataloader = DataLoader(wm_test, shuffle=False,
                                 batch_size=cfg.test.batch_size, **dataloader_kwargs)
    scl_dataloader = DataLoader(wm_scl, batch_size=cfg.finetune.batch_size,    # scl dataloader = scl data + weighted sampler
                                     **finetune_sampling_kwargs, **dataloader_kwargs)
    return {'train': train_dataloader,
            'finetune': finetune_dataloader,
            'finetune_eval': finetune_eval_dataloader,
            'test': test_dataloader,
            'scl':scl_dataloader}
