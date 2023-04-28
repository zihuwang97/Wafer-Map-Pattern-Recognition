from typing import Tuple, List, Dict, Union, Sequence

import numpy as np
import torch
import torch.nn as nn
import random
from torchvision import transforms
import torchvision.transforms.functional as TF
from omegaconf import DictConfig

class RandomWMNoise:
    def __init__(self, flip_probability:float) -> None:
        self._flip_probability = flip_probability
        assert 0.0 <= self._flip_probability and self._flip_probability <= 0.5

    def __call__(self, sample:torch.Tensor):
        # Two condition to be satisfied for flip
        # Sampling probability satisfied
        # Sample > 0.25 indicates there is wafer at this pixel
        mask = torch.logical_and(torch.rand_like(sample) < self._flip_probability, sample > 0.25)
        sample[mask]=1.5-sample[mask]
        return sample

class RandomTwistTransform:
    def __init__(self, m):
        self.m = m
        self.id = np.arange(self.m)[...,None].T
        
    def _random_angle(self, size):
        L=size[1]+size[2]
        # Potential Bug with Pytorch + Numpy, change to torch randn instead to avoid it.
        # See details in https://github.com/pytorch/pytorch/issues/5059 and
        # https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
        a0=torch.randn(1).item()*np.sqrt(1/(1+2*self.m))
        a=torch.randn(self.m).numpy().astype(np.double)*np.sqrt(1/(1+2*self.m))
        b=torch.randn(self.m).numpy().astype(np.double)*np.sqrt(1/(1+2*self.m))

        def f(dist):
            vdist=dist[...,None]
            np.tensordot(vdist, self.id, axes=0)
            angle=np.matmul(vdist, self.id)*2*np.pi/L
            res=np.sqrt(2)*np.sum(a*np.cos(angle)+b*np.sin(angle),axis=-1)
            return res+a0
        return f
    
    @staticmethod
    def _generate_xymap(size):
        xmap, ymap=np.meshgrid(np.arange(size[2]),np.arange(size[1]))
        xmap=xmap-(size[2]-1)/2
        ymap=ymap-(size[1]-1)/2
        return xmap, ymap
    
    @staticmethod
    def _get_twisted_xymap(xmap, ymap, angle_f):
        dist=np.sqrt(xmap*xmap+ymap*ymap)
        angle=angle_f(dist)
        cosangle=np.cos(angle)
        sinangle=np.sin(angle)
        txmap=xmap*cosangle-ymap*sinangle
        tymap=xmap*sinangle+ymap*cosangle
        return txmap, tymap
    
    @staticmethod
    def _regularize_xymap(xmap, ymap, size):
        xmap += (size[2]-1)/2
        ymap += (size[1]-1)/2
        xmap = np.rint(xmap)
        ymap = np.rint(ymap)
        xmap[xmap<0]=0
        ymap[ymap<0]=0
        xmap[xmap>=size[1]]=size[1]-1
        ymap[ymap>=size[2]]=size[2]-1
        return xmap.astype(int), ymap.astype(int)
    
    @staticmethod
    def _map_xy(wm, xmap, ymap):
        new_wm=torch.ones_like(wm)*(-1)
        new_wm[0,xmap,ymap]=wm[0,:,:]
        #for i in range(xmap.shape[0]):
        #    for j in range(xmap.shape[1]):
        #        new_wm[0][xmap[i][j]][ymap[i][j]]=wm[0][i][j]
        return new_wm
    
    @staticmethod
    def _regularize_wm(nwm, owm):
        wm=nwm
        # Make sure wafer shape doesn't change
        wm[owm<0.1]=0
        wm[torch.logical_and(owm>=0.1,wm==0)]=-1
        num_missing_value=len(wm[wm<-0.1])
        while num_missing_value>0:
            # Move to 4 directions
            fwm=torch.cat(4*[wm])
            fwm[0][0][:]=-1
            fwm[0][1:][:]=wm[0][:-1][:]
            fwm[1][-1][:]=-1
            fwm[1][:-1][:]=wm[0][1:][:]
            fwm[2][:][0]=-1
            fwm[2][:][1:]=wm[0][:][:-1]
            fwm[3][:][-1]=-1
            fwm[3][:][:-1]=wm[0][:][1:]
            fwm[fwm<0.1]=-1
            # Compute number of valid cells
            valid=fwm.clone()
            valid[fwm<0.1]=0
            valid[fwm>=0.1]=1
            valid=torch.sum(valid,dim=0,keepdim=True)
            # Set Invalid one to 0, and compute average of normal and bad dies
            fwm[fwm<0.1]=0
            fwm=torch.sum(fwm,dim=0,keepdim=True)
            fwm[valid>0]=fwm[valid>0]/valid[valid>0]
            fwm[fwm>0.75]=1
            fwm[torch.logical_and(fwm>0.25, fwm<0.75)]=0.498
            # Replace to major elements on four directions
            wm[torch.logical_and(wm<-0.1,valid>0)]=fwm[torch.logical_and(wm<-0.1,valid>0)]
            prev_num=num_missing_value
            num_missing_value=len(wm[wm<-0.1])
            if prev_num == num_missing_value:
                break
        wm[wm<-0.1]=0.498
        return wm
    
    def __call__(self, wm):
        size=wm.size()
        angle_f= self._random_angle(size)
        xmap, ymap=RandomTwistTransform._generate_xymap(size)
        txmap, tymap=RandomTwistTransform._get_twisted_xymap(xmap, ymap, angle_f)
        rxmap, rymap=RandomTwistTransform._regularize_xymap(txmap, tymap, size)
        coarse_wm=RandomTwistTransform._map_xy(wm, rxmap, rymap)
        fine_wm=RandomTwistTransform._regularize_wm(coarse_wm, wm)
        return fine_wm

class RandomRotateTransform:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

def get_transform(input_size:int, mean:float, std:float, *,
                  noise_level:float,
                  flip_enabled:bool,
                  rotation_type:str,
                  crop_enabled:bool,
                  nit_enabled:bool=False):
    # Transform defined here
    transform_list = [transforms.ToTensor()]
    if noise_level > 0:
        noise_transform = RandomWMNoise(noise_level)
        transform_list.append(noise_transform)
    transform_list.append(transforms.Resize((input_size,input_size)))
    transform_list.append(transforms.Normalize((mean,),(std,)))

    if flip_enabled:
        transform_list.append(transforms.RandomHorizontalFlip())

    if rotation_type == "None":
        pass
    elif rotation_type == 'Continuous':
        transform_list.append(transforms.RandomRotation(180))
    elif rotation_type == 'Discrete':
        transform_list.append(RandomRotateTransform([0, 90, 180, 270]))
    else:
        raise NotImplementedError

    if crop_enabled:
        transform_list.append(transforms.RandomResizedCrop(size=input_size))

    if nit_enabled:
        transform_list = [
            transforms.ToTensor(),
            RandomWMNoise(0.05),
            transforms.Resize((input_size,input_size)),
            transforms.Normalize((mean,),(std,)),
            transforms.RandomResizedCrop(size=input_size),
            transforms.RandomHorizontalFlip()]
    pretrain_transform = transforms.Compose(transform_list)
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((input_size,input_size)),
        transforms.Normalize((mean,),(std,))
    ])
    return pretrain_transform, eval_transform

def get_single_transform(cfg: DictConfig,
                         size: int, mean: float, std: float, slice) -> nn.Module:
    # Transform defined here
    transform_list = [transforms.ToTensor()]

    if cfg.twist > 0:
        transform_list.append(RandomTwistTransform(cfg.twist))

    if cfg.noise > 0:
        noise_transform = RandomWMNoise(cfg.noise)
        transform_list.append(noise_transform)
    
    transform_list.append(transforms.Resize((size, size)))
    transform_list.append(transforms.Normalize((mean,), (std,)))

    if cfg.flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    if cfg.rotation == "None":
        pass
    elif cfg.rotation == 'Continuous':
        transform_list.append(transforms.RandomRotation(180))
    elif cfg.rotation == 'Discrete':
        transform_list.append(RandomRotateTransform([0, 90, 180, 270]))
    else:
        raise NotImplementedError
 
    if cfg.crop:
        transform_list.append(transforms.RandomResizedCrop(size=size))

    transform = transforms.Compose(transform_list)
    return transform

def get_transform(cfg: DictConfig) -> Union[Tuple[nn.Module, nn.Module],
                                            Tuple[nn.Module, nn.Module, nn.Module]]:
    train_transform = get_single_transform(cfg.train, cfg.size, cfg.mean, cfg.std, slice=True)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((cfg.size, cfg.size)),
        transforms.Normalize((cfg.mean,), (cfg.std,))
    ])
    if hasattr(cfg, 'finetune'):
        finetune_transform = get_single_transform(cfg.finetune, cfg.size, cfg.mean, cfg.std, slice=False)
        return train_transform, finetune_transform, test_transform
    else:
        return train_transform, test_transform