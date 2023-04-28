'''-------------------------------------------------------------------------
This file is part of Semi-CL-WMPD, a Python library for wafer map pattern
detection using semi-supervised constrastive learning with domain-specific
transformation.

Copyright (C) 2020-2021 Hanbin Hu <hanbinhu@ucsb.edu>
                        Peng Li <lip@ucsb.edu>
              University of California, Santa Barbara
-------------------------------------------------------------------------'''

from typing import Tuple, Dict
import logging
import hydra
from omegaconf import DictConfig

import numpy as np
import matplotlib
matplotlib.use("agg")

import pandas as pd
from skimage import measure
from skimage.transform import radon
from skimage import measure
from scipy import interpolate
from scipy import stats

import torch
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.base import _is_pairwise, clone
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.fixes import delayed
from joblib import Parallel

from utils import SummaryWriter, hydra_run_wrapper, compute_perf, write_log
from datasets import setup_SVM_dataframes

logger = logging.getLogger(__name__)

def _fit_binary(estimator, X, y, w, classes=None):
    """Fit a single binary estimator."""
    estimator = clone(estimator)
    if w is None:
        estimator.fit(X, y)
    else:
        estimator.fit(X, y, sample_weight=w)
    return estimator

def _fit_ovo_binary(estimator, X, y, w, i, j):
    """Fit a single binary estimator (one-vs-one)."""
    cond = np.logical_or(y == i, y == j)
    y = y[cond]
    y_binary = np.empty(y.shape, int)
    y_binary[y == i] = 0
    y_binary[y == j] = 1
    if w is None:
        w_binary = None
    else:
        w_binary = w[cond]
    indcond = np.arange(X.shape[0])[cond]
    return _fit_binary(estimator,
                       _safe_split(estimator, X, None, indices=indcond)[0],
                       y_binary, w_binary, classes=[i, j]), indcond

class OneVsOneClassifier(OneVsOneClassifier):
    def fit(self, X, y, sample_weights=None):
        """Fit underlying estimators.

        Parameters
        ----------
        X : (sparse) array-like of shape (n_samples, n_features)
            Data.

        y : array-like of shape (n_samples,)
            Multi-class targets.

        Returns
        -------
        self
        """
        X, y = self._validate_data(X, y, accept_sparse=['csr', 'csc'],
                                   force_all_finite=False)
        check_classification_targets(y)

        self.classes_ = np.unique(y)
        if len(self.classes_) == 1:
            raise ValueError("OneVsOneClassifier can not be fit when only one"
                             " class is present.")
        n_classes = self.classes_.shape[0]
        estimators_indices = list(zip(*(Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_ovo_binary)
            (self.estimator, X, y, sample_weights, self.classes_[i], self.classes_[j])
            for i in range(n_classes) for j in range(i + 1, n_classes)))))

        self.estimators_ = estimators_indices[0]

        pairwise = _is_pairwise(self)
        self.pairwise_indices_ = (
            estimators_indices[1] if pairwise else None)

        return self


def cal_den(x):
    return 100*(np.sum(x==2)/np.size(x))  

def get_ind(n):
    if n >= 5:
        return np.arange(0,n,n//5)
    elif n == 4:
        return np.array([0, 1, 1, 2, 3])
    elif n == 3:
        return np.array([0, 0, 1, 2, 2])
    elif n == 2:
        return np.array([0, 0, 0, 1, 1])
    elif n == 1:
        return np.array([0, 0, 0, 0, 0])

def find_regions(x):
    rows=np.size(x,axis=0)
    cols=np.size(x,axis=1)
    ind1=get_ind(rows)
    ind2=get_ind(cols)
    
    reg1=x[ind1[0]:ind1[1],:]
    reg2=x[:,ind2[4]:]
    reg3=x[ind1[4]:,:]
    reg4=x[:,ind2[0]:ind2[1]]
    reg5=x[ind1[1]:ind1[2],ind2[1]:ind2[2]]
    reg6=x[ind1[1]:ind1[2],ind2[2]:ind2[3]]
    reg7=x[ind1[1]:ind1[2],ind2[3]:ind2[4]]
    reg8=x[ind1[2]:ind1[3],ind2[1]:ind2[2]]
    reg9=x[ind1[2]:ind1[3],ind2[2]:ind2[3]]
    reg10=x[ind1[2]:ind1[3],ind2[3]:ind2[4]]
    reg11=x[ind1[3]:ind1[4],ind2[1]:ind2[2]]
    reg12=x[ind1[3]:ind1[4],ind2[2]:ind2[3]]
    reg13=x[ind1[3]:ind1[4],ind2[3]:ind2[4]]
    
    fea_reg_den = [cal_den(reg1), cal_den(reg2), cal_den(reg3), cal_den(reg4), cal_den(reg5),
                   cal_den(reg6), cal_den(reg7), cal_den(reg8), cal_den(reg9), cal_den(reg10),
                   cal_den(reg11), cal_den(reg12), cal_den(reg13)]
    return fea_reg_den

def cubic_inter_mean(img):
    theta = np.linspace(0., 180., max(img.shape), endpoint=False)
    sinogram = radon(img, theta=theta)
    xMean_Row = np.mean(sinogram, axis = 1)
    x = np.linspace(1, xMean_Row.size, xMean_Row.size)
    y = xMean_Row
    if xMean_Row.size >= 4:
        f = interpolate.interp1d(x, y, kind = 'cubic')
    else:
        f = interpolate.interp1d(x, y, kind = 'nearest')
    xnew = np.linspace(1, xMean_Row.size, 20)
    ynew = f(xnew)/100   # use interpolation function returned by `interp1d`
    return ynew

def cubic_inter_std(img):
    theta = np.linspace(0., 180., max(img.shape), endpoint=False)
    sinogram = radon(img, theta=theta)
    xStd_Row = np.std(sinogram, axis=1)
    x = np.linspace(1, xStd_Row.size, xStd_Row.size)
    y = xStd_Row
    if xStd_Row.size >= 4:
        f = interpolate.interp1d(x, y, kind = 'cubic')
    else:
        f = interpolate.interp1d(x, y, kind = 'nearest')
    xnew = np.linspace(1, xStd_Row.size, 20)
    ynew = f(xnew)/100   # use interpolation function returned by `interp1d`
    return ynew  

def cal_dist(img,x,y):
    dim0=np.size(img,axis=0)    
    dim1=np.size(img,axis=1)
    dist = np.sqrt((x-dim0/2)**2+(y-dim1/2)**2)
    return dist  

def fea_geom(img):
    norm_area=img.shape[0]*img.shape[1]
    norm_perimeter=np.sqrt((img.shape[0])**2+(img.shape[1])**2)
    
    img_labels = measure.label(img, connectivity=1, background=0)

    if img_labels.max()==0:
        img_labels[img_labels==0]=1
        no_region = 0
    else:
        info_region = stats.mode(img_labels[img_labels>0], axis = None)
        no_region = info_region[0][0]-1       
    
    prop = measure.regionprops(img_labels)
    prop_area = prop[no_region].area/norm_area
    prop_perimeter = prop[no_region].perimeter/norm_perimeter 
    
    prop_cent = prop[no_region].local_centroid 
    prop_cent = cal_dist(img,prop_cent[0],prop_cent[1])
    
    prop_majaxis = prop[no_region].major_axis_length/norm_perimeter 
    prop_minaxis = prop[no_region].minor_axis_length/norm_perimeter  
    prop_ecc = prop[no_region].eccentricity  
    prop_solidity = prop[no_region].solidity  
    
    return prop_area,prop_perimeter,prop_majaxis,prop_minaxis,prop_ecc,prop_solidity

def extract_features(df: pd.DataFrame) -> Tuple[np.array, np.array]:
    df_copy = df.copy()
    df_copy.reset_index(inplace=True)
    # Density-based Features (13)
    df_copy['fea_reg']=df_copy.waferMap.apply(find_regions)
    # Randon-based Features (40)
    df_copy['fea_cub_mean'] =df_copy.waferMap.apply(cubic_inter_mean)
    df_copy['fea_cub_std'] =df_copy.waferMap.apply(cubic_inter_std)
    # Geometry-based Features (6)
    df_copy['fea_geom'] =df_copy.waferMap.apply(fea_geom)
    # Combine all features
    df_all = df_copy.copy()
    a=[df_all.fea_reg[i] for i in range(df_all.shape[0])] #13
    b=[df_all.fea_cub_mean[i] for i in range(df_all.shape[0])] #20
    c=[df_all.fea_cub_std[i] for i in range(df_all.shape[0])] #20
    d=[df_all.fea_geom[i] for i in range(df_all.shape[0])] #6
    fea_all = np.concatenate((np.array(a),np.array(b),np.array(c),np.array(d)),axis=1) #59 in total
    label=[df_copy.failureType[i] for i in range(df_copy.shape[0])]
    label=np.array(label)
    return fea_all, label

def train(model: OneVsOneClassifier,
          datasets: Dict[str, Tuple[np.array, np.array]],
          cfg: DictConfig) -> None:
    data = datasets["Train"][0]
    label = datasets["Train"][1]
    if cfg.weight_sampler:
        sample_weights = compute_sample_weight("balanced", label)
        model = model.fit(data, label, sample_weights=sample_weights)
    else:
        model = model.fit(data, label)
    logger.info("Train completed.")

    train_perf, test_perf = evaluate_perf(model, datasets["Train"], datasets["Test"],
                                          cfg.num_classes)
    write_log(train_perf, 'Train(Best)')
    write_log(test_perf, 'Test(Best)')

def evaluate_perf(model: OneVsOneClassifier,
                  train_dataset: Tuple[np.array, np.array], 
                  test_dataset: Tuple[np.array, np.array], 
                  num_classes: int) -> Tuple[Dict[str, float], Dict[str, float]]:
    preds_train, targets_train = evaluation(model, train_dataset)
    preds_test, targets_test = evaluation(model, test_dataset)
    train_perf = compute_perf(0, preds_train, targets_train, num_classes)
    test_perf = compute_perf(0, preds_test, targets_test, num_classes)
    return train_perf, test_perf

def evaluation(model: OneVsOneClassifier,
               dataset: Tuple[np.array, np.array]) -> Tuple[torch.Tensor, torch.Tensor]:
    data = dataset[0]
    label = dataset[1]
    pred = model.predict(data)
    return torch.tensor(pred), torch.tensor(label)

def main(writer: SummaryWriter, cfg: DictConfig) -> None:
    dataframes = setup_SVM_dataframes(cfg.dataset)
    logger.info('WM811K data loaded.')
    datasets = {s:extract_features(df) for s, df in dataframes.items()}
    logger.info('Data transform completed.')
    svm = OneVsOneClassifier(LinearSVC(random_state = cfg.general.seed))
    train(svm, datasets, cfg.training)

@hydra.main(config_path='config', config_name='config_svm')
def run_program(cfg: DictConfig) -> None:
    hydra_run_wrapper(main, cfg, logger)
if __name__ == '__main__':
    run_program()