from argparse import ArgumentParser, Namespace
from typing import Callable, Dict, Union
import warnings
import random
import logging
import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np
import sklearn.metrics as skm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

# Solve the problem of extra folder when using add_hparams
# Solution found on https://github.com/pytorch/pytorch/issues/32651#issuecomment-643791116
class SummaryWriter(SummaryWriter):
    def add_hparams(self, hparam_dict, metric_dict):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        logdir = self._get_file_writer().get_logdir()

        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v)

# Write a score scalar inferred by prediction and target to a writer using a score function.
class ScalarWriter:
    def __init__(self,
                 writer:SummaryWriter,
                 pred:np.array,
                 target:np.array,
                 eval_type:str,
                 epoch:int):
        self.writer = writer
        self.pred = pred
        self.target = target
        self.eval_type = eval_type
        self.epoch = epoch

    def write(self,
              score_type:str,
              score_func:Callable[[np.array, np.array], float],
              **kwargs) -> None:
        score_name = '{}/{}'.format(score_type, self.eval_type)
        score = score_func(self.target, self.pred, **kwargs)
        if self.epoch >= 0:
            self.writer.add_scalar(score_name, score, global_step=self.epoch)
        else:
            self.writer.add_scalar(score_name, score)
        return score

def init_weights(m:nn.Module):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)

# Set random seed for everything
def seed_everything(seed:int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Set the logger Status
def set_logger(logger:logging.Logger, log_file:str):
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.info(f'logs written into file {log_file}.')
    else:
        logger.info(f'logs written into stdout.')

def write_learning_rate_log(writer:SummaryWriter, optimizer:optim.Optimizer, epoch:int = -1):
    for group_id, param_group in enumerate(optimizer.param_groups):
        scalar_name = f'Learning rate/G{group_id}'
        lr = param_group['lr']
        if epoch < 0:
            writer.add_scalar(scalar_name, lr)
        else:
            writer.add_scalar(scalar_name, lr, epoch)

class SingletonLogger:
    __logger = None
    @staticmethod
    def getInstance():
        return SingletonLogger.__logger

    def __init__(self, logger:logging.Logger):
        SingletonLogger.__logger = logger

def get_logger():
    return SingletonLogger.getInstance()

def main_func_wrapper(
    main_func:Callable[[SummaryWriter,Namespace],Dict[str,float]],
    get_parser:Callable[[],ArgumentParser],
    save_hyperparameters:Callable[[SummaryWriter,Dict[str,float],Namespace],None],
    logger:logging.Logger):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    parser = get_parser()
    args = parser.parse_args()
    SingletonLogger(logger)
    set_logger(logger, args.log_file)
    if args.log_dir is not None:
        writer = SummaryWriter(args.log_dir)
    else:
        writer = SummaryWriter()
    logger.info(f'Tensorboard outputs to {writer.log_dir}.')

    if args.seed is not None:
        seed_everything(args.seed)
        logger.info(f'Set seed to {args.seed}.')

    perf = main_func(writer, args)
    save_hyperparameters(writer, perf, args)
    writer.close()

def combine_perf_results(train:Dict[str,float], val:Dict[str,float], test:Dict[str,float]):
    train_prefix = {'Train '+metric:val for metric, val in train.items()}
    val_prefix = {'Val '+metric:val for metric, val in val.items()}
    test_prefix = {'Test '+metric:val for metric, val in test.items()}
    perf = {**train_prefix, **val_prefix, **test_prefix}
    return perf

def hydra_run_wrapper(
    main_func:Callable[[SummaryWriter,DictConfig],None],
    cfg: DictConfig,
    logger: logging.Logger) -> None:
    warnings.simplefilter(action='ignore', category=FutureWarning)
    SingletonLogger(logger)
    if hasattr(cfg.general, 'seed'):
        seed_everything(cfg.general.seed)
        logger.info(f'Set seed to {cfg.general.seed}.')
    writer = SummaryWriter(cfg.general.logname)
    logger.info(f'Configuration:\n{OmegaConf.to_yaml(cfg)}')
    torch.cuda.set_device(cfg.general.gpu)
    main_func(writer, cfg)
    writer.close()

def get_hydra_path(path: str) -> str:
    return hydra.utils.to_absolute_path(path)

def compute_perf(loss: float,
                 pred: torch.Tensor,
                 target: torch.Tensor,
                 num_classes: int) -> Dict[str, Union[float, np.array]]:
    perf = {}
    perf['Loss'] = loss
    perf['Acc'] = skm.accuracy_score(target, pred)
    perf['Balanced Acc'] = skm.balanced_accuracy_score(target, pred)
    perf['F1 Score'] = skm.f1_score(target, pred,
                                    average='macro',
                                    labels=list(range(num_classes)),
                                    zero_division=1)
    perf['Precision'] = skm.precision_score(target, pred,
                                            average='macro',
                                            labels=list(range(num_classes)),
                                            zero_division=1)
    perf['Recall'] = skm.recall_score(target, pred,
                                      average='macro',
                                      labels=list(range(num_classes)),
                                      zero_division=1)
    perf['Confusion Matrix'] = skm.confusion_matrix(target, pred)
    return perf

def write_log(perf: Dict[str, Union[float, np.array]], eval_type: str, round_cnt: int=-1) -> None:
    logger = get_logger()
    round_str = f' at step {round_cnt}' if round_cnt >= 0 else ''
    logger.info(f'Loss/{eval_type}{round_str}: {perf["Loss"]:.4f}')
    logger.info(f'Acc/{eval_type}{round_str}: {perf["Acc"]:.4f}')
    logger.info(f'Balanced Acc/{eval_type}{round_str}: {perf["Balanced Acc"]:.4f}')
    logger.info(f'F1 Score/{eval_type}{round_str}: {perf["F1 Score"]:.4f}')
    logger.info(f'Precision/{eval_type}{round_str}: {perf["Precision"]:.4f}')
    logger.info(f'Recall/{eval_type}{round_str}: {perf["Recall"]:.4f}')

def write_tensorboard(writer: SummaryWriter,
                      perf: Dict[str, Union[float, np.array]],
                      eval_type: str,
                      round_cnt: int) -> None:
    assert round_cnt >= 0
    writer.add_scalar('Loss/{}'.format(eval_type), perf['Loss'], round_cnt)
    writer.add_scalar('Acc/{}'.format(eval_type), perf['Acc'], round_cnt)
    writer.add_scalar('Balanced Acc/{}'.format(eval_type), perf['Balanced Acc'], round_cnt)
    writer.add_scalar('F1 Score/{}'.format(eval_type), perf['F1 Score'], round_cnt)
    writer.add_scalar('Precision/{}'.format(eval_type), perf['Precision'], round_cnt)
    writer.add_scalar('Recall/{}'.format(eval_type), perf['Recall'], round_cnt)

    labels = ['none', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring',
              'Loc', 'Random', 'Scratch', 'Near-full']
    disp = skm.ConfusionMatrixDisplay(perf['Confusion Matrix'], display_labels=labels)
    fig_ = disp.plot().figure_
    writer.add_figure("Confusion matrix ({})".format(eval_type), fig_, round_cnt)