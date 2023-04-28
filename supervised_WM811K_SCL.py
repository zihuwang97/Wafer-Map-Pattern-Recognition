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
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

import numpy as np
import matplotlib
matplotlib.use("agg")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from utils import SummaryWriter, hydra_run_wrapper, compute_perf, init_weights,\
                  write_tensorboard, write_log, write_learning_rate_log
from datasets import setup_SCL_dataloaders
from transform import get_transform
from models import WM811K_Encoder, WM811K_Projection_Head, WM811K_Supervised_Head
from early_stopping import EarlyStopping

logger = logging.getLogger(__name__)

def acc_per_class(pred, target, num_classes=9):
    list_of_classes = range(num_classes)
    acc = [0 for c in list_of_classes]
    for c in list_of_classes:
        acc[c] = ((pred == target) * (target == c)).sum() / (max((target == c).sum(), 1))
        if c == 5:
            pred_5 = np.bincount(pred[target == c])
        elif c == 7:
            pred_7 = np.bincount(pred[target == c])

    return acc, pred_5, pred_7

def scl_loss(z, target, temp=0.7):
    # calculate scores
    scores = z @ z.t()
    scores = scores / temp
    scores = torch.exp(scores) 

    # neg & pos masks
    neg_mask = target.unsqueeze(0) - target.unsqueeze(0).t()
    neg_mask = torch.where(neg_mask==0, 0., 1.)
    pos_mask = 1. - neg_mask
    pos_count = (torch.sum(pos_mask, dim=-1) - 1).clamp(min=1)

    # pos & neg scores
    neg_scores = scores * neg_mask
    pos_scores = scores * pos_mask

    # loss
    neg_sum = torch.sum(neg_scores, dim=-1).unsqueeze(0).t()
    logits = pos_scores / neg_sum
    logits = torch.where(logits>1e-4, logits, torch.ones_like(logits))
    log_logits = torch.sum(torch.log(logits), dim=-1)
    loss = - torch.sum(log_logits / pos_count)
    return loss

def run_one_epoch_scl(encoder: nn.Module,
                            proj_head: nn.Module,
                            dataloader: DataLoader,
                            optimizer: optim.Optimizer,
                            epoch: int,
                            num_epochs: int) -> None:
    encoder.train()
    proj_head.train()
    loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
    for _, (_, wm, target) in loop:
        wm, target = wm.cuda(), target.cuda()
        wm1, wm2 = torch.split(wm, [1,1], dim=1)
        wm = torch.cat((wm1, wm2))
        reps = proj_head(encoder(wm))
        reps = torch.nn.functional.normalize(reps, dim=-1)
        target = torch.cat((target, target))
        loss = scl_loss(reps, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_description("SCL Finetune Epoch [{}/{}]".format(epoch, num_epochs))
        loop.set_postfix(loss=loss.item())

def run_one_epoch(model: nn.Module,
                  optimizer: optim.Optimizer,
                  dataloader: DataLoader,
                  criterion: nn.Module,
                  epoch: int,
                  num_epochs: int) -> None:
    model.train()
    loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
    for _, (_, wm, target) in loop:
        wm, target = wm.cuda(), target.cuda()
        logits = model(wm)
        loss = criterion(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_description("Epoch [{}/{}]".format(epoch, num_epochs))
        loop.set_postfix(loss=loss.item())

def train(encoder: nn.Module,
          proj_head: nn.Module,
          lin_classifier: nn.Module,
          dataloaders: Dict[str, DataLoader],
          writer: SummaryWriter,
          cfg: DictConfig) -> None:
    if cfg.early_stopping.enabled:
        early_stopping = EarlyStopping(cfg.early_stopping.patience, writer.log_dir)

    optimizer = optim.Adam(list(encoder.parameters())+list(proj_head.parameters()), 
                           lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloaders["Train"]))
    criterion = nn.CrossEntropyLoss()
    train_perf, test_perf = evaluate_perf(
        encoder, lin_classifier, dataloaders["Train Eval"], dataloaders["Test"],
        criterion, writer, 9, 0)
    if cfg.early_stopping.enabled:
        early_stopping(test_perf['Loss'], train_perf, test_perf, 0, 
                       encoder=encoder, proj_head=proj_head, lin_classifier=lin_classifier)
    write_learning_rate_log(writer, optimizer, 0)
    for epoch in range(1,cfg.epochs+1):
        # SCL for learning representations
        run_one_epoch_scl(encoder, proj_head, dataloaders["Train"], optimizer, epoch, cfg.epochs)
        # Supervised training of a classifier above the frozen encoder
        train_classifier(encoder, lin_classifier, dataloaders['finetune'], cfg.finetune)
        train_perf, test_perf = evaluate_perf(
            encoder, lin_classifier, dataloaders["Train Eval"], dataloaders["Test"],
            criterion, writer, 9, epoch)
        scheduler.step()
        write_learning_rate_log(writer, optimizer, epoch)

        if cfg.early_stopping.enabled:
            early_stopping(test_perf['Loss'], train_perf, test_perf, epoch, 
                           encoder=encoder, proj_head=proj_head, lin_classifier=lin_classifier)
            if early_stopping.early_stop:
                logger.info(f'Early stopped at epoch {epoch}')
                break

    if cfg.early_stopping.enabled:
        logger.info(f'Loading model from epoch {early_stopping.best_epoch}')
        early_stopping.load_best_checkpoint(encoder=encoder, 
                                            proj_head=proj_head, 
                                            lin_classifier=lin_classifier)
        write_log(early_stopping.best_train_perf, 'Train(Best)')
        write_log(early_stopping.best_val_perf, 'Test(Best)')
    else:
        write_log(train_perf, 'Train(Best)')
        write_log(test_perf, 'Test(Best)')

def run_one_epoch_classify(encoder: nn.Module,
                           lin_classifier: nn.Module,
                           dataloader: DataLoader,
                           criterion: nn.Module,
                           optimizer: optim.Optimizer,
                           epoch: int,
                           num_epochs: int) -> None:
    lin_classifier.train()
    loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
    for _, (_, wm, target) in loop:
        wm, target = wm.cuda(), target.cuda()
        with torch.no_grad():
            embedding = encoder(wm)
        logits = lin_classifier(embedding)
        loss = criterion(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_description("Finetune Epoch [{}/{}]".format(epoch, num_epochs))
        loop.set_postfix(loss=loss.item())

def train_classifier(encoder: nn.Module,
                        lin_classifier: nn.Module,
                        dataloader: DataLoader,
                        cfg: DictConfig) -> None:
    encoder.eval()
    lin_classifier.apply(init_weights)

    optimizer = optim.Adam(lin_classifier.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1,cfg.epochs+1):
        run_one_epoch_classify(encoder, lin_classifier, dataloader,
                               criterion, optimizer, epoch, cfg.epochs)
        scheduler.step()

def evaluate_perf(encoder: nn.Module,
                  lin_classifier: nn.Module,
                  train_loader: DataLoader,
                  test_loader: DataLoader,
                  criterion: nn.Module,
                  writer: SummaryWriter,
                  num_classes: int,
                  round: int) -> Tuple[Dict[str, float], Dict[str, float]]:
    name_train = 'train'
    name_test = 'test'
    loss_train, preds_train, targets_train = evaluation(encoder, lin_classifier, train_loader, criterion, name_train)
    loss_test, preds_test, targets_test = evaluation(encoder, lin_classifier, test_loader, criterion, name_test)
    train_perf = compute_perf(loss_train, preds_train, targets_train, num_classes)
    test_perf = compute_perf(loss_test, preds_test, targets_test, num_classes)
    write_tensorboard(writer, train_perf, name_train, round)
    write_tensorboard(writer, test_perf, name_test, round)
    write_log(train_perf, name_train, round)
    write_log(test_perf, name_test, round)
    acc_class, acc5, acc7 = acc_per_class(preds_test, targets_test) ## ------- Acc per class ------- ##
    print(acc_class)
    print(acc5)
    print(acc7)
    return train_perf, test_perf

def evaluation(encoder: nn.Module,
               lin_classifier: nn.Module,
               dataloader: DataLoader,
               criterion: nn.Module,
               eval_type: str) -> Tuple[float, torch.Tensor, torch.Tensor]:
    encoder.eval()
    lin_classifier.eval()
    with torch.no_grad():
        loss_all = 0
        preds_all = []
        targets_all = []
        loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
        for _, (_, wm, target) in loop:
            wm, target = wm.cuda(), target.cuda()
            logits = lin_classifier(encoder(wm))
            loss = criterion(logits, target)
            loss_all += loss.item()*len(target)
            preds = torch.argmax(logits, dim=1)
            preds_all.append(preds.cpu().numpy())
            targets_all.append(target.cpu().numpy())
            loop.set_description("Eval ({})".format(eval_type))
            loop.set_postfix(loss=loss.item())
        loss_eval = loss_all / len(dataloader.dataset)
        preds_eval = np.concatenate(preds_all)
        targets_eval = np.concatenate(targets_all)
    return loss_eval, preds_eval, targets_eval

def main(writer: SummaryWriter, cfg: DictConfig) -> None:
    train_transform, finetune_transform, test_transform = get_transform(cfg.transform)
    dataloaders = setup_SCL_dataloaders(cfg.dataset, train_transform, test_transform, finetune_transform)
    logger.info('WM811K data loaded.')

    encoder = WM811K_Encoder(cfg.model)
    proj_head = WM811K_Projection_Head(cfg.model)
    lin_classifier = WM811K_Supervised_Head(cfg.model)
    # writer.add_graph(model, torch.rand(1,1,cfg.model.input_size,cfg.model.input_size))
    encoder.cuda()
    proj_head.cuda()
    lin_classifier.cuda()
    train(encoder, proj_head, lin_classifier, dataloaders, writer, cfg.training)

@hydra.main(config_path='config', config_name='config_scl')
def run_program(cfg: DictConfig) -> None:
    hydra_run_wrapper(main, cfg, logger)
if __name__ == '__main__':
    run_program()