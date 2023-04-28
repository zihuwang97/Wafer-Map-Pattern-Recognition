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
import copy
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

from utils import SummaryWriter, hydra_run_wrapper, compute_perf, \
                  write_tensorboard, write_log, write_learning_rate_log, init_weights
from models import WM811K_Encoder, WM811K_Projection_Head, WM811K_Supervised_Head, SimCLRLoss
from transform import get_transform
from datasets.datasets import setup_SimCLR_dataloaders
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

def run_one_epoch_SimCLR(encoder: nn.Module,
                         projection_head: nn.Module,
                         dataloader: DataLoader,
                         criterion: nn.Module,
                         optimizer: optim.Optimizer,
                         epoch: int,
                         num_epochs: int):
    encoder.train()
    projection_head.train()
    loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
    for i, (_, wm, _) in loop:
        x_i, x_j = torch.split(wm, [1,1], dim=1)
        x_i, x_j = x_i.cuda(), x_j.cuda()
        h_i, h_j = encoder(x_i), encoder(x_j)
        z_i, z_j = projection_head(h_i), projection_head(h_j)
        loss = criterion(z_i, z_j)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_description("Train Epoch [{}/{}]".format(epoch, num_epochs))
        loop.set_postfix(loss=loss.item())
    return loss.item()

def train(encoder: nn.Module,
          projection_head: nn.Module,
          sup_head: nn.Module,
          dataloaders: Dict[str, DataLoader],
          writer: SummaryWriter,
          cfg: DictConfig) -> None:
    if cfg.early_stopping.enabled:
        early_stopping = EarlyStopping(cfg.early_stopping.patience, writer.log_dir)

    optimizer = optim.Adam(list(encoder.parameters())+list(projection_head.parameters()),
                           lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloaders['train']))
    criterion = SimCLRLoss(batch_size=dataloaders['train'].batch_size, normalize=True,
                           temperature=cfg.train.temperature).cuda()
    criterion_finetune = nn.CrossEntropyLoss()

    supervised_finetune(encoder, sup_head, dataloaders['finetune'], cfg.finetune)
    if hasattr(cfg, "deep_finetune"):
        encoder_copy = copy.deepcopy(encoder)
        supervised_deep_finetune(encoder_copy, sup_head, dataloaders['finetune'],
                                 cfg.deep_finetune)
        encoder_eval = encoder_copy
    else:
        encoder_eval = encoder

    write_learning_rate_log(writer, optimizer, 0)
    finetune_perf, test_perf = evaluate_perf(encoder_eval, sup_head,
                                             dataloaders['finetune_eval'], dataloaders['test'],
                                             criterion_finetune, writer, sup_head.num_classes, 0)
    if cfg.early_stopping.enabled:
        early_stopping(test_perf['Loss'], finetune_perf, test_perf, 0,
                       encoder=encoder, projection_head=projection_head, sup_head=sup_head)

    for epoch in range(1,cfg.train.epochs+1):
        last_batch_loss = run_one_epoch_SimCLR(encoder, projection_head, dataloaders['train'],
                                               criterion, optimizer, epoch, cfg.train.epochs)
        writer.add_scalar('Train Loss', last_batch_loss, epoch)
        if epoch >= cfg.train.warmup:
            scheduler.step()

        if epoch % cfg.finetune_rounds == 0:
            supervised_finetune(encoder, sup_head, dataloaders['finetune'], cfg.finetune)
            if hasattr(cfg, "deep_finetune"):
                encoder_copy = copy.deepcopy(encoder)
                supervised_deep_finetune(encoder_copy, sup_head, dataloaders['finetune'],
                                         cfg.deep_finetune)
                encoder_eval = encoder_copy
            else:
                encoder_eval = encoder

            write_learning_rate_log(writer, optimizer, epoch)
            finetune_perf, test_perf = evaluate_perf(encoder_eval, sup_head,
                                                     dataloaders['finetune_eval'],
                                                     dataloaders['test'],
                                                     criterion_finetune,
                                                     writer, sup_head.num_classes, epoch)
            if cfg.early_stopping.enabled:
                early_stopping(test_perf['Loss'], finetune_perf, test_perf, epoch,
                               encoder=encoder, projection_head=projection_head, sup_head=sup_head)
                if early_stopping.early_stop:
                    logger.info(f'Early stopped at epoch {epoch}')
                    break

    if cfg.early_stopping.enabled:
        logger.info(f'Loading model from epoch {early_stopping.best_epoch}')
        early_stopping.load_best_checkpoint(encoder=encoder,
                                            projection_head=projection_head,
                                            sup_head=sup_head)
        write_log(early_stopping.best_train_perf, 'Finetune(Best)')
        write_log(early_stopping.best_val_perf, 'Test(Best)')
    else:
        write_log(finetune_perf, 'Finetune(Best)')
        write_log(test_perf, 'Test(Best)')

def evaluation(encoder: nn.Module,
               sup_head: nn.Module,
               dataloader: DataLoader,
               criterion: nn.Module,
               eval_type: str) -> Tuple[float, torch.Tensor, torch.Tensor]:
    encoder.eval()
    sup_head.eval()
    with torch.no_grad():
        loss_all = 0
        preds_all = []
        targets_all = []
        loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
        for _, (_, wm, target) in loop:
            wm, target = wm.cuda(), target.cuda()
            logits = sup_head(encoder(wm))
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

def evaluate_perf(encoder: nn.Module,
                  sup_head: nn.Module,
                  finetune_eval_loader: DataLoader,
                  test_loader: DataLoader,
                  criterion: nn.Module,
                  writer: SummaryWriter,
                  num_classes: int,
                  round: int) -> Tuple[Dict[str, float], Dict[str, float]]:
    name_finetune = 'finetune'
    name_test = 'test'
    loss_finetune, preds_finetune, targets_finetune = evaluation(encoder, sup_head,
                                                                 finetune_eval_loader,
                                                                 criterion, name_finetune)
    loss_test, preds_test, targets_test = evaluation(encoder, sup_head, test_loader,
                                                     criterion, name_test)
    finetune_perf = compute_perf(loss_finetune, preds_finetune, targets_finetune, num_classes)
    test_perf = compute_perf(loss_test, preds_test, targets_test, num_classes)
    write_tensorboard(writer, finetune_perf, name_finetune, round)
    write_tensorboard(writer, test_perf, name_test, round)
    write_log(finetune_perf, name_finetune, round)
    write_log(test_perf, name_test, round)
    acc_class, acc5, acc7 = acc_per_class(preds_test, targets_test) ## ------- Acc per class ------- ##
    print(acc_class)
    print(acc5)
    print(acc7)
    return finetune_perf, test_perf

def run_one_epoch_deep_finetune(encoder: nn.Module,
                                sup_head: nn.Module,
                                dataloader: DataLoader,
                                criterion: nn.Module,
                                optimizer: optim.Optimizer,
                                epoch: int,
                                num_epochs: int) -> None:
    encoder.train()
    sup_head.train()
    loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
    for _, (_, wm, target) in loop:
        wm, target = wm.cuda(), target.cuda()
        logits = sup_head(encoder(wm))
        loss = criterion(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_description("Deep Finetune Epoch [{}/{}]".format(epoch, num_epochs))
        loop.set_postfix(loss=loss.item())

def supervised_deep_finetune(encoder: nn.Module,
                             sup_head: nn.Module,
                             dataloader: DataLoader,
                             cfg: DictConfig) -> None:
    optimizer = optim.Adam(list(encoder.parameters())+list(sup_head.parameters()),
                           lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1,cfg.epochs+1):
        run_one_epoch_deep_finetune(encoder, sup_head, dataloader,
                                    criterion, optimizer, epoch, cfg.epochs)
        scheduler.step()

def run_one_epoch_finetune(encoder: nn.Module,
                           sup_head: nn.Module,
                           dataloader: DataLoader,
                           criterion: nn.Module,
                           optimizer: optim.Optimizer,
                           epoch: int,
                           num_epochs: int) -> None:
    sup_head.train()
    loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
    for _, (_, wm, target) in loop:
        wm, target = wm.cuda(), target.cuda()
        with torch.no_grad():
            embedding = encoder(wm)
        logits = sup_head(embedding)
        loss = criterion(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_description("Finetune Epoch [{}/{}]".format(epoch, num_epochs))
        loop.set_postfix(loss=loss.item())

def supervised_finetune(encoder: nn.Module,
                        sup_head: nn.Module,
                        dataloader: DataLoader,
                        cfg: DictConfig) -> None:
    encoder.eval()
    sup_head.apply(init_weights)

    optimizer = optim.Adam(sup_head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1,cfg.epochs+1):
        run_one_epoch_finetune(encoder, sup_head, dataloader,
                               criterion, optimizer, epoch, cfg.epochs)
        scheduler.step()

def main(writer: SummaryWriter, cfg: DictConfig) -> None:
    train_transform, finetune_transform, test_transform = get_transform(cfg.transform)
    dataloaders = setup_SimCLR_dataloaders(cfg.dataset,
                                           train_transform, finetune_transform, test_transform)
    logger.info('WM811K data loaded.')

    encoder = WM811K_Encoder(cfg.model)
    projection_head = WM811K_Projection_Head(cfg.model)
    sup_head = WM811K_Supervised_Head(cfg.model)
    writer.add_graph(encoder, torch.rand(1,1,cfg.model.input_size,cfg.model.input_size))
    writer.add_graph(projection_head, torch.rand(1,cfg.model.embedding_size))
    writer.add_graph(sup_head, torch.rand(1,cfg.model.embedding_size))
    encoder.cuda()
    projection_head.cuda()
    sup_head.cuda()

    train(encoder, projection_head, sup_head, dataloaders, writer, cfg.training)


@hydra.main(config_path='config', config_name='config_simclr')
def run_program(cfg: DictConfig) -> None:
    hydra_run_wrapper(main, cfg, logger)
if __name__ == '__main__':
    run_program()