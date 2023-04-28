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
from torchvision import transforms

from utils import SummaryWriter, hydra_run_wrapper, compute_perf, \
                  write_tensorboard, write_log, write_learning_rate_log, init_weights
from models import WM811K_Encoder, WM811K_Projection_Head, WM811K_Supervised_Head, SimCLRLoss
from transform import get_transform
from datasets.datasets_CLLD import setup_CLLD_dataloaders
from early_stopping import EarlyStopping

logger = logging.getLogger(__name__)

def momentum_update_key_encoder(model_k, model_q, m):
    """
    Momentum update of the key encoder
    """
    for param_q, param_k in zip(model_q.parameters(), model_k.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1. - m)

def inter_wafer_da(batch_wm):
    b_size = batch_wm.size(0)
    idx_shuffle = torch.randperm(b_size)
    shuffled_wm = batch_wm[idx_shuffle]
    iwda_wm = torch.where(shuffled_wm>=batch_wm, shuffled_wm, batch_wm)
    return iwda_wm, idx_shuffle

def cal_iwda_loss(z_raw, z_iwda, idx_shuffle):
    # normalize
    z_raw = nn.functional.normalize(z_raw, dim=-1)
    z_iwda = nn.functional.normalize(z_iwda, dim=-1)
    z_shuffled = z_raw[idx_shuffle]
    # sort & generate mask
    simi = torch.sum(z_raw * z_shuffled, dim=-1)
    simi_sorted, indices = torch.sort(simi, descending=True)
    indices = indices[(simi_sorted<0.)*(simi_sorted>-0.8)]
    # calculate scores
    num_elig = indices.size(0)
    score = z_raw.detach() * z_iwda
    score = score[indices]
    score_shuffled = z_shuffled.detach() * z_iwda
    score_shuffled = score_shuffled[indices]
    score = torch.sum(score, dim=-1)/num_elig
    score_shuffled = torch.sum(score_shuffled, dim=-1)/num_elig
    loss = nn.functional.softplus(score.detach() - score_shuffled) + nn.functional.softplus(score_shuffled.detach() - score)
    loss = torch.sum(loss)
    # negative pairs
    neg_scores = z_iwda @ z_raw.detach().t()
    neg_scores = torch.exp(neg_scores) / 0.5
    neg_scores[torch.arange(len(z_iwda)).cuda(), torch.arange(len(z_iwda)).cuda()] = 0.
    neg_scores[torch.arange(len(z_iwda)).cuda(), idx_shuffle] = 0.
    neg_scores = neg_scores[indices]
    neg_scores = torch.sum(neg_scores, dim=-1)
    neg_scores = 1e-3 * torch.mean(torch.log(neg_scores))
    # overall loss
    loss = loss + neg_scores
    return loss

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
                         encoder_k,
                         projection_head_k,
                         dataloader: DataLoader,
                         criterion: nn.Module,
                         optimizer: optim.Optimizer,
                         epoch: int,
                         num_epochs: int):
    encoder.train()
    projection_head.train()
    loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
    for i, (_, wm, _) in loop:
        momentum_update_key_encoder(encoder_k, encoder, m=0.999)
        momentum_update_key_encoder(projection_head_k, projection_head, m=0.999)

        x_i, x_j, x_whole, x_raw = torch.split(wm, [1,1,1,1], dim=1)
        x_i, x_j, x_whole, x_raw = x_i.cuda(), x_j.cuda(), x_whole.cuda(), x_raw.cuda()
        
        x_iwda, idx_shuffle = inter_wafer_da(x_raw)
        x_raw = transforms.functional.normalize(x_raw, mean=0.4463, std=0.2564)

        h_i, h_j = encoder(x_i), encoder(x_j)
        z_i, z_j = projection_head(h_i), projection_head(h_j)
        z_iwda = projection_head(encoder(x_iwda))
        with torch.no_grad():
            z_whole = projection_head_k(encoder_k(x_whole))
            z_raw = projection_head_k(encoder_k(x_raw))

        loss_iwda = cal_iwda_loss(z_raw, z_iwda, idx_shuffle)

        loss = ( criterion(z_i, z_whole) + criterion(z_j, z_whole) ) / 2 + loss_iwda
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_description("Train Epoch [{}/{}]".format(epoch, num_epochs))
        loop.set_postfix(loss=loss.item())
    return loss.item()

def train(encoder: nn.Module,
          projection_head: nn.Module,
          encoder_k,
          projection_head_k,
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

    supervised_cl_finetune(encoder,projection_head,encoder_k,projection_head_k,dataloaders['scl']) # initial scl finetune
    
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
        last_batch_loss = run_one_epoch_SimCLR(encoder, projection_head, encoder_k, projection_head_k, dataloaders['train'],
                                               criterion, optimizer, epoch, cfg.train.epochs)
        writer.add_scalar('Train Loss', last_batch_loss, epoch)
        if epoch >= cfg.train.warmup:
            scheduler.step()

        if epoch % cfg.finetune_rounds == 0:
            supervised_cl_finetune(encoder,projection_head,encoder_k,projection_head_k,dataloaders['scl'])

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

def cluster_centers(data, labels):
    M = torch.zeros(9, len(data)).cuda()
    M[labels, torch.arange(len(data))] = 1
    M = torch.nn.functional.normalize(M, p=1, dim=1)
    centers = torch.mm(M, data)
    centers = torch.nn.functional.normalize(centers, dim=-1)
    return centers

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
                        num_epochs: int,) -> None:
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

def supervised_cl_finetune(encoder: nn.Module,
                             proj_head: nn.Module,
                             encoder_k,
                             proj_head_k,
                             dataloader: DataLoader,) -> None:
    optimizer = optim.Adam(list(encoder.parameters())+list(proj_head.parameters()),
                           lr=1e-4, weight_decay=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader))

    for epoch in range(1,10+1):
        run_one_epoch_scl(encoder, proj_head, dataloader,
                                    optimizer, epoch, 10)
        scheduler.step()
    momentum_update_key_encoder(encoder_k, encoder, m=0.)
    momentum_update_key_encoder(proj_head_k, proj_head, m=0.)


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
    dataloaders = setup_CLLD_dataloaders(cfg.dataset,
                                           train_transform, finetune_transform, test_transform)
    logger.info('WM811K data loaded.')

    encoder = WM811K_Encoder(cfg.model)
    projection_head = WM811K_Projection_Head(cfg.model)

    encoder_k = WM811K_Encoder(cfg.model)
    projection_head_k = WM811K_Projection_Head(cfg.model)

    for param_q, param_k in zip(encoder.parameters(), encoder_k.parameters()):
        param_k.data.copy_(param_q.data)  # initialize
        param_k.requires_grad = False  # not update by gradient

    for param_q, param_k in zip(projection_head.parameters(), projection_head_k.parameters()):
        param_k.data.copy_(param_q.data)  # initialize
        param_k.requires_grad = False  # not update by gradient

    sup_head = WM811K_Supervised_Head(cfg.model)
    writer.add_graph(encoder, torch.rand(1,1,cfg.model.input_size,cfg.model.input_size))
    writer.add_graph(projection_head, torch.rand(1,cfg.model.embedding_size))
    writer.add_graph(sup_head, torch.rand(1,cfg.model.embedding_size))
    encoder.cuda()
    projection_head.cuda()
    encoder_k.cuda()
    projection_head_k.cuda()
    sup_head.cuda()
    train(encoder, projection_head, encoder_k, projection_head_k, sup_head, dataloaders, writer, cfg.training)


@hydra.main(config_path='config', config_name='config_simclr')
def run_program(cfg: DictConfig) -> None:
    hydra_run_wrapper(main, cfg, logger)
if __name__ == '__main__':
    run_program()