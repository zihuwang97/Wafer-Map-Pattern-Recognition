import os
import torch
from typing import Dict

# Ideas from https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    def __init__(self, patience, log_dir):
        self._patience = patience
        self._save_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)
        self._best_score = None
        self._best_train_perf = None
        self._best_val_perf = None
        self._early_stop = False
        self._best_epoch = None
        self._counter = 0

    def __call__(self, val_loss:float, train_perf:Dict[str,float], val_perf:Dict[str,float],
                 epoch:int, **models):
        if self._best_score is None:
            self._best_score = val_loss
            self._best_epoch = epoch
            self._best_train_perf = train_perf
            self._best_val_perf = val_perf
            self._save_checkpoint(epoch, **models)
        elif val_loss >= self._best_score:
            self._counter += 1
            if self._counter >= self._patience:
                self._early_stop = True
        else:
            self._best_score = val_loss
            self._best_epoch = epoch
            self._best_train_perf = train_perf
            self._best_val_perf = val_perf
            self._save_checkpoint(epoch, **models)
            self._counter = 0

    def _save_checkpoint(self, epoch, **models):
        data_file = os.path.join(self._save_dir, f'checkpoint_{epoch}.pt')
        model_state_dict = {model_name:model.state_dict() for model_name, model in models.items()}
        torch.save(model_state_dict, data_file)

    def load_best_checkpoint(self, **models):
        data_file = os.path.join(self._save_dir, f'checkpoint_{self.best_epoch}.pt')
        checkpoint = torch.load(data_file)
        for model_name, model in models.items():
            model.load_state_dict(checkpoint[model_name])
            model.eval()

    @property
    def best_train_perf(self):
        if self._best_train_perf is None:
            raise RuntimeError('No best epoch yet.')
        return self._best_train_perf

    @property
    def best_val_perf(self):
        if self._best_val_perf is None:
            raise RuntimeError('No best epoch yet.')
        return self._best_val_perf

    @property
    def best_epoch(self):
        if self._best_epoch is None:
            raise RuntimeError('No best epoch yet.')
        return self._best_epoch

    @property
    def early_stop(self):
        return self._early_stop
