from typing import Tuple
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

class WM811K_Classifier(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        activation_layer = nn.ReLU()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, 5, padding=2),
            nn.MaxPool2d(2),
            activation_layer,
            nn.Conv2d(64, 32, 3, padding=1),
            nn.MaxPool2d(2),
            activation_layer,
            nn.Conv2d(32, 32, 3, padding=1),
            nn.MaxPool2d(2),
            activation_layer,
        )
        self.classifier = nn.Sequential(
            nn.Linear(cfg.input_size*cfg.input_size//2, cfg.embedding_size),
            activation_layer,
            nn.Linear(cfg.embedding_size, cfg.num_classes)
        )
        self.num_classes = cfg.num_classes

    def forward(self, x:torch.Tensor):
        feature = self.feature_extractor(x)
        feature = torch.flatten(feature, start_dim=1)
        y = self.classifier(feature)
        return y

class WM811K_Encoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        activation_layer = nn.ReLU()
        self.conv_feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, 5, padding=2),
            nn.MaxPool2d(2),
            activation_layer,
            nn.Conv2d(64, 32, 3, padding=1),
            nn.MaxPool2d(2),
            activation_layer,
            nn.Conv2d(32, 32, 3, padding=1),
            nn.MaxPool2d(2),
            activation_layer,
        )
        self.linear_feature_extractor = nn.Sequential(
            nn.Linear(cfg.input_size*cfg.input_size//2, cfg.embedding_size),
            nn.BatchNorm1d(cfg.embedding_size),
        )

    def forward(self, x:torch.Tensor):
        conv_feature = self.conv_feature_extractor(x)
        flatten_feature = torch.flatten(conv_feature, start_dim=1)
        embedding = self.linear_feature_extractor(flatten_feature)
        return embedding

class WM811K_Encoder_AE(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.activation_layer = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.linear_feature_extractor = nn.Sequential(
            nn.Linear(cfg.input_size*cfg.input_size//2, cfg.embedding_size),
            nn.BatchNorm1d(cfg.embedding_size),
        )

    def forward(self, x:torch.Tensor, eval=True):
        c1, i1 = self.pool(self.conv1(x))
        c2, i2 = self.pool(self.conv2(self.activation_layer(c1)))
        c3, i3 = self.pool(self.conv3(self.activation_layer(c2)))
        conv_feature = self.activation_layer(c3)
        flatten_feature = torch.flatten(conv_feature, start_dim=1)
        embedding = self.linear_feature_extractor(flatten_feature)
        if eval:
            return embedding
        else:
            return embedding, (i1, i2, i3)

class WM811K_Decoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.activation_layer = nn.ReLU()
        self.unpool = nn.MaxUnpool2d(2)
        self.deconv1 = nn.Conv2d(64, 1, 5, padding=2)
        self.deconv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 32, 3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.linear_feature_decoder = nn.Sequential(
            nn.Linear(cfg.embedding_size, cfg.input_size*cfg.input_size//2),
            nn.Unflatten(1, (32,cfg.input_size//8, cfg.input_size//8))
        )
        self.input_size = cfg.input_size

    def forward(self, embedding:torch.Tensor, indices:Tuple[torch.Tensor]):
        conv_feature = self.linear_feature_decoder(embedding)
        c3 = self.deconv3(self.unpool(self.activation_layer(conv_feature), indices[2]))
        c2 = self.deconv2(self.unpool(self.activation_layer(c3), indices[1]))
        c1 = self.deconv1(self.unpool(self.activation_layer(c2), indices[0]))
        x = self.sigmoid(c1)
        return x

class WM811K_Projection_Head(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(cfg.embedding_size, cfg.embedding_size),
            nn.BatchNorm1d(cfg.embedding_size),
            nn.ReLU(),
            nn.Linear(cfg.embedding_size, cfg.metric_size),
            nn.BatchNorm1d(cfg.metric_size),
        )

    def forward(self, x:torch.Tensor):
        return self.projection_head(x)

class WM811K_Supervised_Head(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.classifier = nn.Linear(cfg.embedding_size, cfg.num_classes)
        self.num_classes = cfg.num_classes

    def forward(self, x:torch.Tensor):
        return self.classifier(x)

class SimCLRLoss(nn.Module):
    '''
    Taken from: https://github.com/google-research/simclr/blob/master/objective.py

    Converted to pytorch, and decomposed for a clearer understanding.

    Args:
        init:
            batch_size (integer): Number of datasamples per batch.

            normalize (bool, optional): Whether to normalise the reprentations.
                (Default: True)

            temperature (float, optional): The temperature parameter of the
                NT_Xent loss. (Default: 1.0)
        forward:
            z_i (Tensor): Reprentation of view 'i'

            z_j (Tensor): Reprentation of view 'j'
    Returns:
        loss (Tensor): NT_Xent loss between z_i and z_j
    '''

    def __init__(self, batch_size, normalize=True, temperature=1.0):
        super(SimCLRLoss, self).__init__()

        self.temperature = temperature
        self.normalize = normalize

        self.register_buffer('labels', torch.zeros(batch_size * 2).long())

        self.register_buffer('mask', torch.ones(
            (batch_size, batch_size), dtype=bool).fill_diagonal_(0))

    def forward(self, z_i, z_j):

        if self.normalize:
            z_i_norm = F.normalize(z_i, p=2, dim=-1)
            z_j_norm = F.normalize(z_j, p=2, dim=-1)

        else:
            z_i_norm = z_i
            z_j_norm = z_j

        bsz = z_i_norm.size(0)

        ''' Note: **
        Cosine similarity matrix of all samples in batch:
        a = z_i
        b = z_j
         ____ ____
        | aa | ab |
        |____|____|
        | ba | bb |
        |____|____|

        Postives:
        Diagonals of ab and ba '\'

        Negatives:
        All values that do not lie on leading diagonals of aa, bb, ab, ba.
        '''

        # Cosine similarity between all views
        logits_aa = torch.mm(z_i_norm, z_i_norm.t()) / self.temperature
        logits_bb = torch.mm(z_j_norm, z_j_norm.t()) / self.temperature
        logits_ab = torch.mm(z_i_norm, z_j_norm.t()) / self.temperature
        logits_ba = torch.mm(z_j_norm, z_i_norm.t()) / self.temperature

        # Compute Postive Logits
        logits_ab_pos = logits_ab[torch.logical_not(self.mask)]
        logits_ba_pos = logits_ba[torch.logical_not(self.mask)]

        # Compute Negative Logits
        logit_aa_neg = logits_aa[self.mask].reshape(bsz, -1)
        logit_bb_neg = logits_bb[self.mask].reshape(bsz, -1)
        logit_ab_neg = logits_ab[self.mask].reshape(bsz, -1)
        logit_ba_neg = logits_ba[self.mask].reshape(bsz, -1)

        # Postive Logits over all samples
        pos = torch.cat((logits_ab_pos, logits_ba_pos)).unsqueeze(1)

        # Negative Logits over all samples
        neg_a = torch.cat((logit_aa_neg, logit_ab_neg), dim=1)
        neg_b = torch.cat((logit_ba_neg, logit_bb_neg), dim=1)

        neg = torch.cat((neg_a, neg_b), dim=0)

        # Compute cross entropy
        logits = torch.cat((pos, neg), dim=1)

        loss = F.cross_entropy(logits, self.labels)

        return loss
