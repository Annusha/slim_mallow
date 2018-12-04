#!/usr/bin/env python

"""Module with Embedding and RankLoss classes. First one is linear embedding
with weights for features and so-called anchor point which roughly represents
position of clusters."""

__author__ = 'Anna Kukleva'
__date__ = 'August 2018'

import torch.nn as nn
import torch

from utils.arg_pars import opt


class Embedding(nn.Module):
    """Linear embedding"""
    def __init__(self, embed_dim, feature_dim, n_subact):
        super(Embedding, self).__init__()
        torch.manual_seed(opt.seed)

        self._embed_dim = embed_dim
        self._n_subact = n_subact
        self._W_a = nn.Linear(embed_dim, n_subact, bias=False)
        self._W_f = nn.Linear(feature_dim, embed_dim, bias=False)
        self._init_weights()

    def forward(self, x):
        x = self._W_f(x)
        x = self._W_a(x)
        return x

    def embedded(self, x):
        return self._W_f(x)

    def anchors(self):
        # mask = torch.ones((self._n_subact, self._embed_dim))
        return self._W_a.weight

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)


class RankLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(RankLoss, self).__init__()
        self._margin = margin

    def forward(self, features, k):
        result = torch.sub(features.t(), features[k.byte()]).t()
        result = result + self._margin
        zeros = torch.zeros(result.size()).cuda()
        result = torch.max(zeros, result)
        result = torch.sum(result)
        result = result - self._margin * features.size()[0]
        result = result # / features.size()[0]
        return result



