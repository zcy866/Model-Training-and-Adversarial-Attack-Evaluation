# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import queue
from typing import List
import math
import time
import timm.models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple, Type, Union, List
#  import higher
import scipy


from domainbed import networks
from domainbed.lib.misc import random_pairs_of_minibatches, split_meta_train_test
from domainbed.optimizers import get_optimizer, get_cosine_schedule
from domainbed.utils import GeneralMovingAverage
from domainbed.distil_base_networks.backbones import get_backbone

def to_minibatch(x, y):
    minibatches = list(zip(x, y))
    return minibatches

def flat_grad(grad_tuple):
    return torch.cat([p.view(-1) for p in grad_tuple]).unsqueeze(0)


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    transforms = {}

    def __init__(self, num_classes, hparams):
        super(Algorithm, self).__init__()
        self.num_classes = num_classes
        self.hparams = hparams

    def update(self, x, y, **kwargs):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.
        """
        raise NotImplementedError



    def predict(self, x):
        raise NotImplementedError

    def forward(self, x):
        return self.predict(x)

    def new_optimizer(self, parameters):
        optimizer = get_optimizer(
            self.hparams["optimizer"],
            parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        return optimizer

    def clone(self):
        clone = copy.deepcopy(self)
        clone.optimizer = self.new_optimizer(clone.network.parameters())
        clone.optimizer.load_state_dict(self.optimizer.state_dict())

        return clone


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, num_classes, hparams):
        super(ERM, self).__init__(num_classes, hparams)
        self.featurizer = get_backbone(hparams["img_model"], pretrained=hparams["pretrained"], freeze_bn=hparams["freeze_bn"])
        output_dim = self.featurizer.output_dim
        self.classifier = nn.Linear(output_dim, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        self.lr_schedule = get_cosine_schedule(self.optimizer, hparams["steps"])

        if self.hparams["use_beta_esm"]:
            beta_dist = scipy.stats.beta(0.5, 0.5)
            total_iter = hparams["steps"]
            weight_func = lambda it: beta_dist.pdf((it + 0.5) / (total_iter + 1))
            self.avg_model = GeneralMovingAverage(self.network, weight_func)

    def update(self, x, y, **kwargs):
        self.pre_step()
        return_item = self.inner_step(x, y, **kwargs)
        self.aft_step()
        return return_item

    def inner_step(self, all_x, all_y, **kwargs):
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_schedule.step()

        return {"loss": loss.item()}

    def pre_step(self):
        pass

    def aft_step(self):
        if self.hparams["use_beta_esm"]:
            self.avg_model.update(self.network)

    def predict(self, x):
        return self.network(x)

    def get_features(self, x):
        return self.featurizer(x)

class Linear_Prob(ERM):
    def __init__(self, num_classes, hparams):
        super(Linear_Prob, self).__init__(num_classes, hparams)
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            self.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def inner_step(self, all_x, all_y, **kwargs):
        loss = F.cross_entropy(self.classifier(self.featurizer(all_x).detach()), all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_schedule.step()

        return {"loss": loss.item()}

class LP_FT(ERM):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, num_classes, hparams):
        super(LP_FT, self).__init__(num_classes, hparams)

    def inner_step(self, all_x, all_y, step):
        if self.hparams["linear_ratio"] * self.hparams["steps"] < step:
            loss = F.cross_entropy(self.classifier(self.featurizer(all_x).detach()), all_y)
        else:
            loss = F.cross_entropy(self.classifier(self.featurizer(all_x)), all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_schedule.step()

        return {"loss": loss.item()}
