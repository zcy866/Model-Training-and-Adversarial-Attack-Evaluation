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
import random

from domainbed import networks
from domainbed.lib.misc import random_pairs_of_minibatches, split_meta_train_test
from domainbed.optimizers import get_optimizer, build_scheduler
from domainbed.utils import GeneralMovingAverage, ModelEMA
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
        self.output_dim = self.featurizer.output_dim
        self.classifier = nn.Linear(self.output_dim, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        self.lr_schedule = build_scheduler(hparams["scheduler_name"], optimizer=self.optimizer, num_training_steps=hparams["steps"])

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

    def inner_step(self, all_x, all_y, step, **kwargs):
        if self.hparams["linear_ratio"] * self.hparams["steps"] < step:
            loss = F.cross_entropy(self.classifier(self.featurizer(all_x).detach()), all_y)
        else:
            loss = F.cross_entropy(self.classifier(self.featurizer(all_x)), all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_schedule.step()

        return {"loss": loss.item()}

class EMA_Distil(ERM):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, num_classes, hparams):
        super(EMA_Distil, self).__init__(num_classes, hparams)
        self.ema_model = ModelEMA(self.featurizer, decay=self.hparams["ema_decay"])

    def inner_step(self, all_x, all_y, **kwargs):
        loss = F.cross_entropy(self.classifier(self.featurizer(all_x)), all_y)
        self.optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            for p1, p2 in zip(self.featurizer.parameters(), self.ema_model.module.parameters()):
                p1.grad.add_(self.hparams["reg_weight"]*(p1-p2))
        self.optimizer.step()
        self.lr_schedule.step()

        self.ema_model.update_parameters(self.featurizer)

        return {"loss": loss.item()}


class SAM(ERM):
    """Sharpness-Aware Minimization
    """
    @staticmethod
    def norm(tensor_list: List[torch.tensor], p=2):
        """Compute p-norm for tensor list"""
        return torch.cat([x.flatten() for x in tensor_list]).norm(p)

    def inner_step(self, all_x, all_y, **kwargs):
        loss1 = F.cross_entropy(self.predict(all_x), all_y)

        # 1. eps(w) = rho * g(w) / g(w).norm(2)
        #           = (rho / g(w).norm(2)) * g(w)
        grad_w = autograd.grad(loss1, self.network.parameters())
        scale = self.hparams["rho"] / self.norm(grad_w)
        eps = [g * scale for g in grad_w]

        # 2. w' = w + eps(w)
        with torch.no_grad():
            for p, v in zip(self.network.parameters(), eps):
                p.add_(v)

        # 3. w = w - lr * g(w')
        loss = F.cross_entropy(self.predict(all_x), all_y)
        with open("sam_0.txt", "a") as f:
            f.write(str(loss.item()))
            f.write("\n")

        self.optimizer.zero_grad()
        loss.backward()
        # restore original network params
        with torch.no_grad():
            for p, v in zip(self.network.parameters(), eps):
                p.sub_(v)
        self.optimizer.step()

        return {"loss": loss.item(), "loss1": loss1.item()}

class Global_Reg(ERM):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, num_classes, hparams):
        super(Global_Reg, self).__init__(num_classes, hparams)
        self.ema_model1 = ModelEMA(self.featurizer, decay=self.hparams["ema_decay"])
        self.ema_model2 = ModelEMA(self.featurizer, decay=self.hparams["ema_decay"])
        self.register_buffer("ema_prototypes1", torch.rand([num_classes, self.output_dim]))
        self.ema_prototypes1 = F.normalize(self.ema_prototypes1, dim=-1)
        self.register_buffer("ema_prototypes2", torch.rand([num_classes, self.output_dim]))
        self.ema_prototypes2 = F.normalize(self.ema_prototypes2, dim=-1)
        self.frozen_model = copy.deepcopy(self.network)
        self.frozen_model.eval()

    def inner_step(self, all_x, all_y, step, **kwargs):
        eps = []
        with torch.no_grad():
            for p1, p2 in zip(self.network.parameters(), self.frozen_model.parameters()):
                teps = self.hparams["reg_weight"] * self.lr_schedule.get_lr()[0] * (p1 - p2)
                p1.sub_(teps)
                eps.append(teps)

        with torch.no_grad():
            self.ema_model1.eval()
            self.ema_model2.eval()

            ema_fea1 = self.ema_model1(all_x)
            ema_fea2 = self.ema_model2(all_x)
            norm_ema_fea1 = F.normalize(ema_fea1, dim=-1)
            norm_ema_fea2 = F.normalize(ema_fea2, dim=-1)

            score1 = torch.sum(self.ema_prototypes1[all_y]*norm_ema_fea1, dim=-1)
            score2 = torch.sum(self.ema_prototypes2[all_y]*norm_ema_fea2, dim=-1)

            distil_fea = (score1 >= score2).float().unsqueeze(1) * ema_fea1 + (1 - (score1 >= score2).float()).unsqueeze(1) * ema_fea2
            neg_distil_fea = (score1 >= score2).float().unsqueeze(1) * ema_fea2 + (1 - (score1 >= score2).float()).unsqueeze(1) * ema_fea1

            update_step = step/self.hparams["steps"] * self.hparams["stop_update_step"] + (1-step/self.hparams["steps"]) * 0.99
            for fi1, fi2, yi in zip(norm_ema_fea1, norm_ema_fea2, all_y):
                self.ema_prototypes1[yi] = update_step * self.ema_prototypes1[yi] + (1-update_step)*fi1
                self.ema_prototypes2[yi] = update_step * self.ema_prototypes2[yi] + (1-update_step)*fi2
            self.ema_prototypes1 = F.normalize(self.ema_prototypes1, dim=-1)
            self.ema_prototypes2 = F.normalize(self.ema_prototypes2, dim=-1)

        fea = self.featurizer(all_x)
        pred = self.classifier(fea)
        cls_loss = F.cross_entropy(pred, all_y)
        align_loss = -(F.normalize(fea, dim=-1) * F.normalize(distil_fea, dim=-1)).sum(1).mean()
        loss = cls_loss + self.hparams["align_weight"] * align_loss
        self.optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            for p1, v in zip(self.network.parameters(), eps):
                p1.add_(v)

        self.optimizer.step()
        self.lr_schedule.step()

        if random.random() > 0.5:
            self.ema_model1.update_parameters(self.featurizer)
        else:
            self.ema_model2.update_parameters(self.featurizer)

        return {"loss": loss.item(), "cls_loss": cls_loss.item(), "align_loss": align_loss.item()}