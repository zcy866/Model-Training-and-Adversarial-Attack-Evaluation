# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np


def _hparams(algorithm, random_state):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    SMALL_IMAGES = ["Debug28", "RotatedMNIST", "ColoredMNIST"]

    hparams = {}

    hparams["num_workers"] = (4, 4)
    hparams["data_augmentation"] = (True, True)
    hparams["val_augment"] = (False, False)  # augmentation for in-domain validation set
    hparams["resnet_dropout"] = (0.0, random_state.choice([0.0, 0.1, 0.5]))
    hparams["class_balanced"] = (False, False)
    hparams["optimizer"] = ("adam", "adam")

    hparams["freeze_bn"] = (True, True)
    hparams["pretrained"] = (True, True)  # only for ResNet

    hparams["lr"] = (5e-5, 10 ** random_state.uniform(-5, -3.5))
    hparams["batch_size"] = (32, int(2 ** random_state.uniform(3, 5)))
    hparams["weight_decay"] = (1e-4, 0.0)
    hparams["use_beta_esm"] = (False, False)
    hparams["linear_ratio"] = (0.1, random_state.choice([0.05, 0.1, 0.15]))

    return hparams


def default_hparams(algorithm):
    dummy_random_state = np.random.RandomState(0)
    return {a: b for a, (b, c) in _hparams(algorithm, dummy_random_state).items()}


def random_hparams(algorithm, seed):
    random_state = np.random.RandomState(seed)
    return {a: c for a, (b, c) in _hparams(algorithm, random_state).items()}
