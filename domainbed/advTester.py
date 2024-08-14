import collections
import json
import time
import copy
from pathlib import Path
import os
import numpy as np
import torch
import torch.utils.data

from domainbed.dataset_utils import build_adv_dataset
from domainbed import algorithms
from domainbed.evaluator import Evaluator
from domainbed.lib import misc
from domainbed.lib import swa_utils
from domainbed import swad as swad_module

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def json_handler(v):
    if isinstance(v, (Path, range)):
        return str(v)
    raise TypeError(f"`{type(v)}` is not JSON Serializable")



def advTest(args, hparams, n_steps, checkpoint_freq, logger):
    logger.info("")

    #######################################################
    # setup dataset & loader
    #######################################################
    data_info, dataset_clean, dataset_adv = build_adv_dataset(args)

    data_loader_clean = torch.utils.data.DataLoader(
        dataset_clean, sampler=torch.utils.data.SequentialSampler(dataset_clean),
        batch_size=hparams["batch_size"],
        num_workers=hparams["num_workers"],
        drop_last=False
    )

    data_loader_adv = torch.utils.data.DataLoader(
        dataset_adv, sampler=torch.utils.data.SequentialSampler(dataset_adv),
        batch_size=hparams["batch_size"],
        num_workers=hparams["num_workers"],
        drop_last=False
    )

    #######################################################
    # setup and load algorithm (model)
    #######################################################
    hparams["steps"] = args.steps
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(
        data_info.num_classes,
        hparams
    )
    model_path = os.path.join('./save_model', args.task, args.algorithm, args.name, "best_model", "model.pkl")
    if hparams["swad"]:
        model_path = os.path.join('./save_model', args.task, args.algorithm, args.name, "swad", "model.pkl")
    algorithm.load_state_dict(torch.load(model_path)["model"])
    algorithm.to(device)

    n_params = sum([p.numel() for p in algorithm.parameters()])
    logger.info("# of params = %d" % n_params)

    #######################################################
    # start test
    #######################################################
    algorithm.eval()
    ACTC = 0
    ALD = 0
    RGB_denominator = 0
    RGB_member = 0
    use_num = 0
    with torch.no_grad():
        for clean_batch, adv_batch in zip(data_loader_clean, data_loader_adv):
            assert clean_batch["y"].sum() == adv_batch["y"].sum()
            clean_x, y, ori_clean_x = clean_batch["x"].cuda(), clean_batch["y"].cuda(), clean_batch["ori_x"]
            adv_x, ori_adv_x, blur_adv_x = adv_batch["x"].cuda(), adv_batch["ori_x"], adv_batch["blur_x"].cuda()

            clean_pred = algorithm(clean_x)
            adv_pred = algorithm(adv_x)
            blur_adv_pred = algorithm(blur_adv_x)
            soft_clean_pred = torch.softmax(clean_pred, dim=1)
            soft_adv_pred = torch.softmax(adv_pred, dim=1)
            hard_clean_pred = soft_clean_pred.data.max(1)[1]
            hard_adv_pred = soft_adv_pred.data.max(1)[1]
            hard_blur_adv_pred = blur_adv_pred.data.max(1)[1]

            weight = (hard_clean_pred == y).float() #weight is used to filter the misclassification of the model on clean samples
            use_num += (hard_clean_pred == y).float().sum() # only clean samples that are correctly classified by the model will be used to calculate the score

            ACTC += (soft_adv_pred.gather(1, y.unsqueeze(1)).reshape(-1) * weight).sum()
            inner_ALD = 0
            for x1, x2, yi, true_yi, wi in zip(ori_clean_x, ori_adv_x, hard_adv_pred, y, weight):
                distance = abs(x1-x2).max()#np.max(np.max(abs(x1-x2), axis=2),axis=1).max()
                correct = (yi == true_yi)
                if correct:
                    distance = 64
                if distance > 64:
                    print("all score is set to zero!!!")
                    raise
                inner_ALD += wi * distance / 64
            ALD += inner_ALD

            RGB_denominator += ((hard_adv_pred != y) * weight).sum()
            RGB_member += ((hard_adv_pred != y) * (hard_blur_adv_pred != y) * weight).sum()

    ACTC /= use_num
    ALD /= use_num
    RGB = RGB_member / (RGB_denominator + 1e-6)
    final_score = ((1-ACTC) + (1-ALD) + RGB) / 3

    logger.info(f"ACTC_score = {1 - ACTC:.3%}")
    logger.info(f"ALD_score = {1 - ALD:.3%}")
    logger.info(f"RGB_score = {RGB:.3%}")
    logger.info(f"final_score = {final_score:.3%}")



