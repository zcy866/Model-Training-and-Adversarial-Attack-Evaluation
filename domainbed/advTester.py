import collections
import json
import math
import time
import copy
from pathlib import Path
import os
import numpy as np
import torch
import torch.utils.data
import csv

from domainbed.dataset_utils import build_adv_dataset, default_loader
from domainbed import algorithms

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def json_handler(v):
    if isinstance(v, (Path, range)):
        return str(v)
    raise TypeError(f"`{type(v)}` is not JSON Serializable")

def balance_analysis(args, hparams, data_loader, algorithm) -> dict:
    #######################################################
    # start test
    #######################################################
    algorithm.eval()
    confusion_matrix = torch.zeros([hparams["num_classes"], hparams["num_classes"]])
    with torch.no_grad():
        for batch in data_loader:
            x, y = batch["x"].cuda(), batch["y"].cuda()
            pred = algorithm(x)
            soft_pred = torch.softmax(pred, dim=-1)
            hard_pred = soft_pred.data.max(1)[1]

            for py, ty in zip(hard_pred, y):
                confusion_matrix[py][ty] += 1
    confusion_matrix = confusion_matrix.cpu().numpy()
    results = {}
    results["info"] = "Each row means a prediction class, each column means a true class."
    #results["confusion_matrix"] = confusion_matrix
    results["accuracy_each_class"] = [confusion_matrix[i][i] / sum(confusion_matrix[:, i]) for i in range(hparams["num_classes"])]
    results["sorted_class"] = sorted(list(enumerate(results["accuracy_each_class"])), key=lambda x:x[1])
    return results

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

    if hparams["use_beta_esm"]:
        algorithm = algorithm.avg_model
    n_params = sum([p.numel() for p in algorithm.parameters()])
    logger.info("# of params = %d" % n_params)

    #######################################################
    # start test
    #######################################################
    algorithm.eval()
    ACTC = 0
    ALD = 0
    avg_ALD_for_all = 0
    RGB_denominator = 0
    RGB_member = 0
    use_num = 0
    over_confident_num = 0
    each_sample_score = [["path", "ACTC", "ALD", "RGB", "overall"]] #used for recording score for each sample
    with torch.no_grad():
        for clean_batch, adv_batch in zip(data_loader_clean, data_loader_adv):
            assert clean_batch["y"].sum() == adv_batch["y"].sum()
            clean_x, y, ori_clean_x_path = clean_batch["x"].cuda(), clean_batch["y"].cuda(), clean_batch["ori_x_path"]
            adv_x, ori_adv_x_path, blur_adv_x = adv_batch["x"].cuda(), adv_batch["ori_x_path"], adv_batch["blur_x"].cuda()

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

            inner_ALD = 0
            inner_i = 0
            for x1_path, x2_path, adv_yi, true_yi, wi, adv_pi, blur_adv_yi in \
                    zip(ori_clean_x_path, ori_adv_x_path, hard_adv_pred, y, weight, soft_adv_pred, hard_blur_adv_pred):
                x1 = np.array(default_loader(x1_path))#(Image.open(x1_path))
                x2 = np.array(default_loader(x2_path))
                x1 = torch.tensor(x1).float()
                x2 = torch.tensor(x2).float()
                distance = abs(x1 - x2).max().item()  # np.max(np.max(abs(x1-x2), axis=2),axis=1).max()
                correct = (adv_yi == true_yi)
                if correct:
                    distance = 64
                if distance > 64.1:
                    print(distance)
                    print(x1_path)
                    print(x2_path)
                    soft_adv_pred[inner_i][true_yi] = 1
                    adv_pi[true_yi] = 1
                    hard_blur_adv_pred[inner_i] = true_yi
                    blur_adv_yi = true_yi
                    print("a sample is over-perturbed")
                    raise

                ald_score = 1 - distance / 64
                actc_score = 1 - adv_pi[true_yi].item()
                rgb_score = ((adv_yi != true_yi).float() * (blur_adv_yi != true_yi).float() / ((adv_yi != true_yi).float() + 1e-6)).item()
                each_sample_score.append([x2_path, actc_score, ald_score, rgb_score, (actc_score+ald_score+rgb_score)/3])

                inner_ALD += wi * distance / 64
                avg_ALD_for_all += wi * math.sqrt(((x1 - x2)**2).sum())
                inner_i += 1
            ALD += inner_ALD
            ACTC += (soft_adv_pred.gather(1, y.unsqueeze(1)).reshape(-1) * weight).sum().item()
            RGB_denominator += ((hard_adv_pred != y) * weight).sum().item()
            RGB_member += ((hard_adv_pred != y) * (hard_blur_adv_pred != y) * weight).sum().item()

    ACTC /= use_num
    ALD /= use_num
    avg_ALD_for_all /= use_num
    RGB = RGB_member / (RGB_denominator + 1e-6)
    final_score = ((1-ACTC) + (1-ALD) + RGB) / 3

    use_ratio = use_num / len(dataset_clean)

    hparams["num_classes"] = data_info.num_classes
    attack_balance = balance_analysis(args, hparams, data_loader_adv, algorithm)
    logger.info(f"Only clean samples that are correctly classified by the model will be used to calculate the score")
    logger.info(f"use_num = "+str(use_num))
    logger.info(f"use_ratio = {use_ratio:.3%}")
    logger.info(f"ACTC_score = {1 - ACTC:.3%}")
    logger.info(f"ALD_score = {1 - ALD:.3%}")
    logger.info(f"RGB_score = {RGB:.3%}")
    logger.info(f"final_score = {final_score:.3%}")
    logger.info(f"avg_l2_distance = "+str(avg_ALD_for_all.item()))
    for key, val in attack_balance.items():
        logger.info(key)
        logger.info(val)
    output_name = 'output.csv'
    output_path = os.path.join(args.each_sample_score_record_folder, output_name)
    logger.info(f"score of each sample are recorded in " + str(output_path))
    if not os.path.exists(args.each_sample_score_record_folder):
        os.makedirs(args.each_sample_score_record_folder)
    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(each_sample_score)
    return [each_sample_score[-1] for i in range(len(each_sample_score))]



