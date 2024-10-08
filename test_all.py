import argparse
import collections
import os.path
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import PIL
import torch
import torchvision
from sconf import Config

from domainbed import hparams_registry
from domainbed.lib import misc
from domainbed.lib.writers import get_writer
from domainbed.lib.logger import Logger
from domainbed.advTester import advTest


def main():
    parser = argparse.ArgumentParser(description="Sample Attack Competition")
    parser.add_argument("all_name", type=str)
    parser.add_argument("configs", nargs="*")
    parser.add_argument("--data_dir", type=str, default="./datasets")
    parser.add_argument("--adv_data_dir_name", type=str, default="all_adv_samples")
    parser.add_argument("--each_sample_score_record_folder", type=str, default="./score_record")
    parser.add_argument("--goal_adv_name", type=str, default="adv_samples")
    parser.add_argument("--task", type=str, default="attack_dataset")
    parser.add_argument("--data", type=str, default="train_train_val_clean")
    parser.add_argument("--all_img_model", type=str, default="ResNet-50")
    #ResNet-50, mae-B, swin_transformer-B, swin_transformer-S, convnext-S, convnext-B, DINO_V2_VIT-S, DINO_V2_VIT-B, CLIP_img_encoder_VIT-L, CLIP_img_encoder_VIT-B
    parser.add_argument("--all_algorithm", type=str, default="ERM")
    parser.add_argument(
        "--trial_seed",
        type=int,
        default=0,
        help="Trial number (used for seeding split_dataset and random_hparams).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for everything else")
    parser.add_argument(
        "--steps", type=int, default=10000, help="Number of steps."
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=1000,
        help="Checkpoint every N steps.",
    )
    parser.add_argument(
        "--model_save",
        type=int,
        default=3000,
        help="Checkpoint every N steps.",
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Run w/ debug mode")
    parser.add_argument(
        "--evalmode",
        default="fast",
        help="[fast, all]. if fast, ignore train_in datasets in evaluation time.",
    )
    parser.add_argument("--show", action="store_true", help="Show args and hparams w/o run")
    parser.add_argument("--training_mode", action="store_true", help="using training or testing")
    parser.add_argument(
        "--blur_scale", type=float, default=0.1, help="Number of steps."
    )

    parser.add_argument("--prebuild_loader", action="store_true", help="Pre-build eval loaders")
    args, left_argv = parser.parse_known_args()

    timestamp = misc.timestamp()
    args.unique_name = f"{timestamp}_{args.all_name}"

    # path setup
    args.work_dir = Path(".")
    args.data_dir = Path(args.data_dir)

    args.out_root = args.work_dir / Path("train_output") / args.task
    args.out_dir = args.out_root / args.unique_name
    args.out_dir.mkdir(exist_ok=True, parents=True)

    logger = Logger.get(args.out_dir / "log.txt")
    if args.debug:
        logger.setLevel("DEBUG")
    cmd = " ".join(sys.argv)
    logger.info(f"Command :: {cmd}")

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.deterministic
    torch.backends.cudnn.benchmark = not args.deterministic

    adv_path = os.listdir(os.path.join(args.data_dir, args.task, args.adv_data_dir_name))
    names = args.all_name.split(',')
    algos = args.all_algorithm.split(',')
    img_models = args.all_img_model.split(',')
    each_adv_score = []
    each_adv_path = []
    for adv_sample_name in adv_path:
        logger.info(f"adv_sample_name: "+str(adv_sample_name))
        args.adv_sample_name = adv_sample_name
        each_model_score = []
        adv_path = None
        for name, algo, img_model in zip(names, algos, img_models):
            args.name = name
            args.algorithm = algo
            args.img_model = img_model

            # setup hparams
            hparams = hparams_registry.default_hparams(args.algorithm)
            hparams["img_model"] = args.img_model

            keys = ["config.yaml"] + args.configs
            keys = [open(key, encoding="utf8") for key in keys]
            hparams = Config(*keys, default=hparams)
            hparams.argv_update(left_argv)

            inner_score, adv_path = advTest(args, hparams, logger)
            each_model_score.append(inner_score)
        each_model_score = np.array(each_model_score, dtype=float)
        each_sample_score = np.mean(each_model_score, axis=0)
        each_adv_score.append(each_sample_score)
        each_adv_path.append(adv_path)
        assert adv_path is not None

    if args.training_mode:
        each_adv_score = torch.tensor(np.array(each_adv_score))
        best_score, best_score_idx = each_adv_score.data.max(0)
        best_paths = []
        for i in range(len(best_score)):
            best_idx = best_score_idx[i]
            best_paths.append(each_adv_path[best_idx][i])

        adv_folder = os.path.join(args.data_dir, args.task, args.goal_adv_name)
        if os.path.exists(adv_folder):
            shutil.rmtree(adv_folder)
        for i in range(len(best_paths)):
            src_path = best_paths[i]
            file_folder_name, file_name = os.path.split(src_path)
            _, class_name = os.path.split(file_folder_name)
            tar_folder = os.path.join(args.data_dir, args.task, args.goal_adv_name, class_name)
            if not os.path.exists(tar_folder):
                os.makedirs(tar_folder)
            tar_path = os.path.join(args.data_dir, args.task, args.goal_adv_name, class_name, file_name)
            shutil.copy(src_path, tar_path)




if __name__ == "__main__":
    main()
