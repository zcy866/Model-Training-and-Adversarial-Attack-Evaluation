import argparse
import collections
import random
import sys
from pathlib import Path
import os
import numpy as np
import PIL
import torch
import torchvision
from sconf import Config
import shutil

from domainbed import hparams_registry
from domainbed.lib import misc
from domainbed.lib.writers import get_writer
from domainbed.lib.logger import Logger
from domainbed.dataset_utils import build_ana_dataset
from domainbed import algorithms

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
    results["confusion_matrix"] = confusion_matrix
    results["accuracy_each_class"] = [confusion_matrix[i][i] / sum(confusion_matrix[:, i]) for i in range(hparams["num_classes"])]
    results["sorted_class"] = sorted(list(enumerate(results["accuracy_each_class"])), key=lambda x:x[1])
    return results

def acquire_classification_wrong(args, hparams, data_loader, algorithm):
    #######################################################
    # start test
    #######################################################
    algorithm.eval()
    each_class_wrong_samples_path = [[] for _ in range(hparams["num_classes"])]
    with torch.no_grad():
        for batch in data_loader:
            x, y, path = batch["x"].cuda(), batch["y"].cuda(), batch["path"]
            pred = algorithm(x)
            soft_pred = torch.softmax(pred, dim=-1)
            hard_pred = soft_pred.data.max(1)[1]

            for py, ty, pp in zip(hard_pred, y, path):
                if py != ty:
                    each_class_wrong_samples_path[ty].append(pp)
    results = {}
    for i in range(hparams["num_classes"]):
        results[str(i)] = each_class_wrong_samples_path[i]
    return results

def clear_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))

def main():
    parser = argparse.ArgumentParser(description="Sample Attack Competition")
    parser.add_argument("name", type=str)
    parser.add_argument("configs", nargs="*")
    parser.add_argument("--data_dir", type=str, default="./datasets")
    parser.add_argument("--task", type=str, default="attack_dataset")
    parser.add_argument("--data", type=str, default="train_train_val_clean")
    parser.add_argument("--img_model", type=str, default="ResNet-50")
    #ResNet-50, mae-B, swin_transformer-B, swin_transformer-S, convnext-S, convnext-B, DINO_V2_VIT-S, DINO_V2_VIT-B, CLIP_img_encoder_VIT-L, CLIP_img_encoder_VIT-B
    parser.add_argument("--algorithm", type=str, default="ERM")
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
    parser.add_argument(
        "--blur_scale", type=float, default=0.1, help="Number of steps."
    )

    parser.add_argument("--prebuild_loader", action="store_true", help="Pre-build eval loaders")
    args, left_argv = parser.parse_known_args()

    # setup hparams
    hparams = hparams_registry.default_hparams(args.algorithm)
    hparams["img_model"] = args.img_model

    keys = ["config.yaml"] + args.configs
    keys = [open(key, encoding="utf8") for key in keys]
    hparams = Config(*keys, default=hparams)
    hparams.argv_update(left_argv)

    # setup debug
    if args.debug:
        args.checkpoint_freq = 5
        args.steps = 10
        args.name += "_debug"

    timestamp = misc.timestamp()
    args.unique_name = f"{timestamp}_{args.name}"

    # path setup
    args.work_dir = Path(".")
    args.data_dir = Path(args.data_dir)

    args.out_root = args.work_dir / Path("train_output") / args.task
    args.out_dir = args.out_root / args.unique_name
    args.out_dir.mkdir(exist_ok=True, parents=True)

    writer = get_writer(args.out_root / "runs" / args.unique_name)
    logger = Logger.get(args.out_dir / "log.txt")
    if args.debug:
        logger.setLevel("DEBUG")
    cmd = " ".join(sys.argv)
    logger.info(f"Command :: {cmd}")

    logger.nofmt("Environment:")
    logger.nofmt("\tPython: {}".format(sys.version.split(" ")[0]))
    logger.nofmt("\tPyTorch: {}".format(torch.__version__))
    logger.nofmt("\tTorchvision: {}".format(torchvision.__version__))
    logger.nofmt("\tCUDA: {}".format(torch.version.cuda))
    logger.nofmt("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    logger.nofmt("\tNumPy: {}".format(np.__version__))
    logger.nofmt("\tPIL: {}".format(PIL.__version__))

    # Different to DomainBed, we support CUDA only.
    assert torch.cuda.is_available(), "CUDA is not available"

    logger.nofmt("Args:")
    for k, v in sorted(vars(args).items()):
        logger.nofmt("\t{}: {}".format(k, v))

    logger.nofmt("HParams:")
    for line in hparams.dumps().split("\n"):
        logger.nofmt("\t" + line)

    if args.show:
        exit()

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.deterministic
    torch.backends.cudnn.benchmark = not args.deterministic

    logger.info("")

    #######################################################
    # setup dataset & loader
    #######################################################
    data_info, dataset_train, dataset_val = build_ana_dataset(args)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=torch.utils.data.SequentialSampler(dataset_train),
        batch_size=hparams["batch_size"],
        num_workers=hparams["num_workers"],
        drop_last=False
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=torch.utils.data.SequentialSampler(dataset_val),
        batch_size=hparams["batch_size"],
        num_workers=hparams["num_workers"],
        drop_last=False
    )
    hparams["num_classes"] = data_info.num_classes
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
    algorithm.cuda()

    if hparams["use_beta_esm"]:
        algorithm = algorithm.avg_model
    n_params = sum([p.numel() for p in algorithm.parameters()])
    logger.info("# of params = %d" % n_params)

    logger.info("balance analysis of training dataset")
    results = balance_analysis(args, hparams, data_loader_train, algorithm)
    for key, val in results.items():
        logger.info(key)
        logger.info(val)

    logger.info("balance analysis of validation dataset")
    results = balance_analysis(args, hparams, data_loader_val, algorithm)
    for key, val in results.items():
        logger.info(key)
        logger.info(val)

    save_path = "./ana_dataset"
    logger.info("the paths of mis-classification samples are saved in " + str(save_path))
    logger.info("acquire mis-classification sample paths of training dataset")
    results = acquire_classification_wrong(args, hparams, data_loader_train, algorithm)
    for key, val in results.items():
        logger.info(key)
        logger.info(val)
        if not os.path.exists(os.path.join(save_path, "train", key)):
            os.makedirs(os.path.join(save_path, "train", key))
        else:
            clear_directory(os.path.join(save_path, "train", key))
        for i, sample in enumerate(val):
            shutil.copy(sample, os.path.join(save_path, "train", key, str(i)+'.jpg'))

    logger.info("acquire mis-classification sample paths of validation dataset")
    results = acquire_classification_wrong(args, hparams, data_loader_val, algorithm)
    for key, val in results.items():
        logger.info(key)
        logger.info(val)
        if not os.path.exists(os.path.join(save_path, "val", key)):
            os.makedirs(os.path.join(save_path, "val", key))
        else:
            clear_directory(os.path.join(save_path, "val", key))
        for i, sample in enumerate(val):
            shutil.copy(sample, os.path.join(save_path, "val", key, str(i)+'.jpg'))




if __name__ == "__main__":
    main()
