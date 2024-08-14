import argparse
import collections
import random
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
from domainbed.trainer import train


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

    n_steps = args.steps
    checkpoint_freq = args.checkpoint_freq
    train(args, hparams, n_steps, checkpoint_freq, logger)

if __name__ == "__main__":
    main()
