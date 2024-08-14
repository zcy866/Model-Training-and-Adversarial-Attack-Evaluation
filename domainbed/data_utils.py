import torch
import numpy as np

from domainbed.datasets import datasets
from domainbed.lib import misc
from domainbed.datasets import transforms as DBT
from domainbed.data_loader import GlobalDataloader

def set_transfroms(dset, data_type, hparams, algorithm_class=None):
    """
    Args:
        data_type: ['train', 'valid', 'test', 'mnist']
    """
    assert hparams["data_augmentation"]

    additional_data = False
    if data_type == "train":
        dset.transforms = {"x": DBT.aug}
        additional_data = True
    elif data_type == "valid":
        if hparams["val_augment"] is False:
            dset.transforms = {"x": DBT.basic}
        else:
            # Originally, DomainBed use same training augmentation policy to validation.
            # We turn off the augmentation for validation as default,
            # but left the option to reproducibility.
            dset.transforms = {"x": DBT.aug}
    elif data_type == "test":
        dset.transforms = {"x": DBT.basic}
    elif data_type == "mnist":
        # No augmentation for mnist
        dset.transforms = {"x": lambda x: x}
    else:
        raise ValueError(data_type)

    if additional_data and algorithm_class is not None:
        for key, transform in algorithm_class.transforms.items():
            dset.transforms[key] = transform

class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""

    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
        self.transforms = {}

        self.direct_return = isinstance(underlying_dataset, _SplitDataset)

    def __getitem__(self, key):
        if self.direct_return:
            return self.underlying_dataset[self.keys[key]]

        x, y = self.underlying_dataset[self.keys[key]]
        ret = {"y": y}

        for key, transform in self.transforms.items():
            ret[key] = transform(x)

        return ret

    def __len__(self):
        return len(self.keys)


def get_data_for_classification(test_envs, args, hparams, algorithm_class=None):
    """Get dataset and split."""
    is_mnist = "MNIST" in args.dataset
    dataset = vars(datasets)[args.dataset](args.data_dir)
    #  if not isinstance(dataset, MultipleEnvironmentImageFolder):
    #      raise ValueError("SMALL image datasets are not implemented (corrupted), for transform.")

    in_split_keys = []
    out_splits = []
    test_in_splits = []
    for env_i, env in enumerate(dataset):
        # The split only depends on seed_hash (= trial_seed).
        # It means that the split is always identical only if use same trial_seed,
        # independent to run the code where, when, or how many times.
        n = int(len(env) * args.holdout_fraction)
        assert n <= len(env)
        keys = list(range(len(env)))
        np.random.RandomState(misc.seed_hash(args.trial_seed, env_i)).shuffle(keys)
        keys_1 = keys[:n]
        keys_2 = keys[n:]
        in_split_keys.append(keys_2)

        in_ = _SplitDataset(env, keys_2)
        out = _SplitDataset(env, keys_1)

        if env_i in test_envs:
            in_type = "test"
            out_type = "test"
        else:
            in_type = "train"
            out_type = "valid"

        if is_mnist:
            in_type = "mnist"
            out_type = "mnist"

        set_transfroms(in_, in_type, hparams, algorithm_class)
        set_transfroms(out, out_type, hparams, algorithm_class)

        if hparams["class_balanced"]:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
        else:
            in_weights, out_weights = None, None
        test_in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))

    in_list_path = []
    num_group_type = 2
    group_type_names = ["y", "domain_idx"]
    group_nums = [dataset.num_classes, len(dataset) - len(test_envs)]
    train_transform = DBT.aug
    if is_mnist:
        train_transform = lambda x: x
    batch_size = (len(dataset) - len(test_envs)) * hparams["batch_size"]
    num_workers = dataset.N_WORKERS
    fix_group_index = 1
    domain_idx = 0
    for env_i, env in enumerate(dataset):
        if env_i in test_envs:
            continue
        keys = in_split_keys[env_i]
        for key in keys:
            path, target = env.samples[key]
            in_list_path.append([path, target, domain_idx])
        domain_idx += 1
    in_loader = GlobalDataloader(list_path=in_list_path, num_group_type=num_group_type,
                                 group_type_names=group_type_names, group_nums=group_nums,
                                 transform=train_transform, batch_size=batch_size,
                                 num_workers=num_workers, fix_group_index=fix_group_index)

    return dataset, in_loader, out_splits, test_in_splits
