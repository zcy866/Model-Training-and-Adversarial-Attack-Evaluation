import random

from torch.utils.data import Sampler, Dataset, DataLoader, SequentialSampler
import torch
import numpy as np
import os.path as osp
from PIL import Image
import time
import queue
import copy

class GlobalDataloader():
    def __init__(self, list_path, num_group_type, group_type_names, group_nums, transform, batch_size, num_workers, fix_group_index=-1):
        self.list_path = list_path
        self.num_group_type = num_group_type
        self.group_type_names = group_type_names
        self.group_nums = group_nums
        self.batch_size = batch_size
        self.all_group_nums = 1
        for ele in group_nums:
            self.all_group_nums *= ele
        self.fix_group_index = fix_group_index #Make sure all groups of that type are selected each time
        if fix_group_index >= 0 and batch_size % group_nums[fix_group_index] != 0:
            raise "group number of fix_group_index" + str(fix_group_index) + "is less than batch_size" + str(batch_size)

        #init
        self.global_dataset_init()
        self.all_datasets = []

        self.train_sampler = TempSampler() #Sample in order
        self.train_dataset = TempDataset(group_type_names, transform)
        self.data_loader = DataLoader(
            self.train_dataset,
            sampler=self.train_sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True)
        self.data_loader_iter = None

    def global_dataset_init(self):
        self.img_ids = [l[0] for l in self.list_path]
        self.img_ids = np.array(self.img_ids, dtype=object)
        self.group_ids = dict()
        for i in range(len(self.group_type_names)):
            name = self.group_type_names[i]
            self.group_ids[name] = [l[i+1] for l in self.list_path]
            self.group_ids[name] = np.array(self.group_ids[name])
            if max(self.group_ids[name]) != self.group_nums[i]-1:
                raise "the number of group type " + name + "is" + str(self.group_nums[i]) + ", but the maximum number is " + str(max(self.group_ids[name]))
            assert min(self.group_ids[name]) == 0
        #split full dataset to disjoint sub-datasets
        self.gloabal_loader = self.split_dataset(list(range(len(self.img_ids))), 0)
        self.gloabal_loader = self.flatten_list(self.gloabal_loader, self.num_group_type)
        #self.gloabal_loader_iter = [iter(tloader) for tloader in self.gloabal_loader]
        if self.fix_group_index >= 0:
            self.group_global_loader = []
            self.group_global_loader_iter = []
            for i in range(self.group_nums[self.fix_group_index]):
                self.group_global_loader.append([])
                self.group_global_loader_iter.append([])
            intervals = 1
            for i in range(self.fix_group_index+1, len(self.group_nums)):
                intervals *= self.group_nums[i]
            tidx = 0
            while tidx < self.all_group_nums:
                for i in range(self.group_nums[self.fix_group_index]):
                    if intervals == 1:
                        self.group_global_loader[i].append(self.gloabal_loader[tidx])
                        self.group_global_loader_iter[i].append(randomQueue(self.gloabal_loader[tidx]))
                        #self.group_global_loader_iter[i].append(iter(self.gloabal_loader[tidx]))
                    else:
                        self.group_global_loader += self.gloabal_loader[tidx: tidx+intervals]
                        raise "no iter"
                    tidx += intervals
            for i in range(self.group_nums[self.fix_group_index]):
                assert len(self.group_global_loader[i]) == self.all_group_nums/self.group_nums[self.fix_group_index]

    def split_dataset(self, all_index, tier):
        if tier == self.num_group_type:
            assert len(all_index) > 1
            #tdataset = unitDataset(all_index)
            #tloader = InfiniteDataLoader(
            #    tdataset,
            #    batch_size=1,
            #    num_workers=1)
            return all_index
        else:
            #Group the input indices according to the specified dimension
            new_all_index = []
            for i in range(self.group_nums[tier]):
                new_all_index.append([])
            tier_name = self.group_type_names[tier]
            spe_group = self.group_ids[tier_name]
            for ele in all_index:
                id = spe_group[ele]
                new_all_index[id].append(ele)
            #Further group each group according to the next dimension
            final_all_index = []
            for i in range(self.group_nums[tier]):
                final_all_index.append(self.split_dataset(new_all_index[i], tier+1))
            return final_all_index

    def flatten_list(self, in_list, next_size):
        if next_size == 0:
            return [in_list]
        else:
            raw_list = []
            for i in range(len(in_list)):
                raw_list += self.flatten_list(in_list[i], next_size-1)
            return raw_list

    def reload(self, focus_group_index_stream, diversity_ratio_stream):
        index_stream = []
        for focus_group_index, diversity_ratio in zip(focus_group_index_stream, diversity_ratio_stream):
            index = self.simulate(focus_group_index, diversity_ratio)
            index_stream += index
        index_stream = np.array(index_stream)
        tgroups_ids = dict()
        for name in self.group_ids.keys():
            tgroups_ids[name] = self.group_ids[name][index_stream]
        self.train_dataset.load(self.img_ids[index_stream], tgroups_ids)
        self.train_sampler.load(self.train_dataset)
        self.data_loader_iter = iter(self.data_loader)

    def simulate(self, focus_group_index, diversity_ratio):
        if self.fix_group_index >= 0:

            per_base_num = self.batch_size // self.group_nums[self.fix_group_index]
            #sample based on focus_group_index
            sample_goal = np.arange(self.group_nums[focus_group_index])
            weight = 0.99 + 0.02 * sample_goal / self.group_nums[focus_group_index]
            weight = weight ** (100*(1.-diversity_ratio))
            weight /= sum(weight)
            shuffle_weight = weight
            np.random.shuffle(shuffle_weight)
            samples = np.random.choice(sample_goal, per_base_num, replace=True, p=shuffle_weight)
            #get image index
            all_index = []
            for ele in samples:
                for i in range(self.group_nums[self.fix_group_index]):
                    tidx = next(self.group_global_loader_iter[i][ele])
                    all_index.append(tidx)
            assert len(all_index) == self.batch_size

            return all_index
        else:
            raise "Unrealized multi-group random sampling"

    def get_next(self):
        return next(self.data_loader_iter)
    '''
    def get_next(self, focus_group_index, diversity_ratio):
        #sample dataset
        #diversity_ratio --- Control the diversity of groups in a batch
        if self.fix_group_index >= 0:

            per_base_num = self.batch_size // self.group_nums[self.fix_group_index]
            #sample based on focus_group_index
            sample_goal = np.arange(self.group_nums[focus_group_index])
            weight = 0.99 + 0.02 * sample_goal / self.group_nums[focus_group_index]
            weight = weight ** (100*(1.-diversity_ratio))
            weight /= sum(weight)
            shuffle_weight = weight
            np.random.shuffle(shuffle_weight)
            samples = np.random.choice(sample_goal, per_base_num, replace=True, p=shuffle_weight)

            #get image index
            all_index = []
            for ele in samples:
                for i in range(self.group_nums[self.fix_group_index]):
                    tidx = next(self.group_global_loader_iter[i][ele])
                    all_index.append(tidx)
            assert len(all_index) == self.batch_size

            #load data

            tgroups_ids = dict()
            for name in self.group_ids.keys():
                tgroups_ids[name] = self.group_ids[name][all_index]
            self.train_dataset.load(self.img_ids[all_index], tgroups_ids)
            self.train_sampler.load(self.train_dataset)
            data_loader_iter = iter(self.data_loader)
            batch = next(data_loader_iter)
            assert len(batch["x"]) == self.batch_size
            return batch
        else:
            raise "Unrealized multi-group random sampling"
    '''


class TempSampler(Sampler):
    def __init__(self):
        self.data_source = None #dataset

    def load(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)

class TempDataset(Dataset):
    def __init__(self, group_type_names, transform):
        self.img_ids = None
        self.group_ids = dict()
        self.group_type_names = group_type_names
        for name in group_type_names:
            self.group_ids[name] = None
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def load(self, img_ids, group_ids):
        self.img_ids = img_ids
        self.group_ids = group_ids

    def __getitem__(self, index):
        if self.img_ids is None:
            raise "no data is load in TempDataset"
        name = self.img_ids[index]
        image = Image.open(name).convert('RGB')
        image = self.transform(image)
        ret = {"x": image}
        for name in self.group_type_names:
            ret[name] = self.group_ids[name][index]
        return ret

class unitDataset(Dataset):
    def __init__(self, inner_index):
        self.inner_index = inner_index

    def __len__(self):
        return len(self.inner_index)

    def load(self, inner_index):
        self.inner_index = inner_index

    def __getitem__(self, index):
        return self.inner_index[index]

class unitRandomSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def load(self, data_source):
        self.data_source = data_source
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        generator = self.generator
        for _ in range(self.num_samples // n):
            yield from torch.randperm(n, generator=generator).tolist()
        yield from torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]

    def __len__(self) -> int:
        return self.num_samples

class InfiniteDataLoader:
    def __init__(self, dataset, batch_size, num_workers):
        super().__init__()
        self.dataset = dataset

        sampler = torch.utils.data.RandomSampler(dataset)

        batch_sampler = torch.utils.data.BatchSampler(
            sampler, batch_size=batch_size, drop_last=False
        )

        self._infinite_iterator = iter(
            torch.utils.data.DataLoader(
                dataset,
                num_workers=num_workers,
                batch_sampler=_InfiniteSampler(batch_sampler),
            )
        )

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError

class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch

class randomQueue():
    def __init__(self, store_index, min_shuffle_size=100):
        self.store_index = store_index
        self.min_shuffle_size = min_shuffle_size
        self.reload()

    def reload(self):
        self.queue = []
        num = 0
        #load multiple shuffle outs in one reload
        while num < max(self.min_shuffle_size, len(self.store_index)):
            random.shuffle(self.store_index)
            self.queue += copy.deepcopy(self.store_index)
            num += len(self.store_index)

    def __next__(self):
        if len(self.queue) > 0:
            return self.queue.pop(0)
        else:
            self.reload()
            return self.queue.pop(0)

