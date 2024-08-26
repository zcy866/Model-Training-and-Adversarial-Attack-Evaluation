import torchvision
import torch
from torchvision import transforms
import os
import numpy as np

class dataInfo():
    def __init__(self, num_classes):
        self.num_classes = num_classes

def default_loader(path: str):
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return torchvision.datasets.folder.accimage_loader(path)
    else:
        return torchvision.datasets.folder.pil_loader(path)

class imageDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform):
        super(imageDataset, self).__init__()
        self.samples = samples
        self.transform = transform
        self.loader = default_loader

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return {"x": sample, "y": target}

    def __len__(self):
        return len(self.samples)

class imageAnaDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform):
        super(imageAnaDataset, self).__init__()
        self.samples = samples
        self.transform = transform
        self.loader = default_loader

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return {"x": sample, "y": target, "path": path}

    def __len__(self):
        return len(self.samples)


class imageTestDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform, adv_transform):
        super(imageTestDataset, self).__init__()
        self.samples = samples
        self.transform = transform
        self.adv_transform = adv_transform
        self.loader = default_loader

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        trans_sample = self.transform(sample)
        return {"x": trans_sample, "y": target, "ori_x_path": path, "blur_x": self.adv_transform(sample)}

    def __len__(self):
        return len(self.samples)

def get_train_transform(model_name):
    if "CLIP" in model_name:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transform

def get_val_transform(model_name):
    if "CLIP" in model_name:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transform

def get_test_blur_transform(model_name, blur_scale=0.1):
    if "CLIP" in model_name:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=blur_scale),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=blur_scale),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transform

def build_dataset(args):
    data_dir = args.data_dir
    task_name = args.task
    model = args.img_model
    train_transform = get_train_transform(model)
    val_transform = get_val_transform(model)

    if args.data == "train_train_val_clean":
        train_path = os.path.join(data_dir, task_name, "train")
        in_img_folder_dataset = torchvision.datasets.ImageFolder(train_path)
        in_img_samples = in_img_folder_dataset.samples
        classes = in_img_folder_dataset.classes

        test_path = os.path.join(data_dir, task_name, "clean_samples")
        out_img_samples = torchvision.datasets.ImageFolder(test_path).samples

    elif args.data == "train_const_val_const":
        train_path = os.path.join(data_dir, task_name, "split_train")
        in_img_folder_dataset = torchvision.datasets.ImageFolder(train_path)
        in_img_samples = in_img_folder_dataset.samples
        classes = in_img_folder_dataset.classes

        test_path = os.path.join(data_dir, task_name, "split_val")
        out_img_samples = torchvision.datasets.ImageFolder(test_path).samples

    elif args.data == "train_random_val_random":
        data_path = os.path.join(data_dir, task_name, "mix_dataset")
        datas = torchvision.datasets.ImageFolder(data_path)
        keys = list(range(len(datas)))
        np.random.RandomState(0).shuffle(keys)
        keys_1 = keys[:int(len(datas)*0.8)]
        keys_2 = keys[int(len(datas)*0.8):]

        in_img_samples = datas.samples[keys_1]
        out_img_samples = datas.samples[keys_2]
        classes = datas.classes

    else:
        raise NotImplementedError

    dataset_train = imageDataset(in_img_samples, train_transform)
    dataset_val = imageDataset(out_img_samples, val_transform)

    data_info = dataInfo(len(classes))
    return data_info, dataset_train, dataset_val

def build_ana_dataset(args):
    data_dir = args.data_dir
    task_name = args.task
    model = args.img_model
    val_transform = get_val_transform(model)

    if args.data == "train_train_val_clean":
        train_path = os.path.join(data_dir, task_name, "train")
        in_img_folder_dataset = torchvision.datasets.ImageFolder(train_path)
        in_img_samples = in_img_folder_dataset.samples
        classes = in_img_folder_dataset.classes

        test_path = os.path.join(data_dir, task_name, "clean_samples")
        out_img_samples = torchvision.datasets.ImageFolder(test_path).samples

    elif args.data == "train_const_val_const":
        train_path = os.path.join(data_dir, task_name, "split_train")
        in_img_folder_dataset = torchvision.datasets.ImageFolder(train_path)
        in_img_samples = in_img_folder_dataset.samples
        classes = in_img_folder_dataset.classes

        test_path = os.path.join(data_dir, task_name, "split_val")
        out_img_samples = torchvision.datasets.ImageFolder(test_path).samples
    else:
        raise NotImplementedError

    dataset_train = imageAnaDataset(in_img_samples, val_transform)
    dataset_val = imageAnaDataset(out_img_samples, val_transform)

    data_info = dataInfo(len(classes))
    return data_info, dataset_train, dataset_val

def build_adv_dataset(args):
    data_dir = args.data_dir
    task_name = args.task
    model = args.img_model
    val_transform = get_val_transform(model)
    adv_base_name = args.adv_data_dir_name
    adv_folder_name = args.adv_sample_name
    adv_transform = get_test_blur_transform(model, args.blur_scale)

    if args.data == "train_train_val_clean":
        train_path = os.path.join(data_dir, task_name, "clean_samples")
        in_img_folder_dataset = torchvision.datasets.ImageFolder(train_path)
        in_img_samples = in_img_folder_dataset.samples
        classes = in_img_folder_dataset.classes

        if len(adv_base_name) == 0:
            test_path = os.path.join(data_dir, task_name, adv_folder_name)
        else:
            test_path = os.path.join(data_dir, task_name, adv_base_name, adv_folder_name)
        out_img_samples = torchvision.datasets.ImageFolder(test_path).samples

    else:
        raise NotImplementedError

    '''
    elif args.data == "train_const_val_const":
        train_path = os.path.join(data_dir, task_name, "split_train")
        in_img_folder_dataset = torchvision.datasets.ImageFolder(train_path)
        in_img_samples = in_img_folder_dataset.samples
        classes = in_img_folder_dataset.classes

        test_path = os.path.join(data_dir, task_name, "adv_samples")
        out_img_samples = torchvision.datasets.ImageFolder(test_path).samples
    '''

    dataset_train = imageTestDataset(in_img_samples, val_transform, adv_transform)
    dataset_val = imageTestDataset(out_img_samples, val_transform, adv_transform)

    data_info = dataInfo(len(classes))
    return data_info, dataset_train, dataset_val