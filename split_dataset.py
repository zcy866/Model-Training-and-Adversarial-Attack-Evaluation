import torch
from torchvision import datasets
import os
import torchvision.transforms as transforms
import random
from PIL import Image
import shutil
import urllib


data_path = './datasets'
save_path = "./datasets/attack_dataset"


task_name = "attack_dataset/train"
domain_path = os.path.join(data_path, task_name)
class_folders = os.listdir(domain_path)
print(class_folders)
for class_name in class_folders:
    class_path = os.path.join(data_path, task_name, class_name)
    data_files = os.listdir(class_path)
    lens = len(data_files)
    test_list = list(range(lens))
    train_result = random.sample(test_list, k=int(lens * 0.8))

    class_save_path = os.path.join(save_path, 'split_train', class_name)
    if not os.path.exists(class_save_path):
        # os.system('mkdir -p ' + save_path)
        os.makedirs(class_save_path)

    class_save_path = os.path.join(save_path, 'split_val', class_name)
    if not os.path.exists(class_save_path):
        # os.system('mkdir -p ' + save_path)
        os.makedirs(class_save_path)

    for name, val in zip(data_files, test_list):
        source_path = os.path.join(data_path, task_name, class_name, name)
        if val in train_result:
            destination_path = os.path.join(save_path, 'split_train', class_name, name)
        else:
            destination_path = os.path.join(save_path, 'split_val', class_name, name)
        shutil.copy(source_path, destination_path)