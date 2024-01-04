import os
from glob import glob
from pathlib import Path
import shutil
import numpy as np
import csv
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import torchvision.datasets as datasets
from torchvision.datasets import CIFAR10


class Dataset_maker(torch.utils.data.Dataset):
    def __init__(self, root, category, config, is_train=True):
        print("Dataset root",root)
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((config.data.image_size, config.data.image_size)),  
                transforms.ToTensor(), # Scales data into [0,1] 
                transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
            ]
        )
        self.config = config
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((config.data.image_size, config.data.image_size)),
                transforms.ToTensor(), # Scales data into [0,1] 
            ]
        )
        if is_train:
            if category:
                self.image_files = glob(
                    os.path.join(root, "train",  "*.jpg")
                )
            else:
                self.image_files = glob(
                    os.path.join(root, "train",  "*.jpg")
                )
        else:
            if category:
                self.image_files = glob(os.path.join(root, "test", "*.jpg"))
            else:
                self.image_files = glob(os.path.join(root, "test", "*.jpg"))
        print("Images",len(self.image_files))
        self.is_train = is_train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = self.image_transform(image)
        if(image.shape[0] == 1):
            image = image.expand(3, self.config.data.image_size, self.config.data.image_size)
        label = 'good'
        return image, label

    def __len__(self):
        return len(self.image_files)
