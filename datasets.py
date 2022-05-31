import glob
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def to_rgb(image):
    rgb_image = Image.new("RGB",image.size)
    rgb_image.paste(image)
    return rgb_image

def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)


class ImageDataset(Dataset):
    def __init__(self, root,hr_shape,transform_=None,unaligned=False, mode="train"):
        hr_height, hr_width = hr_shape
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)]).convert('RGB')
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)
        # Convert grayscale images to rgb
        # if img_lr.mode != "RGB":
        #     img_lr = to_rgb(img_lr)
        # if img_hr.mode != "RGB":
        #     img_hr = to_rgb(img_hr)
        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files)