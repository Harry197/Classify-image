from __future__ import print_function, division
import os
import torch
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2


import albumentations as A
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class MyDataset(Dataset):
    
    def __init__(self, csv_file, root_dir,transform=None):
        self.data_frame = pd.read_csv(csv_file, header=None)
        self.rootdir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        image_name = os.path.join(self.rootdir, self.data_frame.iloc[idx, 0])
        # print(image_name)
        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.data_frame.iloc[idx, 1]
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return [image, label]