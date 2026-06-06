
import os
import random
import torch
import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join
from data.util import *
from torchvision import transforms as t

    
class LOLDatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(LOLDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.norm = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):

        folder = self.data_dir+'/low'
        folder2= self.data_dir+'/high'
        data_filenames = [join(folder, x) for x in listdir(folder) if is_image_file(x)]
        data_filenames2 = [join(folder2, x) for x in listdir(folder2) if is_image_file(x)]
        num = len(data_filenames)

        im1 = load_img(data_filenames[index])
        im2 = load_img(data_filenames2[index])
        _, file1 = os.path.split(data_filenames[index])
        _, file2 = os.path.split(data_filenames2[index])
        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed) # make a seed with numpy generator 
        if self.transform:
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            im1 = self.transform(im1)
            random.seed(seed)
            torch.manual_seed(seed)         
            im2 = self.transform(im2) 
        return im1, im2, file1, file2

    def __len__(self):
        return 485

    
class LOLv2DatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(LOLv2DatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform

    def __getitem__(self, index):

        folder = self.data_dir + '/low'
        folder2 = self.data_dir + '/high'

        data_filenames = sorted([join(folder, x) for x in listdir(folder) if is_image_file(x)])

        low_path = data_filenames[index]
        name = os.path.basename(low_path)

        # 🔥 correct pairing
        gt_name = name.replace('low', 'normal')
        high_path = os.path.join(folder2, gt_name)

        im1 = load_img(low_path)
        im2 = load_img(high_path)

        seed = np.random.randint(1000000)

        if self.transform:
            random.seed(seed)
            torch.manual_seed(seed)
            im1 = self.transform(im1)

            random.seed(seed)
            torch.manual_seed(seed)
            im2 = self.transform(im2)

        return im1, im2, name, gt_name

    def __len__(self):
        return len(sorted([x for x in listdir(self.data_dir + '/low') if is_image_file(x)]))

class LOLv2SynDatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(LOLv2SynDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform

        self.low_folder = self.data_dir + '/low'
        self.high_folder = self.data_dir + '/high'

        # 🔥 SORTED (important)
        self.low_files = sorted([x for x in listdir(self.low_folder) if is_image_file(x)])

    def __getitem__(self, index):

        low_name = self.low_files[index]
        low_path = join(self.low_folder, low_name)

        # 🔥 correct pairing (Low → Normal)
        high_name = low_name.replace('low', 'normal')
        high_path = join(self.high_folder, high_name)

        im1 = load_img(low_path)
        im2 = load_img(high_path)

        seed = np.random.randint(1000000)

        if self.transform:
            random.seed(seed)
            torch.manual_seed(seed)
            im1 = self.transform(im1)

            random.seed(seed)
            torch.manual_seed(seed)
            im2 = self.transform(im2)

        return im1, im2, low_name, high_name

    def __len__(self):
        return len(self.low_files)



    

