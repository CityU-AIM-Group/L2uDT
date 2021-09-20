import os
import torch
from torch.utils.data import Dataset

from .rand import Uniform
from .transforms import Rot90, Flip, Identity, Compose
from .transforms import GaussianBlur, Noise, Normalize, RandSelect
from .transforms import RandCrop, CenterCrop, Pad,RandCrop3D,RandomRotion,RandomFlip,RandomIntensityChange
from .transforms import NumpyType
from .data_utils import pkload

import numpy as np


class ProstateDataset(Dataset):
    def __init__(self, list_file, root='', for_train=False,transforms=''):
        paths, names = [], []
        with open('/home/xiaoqiguo2/DMFNet_L2uDT'+list_file) as f:
            for line in f:
                line = line.strip()
                name = line.split('/')[-1]
                names.append(name.split('.')[0])
                path = os.path.join(root, line.split('.')[0])
                paths.append('/home/xiaoqiguo2/DMFNet_L2uDT/'+path)

        self.names = names
        self.paths = paths
        self.transforms = eval(transforms or 'Identity()')

    def __getitem__(self, index):
        path = self.paths[index]
        x, y = pkload(path + 'data_f32.pkl')
        x = x.transpose(1, 2, 0, 3)
        y = y.transpose(1, 2, 0)
        # print(x.shape, y.shape)#(240, 240, 155, 4) (240, 240, 155)
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155, 4) (1, 240, 240, 155)
        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        y = np.ascontiguousarray(y)

        x, y = torch.from_numpy(x), torch.from_numpy(y)
        ones = torch.ones_like(y)
        y = torch.where(y>ones*2, ones*0, y)
        # print(x.shape, y.shape)  # (240, 240, 155, 4) (240, 240, 155)
        return x, y

    def __len__(self):
        return len(self.paths)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]

