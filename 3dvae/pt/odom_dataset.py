import os
import sys
import re
import cv2
import numpy as np
import torch

from glob import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size*h/w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size*w/h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h),int(new_w)
        image = cv2.resize(image,(new_h,new_w))
        return image

class ToTensor(object):
    def __call__(self,image):
        return torch.from_numpy(image).unsqueeze(0)

class FramesDataset(Dataset):
    def __init__(self, re_dir, transform=None):
        self.fnames = sorted([fn for fn in glob(re_dir) if os.path.isfile(fn)])
        self.len = len(self.fnames)
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        frame = cv2.imread(self.fnames[idx],0)
        if self.transform:
            frame = self.transform(frame)
        return frame

if __name__=='__main__':
    re_dir = '/home/ronnypetson/Documents/deep_odometry/kitti/dataset_frames/sequences/*/image_0/*'
    new_dim = (int(sys.argv[1]),int(sys.argv[2]))
    transf = transforms.Compose([Rescale(new_dim),ToTensor()])
    fdataset = FramesDataset(re_dir,transf)
    data_loader = DataLoader(fdataset,batch_size=64,shuffle=True,num_workers=4)
    '''for i, batch in enumerate(data_loader):
        print(i,batch.size())
        pass'''
