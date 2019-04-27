import os
import sys
import re
import cv2
import h5py
import numpy as np
import torch
import torch.optim as optim

from glob import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pt_ae import VanillaAutoencoder

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
        return torch.from_numpy(image).unsqueeze(0).float()

class H5Dataset(Dataset):
    ''' Assumes the images are of shape (C,H,W)
    '''
    def __init__(self, file_path, transform=None):
        super(H5Dataset, self).__init__()
        h5_file = h5py.File(file_path)
        self.data = h5_file.get('data')

    def __getitem__(self, index):
        x = torch.from_numpy(self.data[index,:,:,:]).float()
        if self.transform:
            x = self.transform(x)
        return x

    def __len__(self):
        return self.data.shape[0]

class FramesDataset(Dataset):
    def __init__(self, fnames, transform=None):
        self.fnames = fnames #sorted([fn for fn in glob(re_dir) if os.path.isfile(fn)])
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
    train_dir = sys.argv[1] #'/home/ronnypetson/Documents/deep_odometry/kitti/dataset_frames/sequences/*/image_0/*'
    valid_dir = sys.argv[2] #'/home/ronnypetson/Documents/deep_odometry/kitti/dataset_frames/00/image_0/*'
    test_dir = sys.argv[3] #'/home/ronnypetson/Documents/deep_odometry/kitti/dataset_frames/01/image_0/*'

    print(train_dir,valid_dir,test_dir)

    train_dir = sorted([fn for fn in glob(train_dir) if os.path.isfile(fn)])
    valid_dir = sorted([fn for fn in glob(valid_dir) if os.path.isfile(fn)])
    test_dir = sorted([fn for fn in glob(test_dir) if os.path.isfile(fn)])

    new_dim = (int(sys.argv[4]),int(sys.argv[5]))
    transf = transforms.Compose([Rescale(new_dim),ToTensor()])

    train_dataset = FramesDataset(train_dir,transf)
    valid_dataset = FramesDataset(valid_dir,transf)
    test_dataset = FramesDataset(test_dir,transf)

    train_loader = DataLoader(train_dataset,batch_size=384,shuffle=True,num_workers=1)
    valid_loader = DataLoader(valid_dataset,batch_size=64,shuffle=True,num_workers=4)
    test_loader = DataLoader(test_dataset,batch_size=64,shuffle=True,num_workers=4)

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = VanillaAutoencoder((1,)+new_dim).to(device)
    params = model.parameters()
    optimizer = optim.Adam(params,lr=3e-4)
    loss_fn = torch.nn.MSELoss()
    for i in range(2):
        model.train()
        losses = []
        for j,x in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.to(device)
            y_ = model(x)
            loss = loss_fn(y_,x)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            print('Batch {} loss {}'.format(j,loss.item()))
        print('Epoch {}: {}'.format(i,np.mean(losses)))
