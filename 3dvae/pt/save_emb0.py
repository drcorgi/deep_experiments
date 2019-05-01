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
from datetime import datetime
from odom_dataset import my_collate, Rescale, ToTensor, H5Dataset, FramesDataset

if __name__=='__main__':
    train_dir = sys.argv[1] #'/home/ronnypetson/Documents/deep_odometry/kitti/frames_odom_train.h5'
    valid_dir = sys.argv[2] #'/home/ronnypetson/Documents/deep_odometry/kitti/frames_odom_valid.h5'
    test_dir = sys.argv[3] #'/home/ronnypetson/Documents/deep_odometry/kitti/frames_odom_test.h5'

    emb_dir = sys.argv[4] #'/home/ronnypetson/Documents/deep_odometry/kitti/frames_emb0.pck'
    model_fn = sys.argv[5]
    new_dim = (int(sys.argv[6]),int(sys.argv[7]))
    batch_size = int(sys.argv[8])
    transf = transforms.Compose([Rescale(new_dim),ToTensor()])

    valid_dataset = H5Dataset(valid_dir,10,transf)
    test_dataset = H5Dataset(test_dir,10,transf)
    train_dataset = H5Dataset(train_dir,10,transf)

    valid_loader = DataLoader(valid_dataset,batch_size=batch_size,shuffle=False,num_workers=0,collate_fn=my_collate)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=0,collate_fn=my_collate)
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=False,num_workers=0,collate_fn=my_collate)

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(device)
    model = VanillaAutoencoder((1,)+new_dim).to(device)
    params = model.parameters()
    optimizer = optim.Adam(params,lr=3e-4)
    min_loss = 1e15
    epoch = 0

    if os.path.isfile(model_fn):
        print('Loading existing model')
        checkpoint = torch.load(model_fn)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        min_loss = checkpoint['min_loss']
        epoch = checkpoint['epoch']
    else:
        print('Model not found')
        exit()

    model.eval()
    all_enc = []
    for loader in [eval_loader,test_loader,train_loader]:
        for x in loader:
            x = x.to(device)
            z = model.forward_z(x).detach().numpy()
            all_enc.append(z)
    all_enc = np.concatenate(all_enc,axis=0)
