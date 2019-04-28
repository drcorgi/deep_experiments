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

def my_collate(batch):
    batch_ = []
    for b in batch:
        if b is not None:
            batch_.append(b)
    return torch.stack(batch_)

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
        #print(self.fnames[idx])
        try:
            frame = cv2.imread(self.fnames[idx],0)
            if frame is not None and self.transform:
                frame = self.transform(frame)
            return frame
        except Exception as e:
            print(e)

if __name__=='__main__':
    train_dir = sys.argv[1] #'/home/ronnypetson/Documents/deep_odometry/kitti/dataset_frames/sequences/*/image_0/*'
    valid_dir = sys.argv[2] #'/home/ronnypetson/Documents/deep_odometry/kitti/dataset_frames/00/image_0/*'
    test_dir = sys.argv[3] #'/home/ronnypetson/Documents/deep_odometry/kitti/dataset_frames/01/image_0/*'

    train_dir = sorted([fn for fn in glob(train_dir) if os.path.isfile(fn)])
    valid_dir = sorted([fn for fn in glob(valid_dir) if os.path.isfile(fn)])
    test_dir = sorted([fn for fn in glob(test_dir) if os.path.isfile(fn)])

    model_fn = sys.argv[4]
    log_folder = sys.argv[5]
    new_dim = (int(sys.argv[6]),int(sys.argv[7]))
    batch_size = int(sys.argv[8])
    num_epochs = int(sys.argv[9])
    transf = transforms.Compose([Rescale(new_dim),ToTensor()])

    train_dataset = FramesDataset(train_dir,transf)
    valid_dataset = FramesDataset(valid_dir,transf)
    test_dataset = FramesDataset(test_dir,transf)

    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=2,collate_fn=my_collate)
    valid_loader = DataLoader(valid_dataset,batch_size=batch_size//2,shuffle=False,num_workers=2,collate_fn=my_collate)
    test_loader = DataLoader(test_dataset,batch_size=batch_size//2,shuffle=False,num_workers=2,collate_fn=my_collate)

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
        print('Creating new model')

    loss_fn = torch.nn.MSELoss()
    epoch_losses = []
    for i in range(epoch,num_epochs):
        print('Start of epoch',i)
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
            print('Batch {} loss {:.3f}'.format(j,loss.item()))
        model.eval()
        v_losses = []
        for j,x in enumerate(valid_loader):
            x = x.to(device)
            y_ = model(x)
            loss = loss_fn(y_,x)
            v_losses.append(loss.item())
        mean_train, mean_valid = np.mean(losses),np.mean(v_losses)
        epoch_losses.append([i,mean_train,mean_valid])
        print('Epoch {} loss {:.3f} Valid loss {:.3f}'.format(i,mean_train,mean_valid)) # Mean train loss, mean validation loss
        if mean_valid < min_loss:
            min_loss = mean_valid
            torch.save({'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'min_loss': min_loss,
                        'epoch': i+1}, model_fn)
    model.eval()
    t_losses = []
    for j,x in enumerate(test_loader):
        x = x.to(device)
        y_ = model(x)
        loss = loss_fn(y_,x)
        t_losses.append(loss.item())
    mean_test = np.mean(t_losses)
    epoch_losses.append([i,0.0,mean_test])
    print('Test loss:',np.mean(mean_test))
    # Save training log
    np.save('{}/{}_log.npy'.format(log_folder,datetime.now()),epoch_losses)
