import itertools
import os
import sys
import re
import cv2
import h5py
import numpy as np
import torch
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/topos')
sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/preprocess')
sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/misc')
sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/sample')

from glob import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pt_ae import DirectOdometry, FastDirectOdometry,\
VanillaAutoencoder, MLPAutoencoder, VanAE, Conv1dMapper, seq_pose_loss
from datetime import datetime
from plotter import c3dto2d, abs2relative, plot_eval
from odom_loader import load_kitti_odom
from tensorboardX import SummaryWriter
from time import time

def my_collate(batch):
    batch_x = []
    batch_y = []
    batch_abs = []
    for b in batch:
        if b is not None:
            batch_x.append(b[0])
            batch_y.append(b[1])
            batch_abs.append(b[2])
    return torch.stack(batch_x), torch.stack(batch_y), batch_abs

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        image = cv2.resize(image,self.output_size)
        return image

class ToTensor(object):
    def __call__(self,x):
        return torch.from_numpy(x).float()

def list_split_kitti_frames(h,w,tipo=''):
    base = '/home/ubuntu/kitti/{}/{}x{}/'.format(tipo,h,w)
    assert os.path.isdir(base)
    pbase = '/home/ubuntu/kitti/dataset/'
    all_seqs = [sorted(glob(base+'{:02d}/*.npy'\
                .format(i))) for i in range(11)]
    train_seqs = all_seqs[2:]
    valid_seqs = all_seqs[0:1]
    test_seqs = all_seqs[1:2]
    train_seqs = list(itertools.chain.from_iterable(train_seqs))
    valid_seqs = list(itertools.chain.from_iterable(valid_seqs))
    test_seqs = list(itertools.chain.from_iterable(test_seqs))
    return train_seqs, valid_seqs, test_seqs

def list_split_kitti_flux(h,w):
    base = '/home/ubuntu/kitti/flux/{}x{}/'.format(h,w)
    pbase = '/home/ubuntu/kitti/dataset/'
    all_seqs = [sorted(glob(base+'{:02d}/*.npy'\
                .format(i))) for i in range(11)]
    all_poses = [pbase+'poses/{:02d}.txt'.format(i) for i in range(11)]
    train_seqs, train_poses = all_seqs[2:], all_poses[2:]
    valid_seqs, valid_poses = all_seqs[0:1], all_poses[0:1]
    test_seqs, test_poses = all_seqs[1:2], all_poses[1:2]
    return (train_seqs,train_poses), (valid_seqs,valid_poses), (test_seqs,test_poses)

class FluxDataset(Dataset):
    def __init__(self, fnames, transform=None):
        self.fnames = fnames #sorted([fn for fn in glob(re_dir) if os.path.isfile(fn)])
        self.len = len(self.fnames)
        self.transform = transform
        self.cache = {}

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        try:
            if idx in self.cache:
                return self.cache[idx]
            frame = np.load(self.fnames[idx])
            if frame is not None and self.transform:
                frame = frame.transpose(2,0,1)
                frame = self.transform(frame)
            self.cache[idx] = frame
            return frame
        except Exception as e:
            print(e)

class FramesDataset(Dataset):
    def __init__(self, fnames, transform=None):
        self.fnames = fnames #sorted([fn for fn in glob(re_dir) if os.path.isfile(fn)])
        self.len = len(self.fnames)
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        try:
            frame = cv2.imread(self.fnames[idx],0)
            if frame is not None and self.transform:
                frame = np.expand_dims(frame,axis=0)
                frame = self.transform(frame)
            return frame
        except Exception as e:
            print(e)

if __name__=='__main__':
    model_fn = sys.argv[1]
    h_dim = int(sys.argv[2])
    new_dim = (int(sys.argv[3]),int(sys.argv[4]))
    batch_size = int(sys.argv[5])
    num_epochs = int(sys.argv[6])
    tipo = 'flux'
    transf = ToTensor()

    train_dir,valid_dir,test_dir = list_split_kitti_frames(new_dim[0],new_dim[1],tipo=tipo)

    if tipo == 'flux':
        FrDataset = FluxDataset
    else:
        FrDataset = FramesDataset

    train_dataset = FrDataset(train_dir,transf)
    valid_dataset = FrDataset(valid_dir,transf)
    test_dataset = FrDataset(test_dir,transf)

    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=4)
    valid_loader = DataLoader(valid_dataset,batch_size=batch_size,shuffle=False,num_workers=4)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=4)

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(device)

    frshape = (1,)+new_dim if tipo != 'flux' else (2,)+new_dim
    model = VanAE(frshape,h_dim).to(device)
    params = model.parameters()
    optimizer = optim.Adam(params,lr=1e-3)
    min_loss = 1e15
    epoch = 0
    writer = SummaryWriter('/home/ubuntu/log/enc_h{}_{}x{}'\
                           .format(h_dim,new_dim[0],new_dim[1]))

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
    k,kv = 0,0
    #epoch = num_epochs-1
    for i in range(epoch,num_epochs):
        model.train()
        losses = []
        for j,x in enumerate(train_loader):
            #t = time()
            x = x.to(device)
            optimizer.zero_grad()
            x_ = model(x)
            #print(x.size(),x_.size())
            loss = loss_fn(x,x_) #loss_fn(y_,y)
            loss.backward()
            optimizer.step()
            writer.add_scalar('train_cost',loss.item(),k)
            losses.append(loss.item())
            k += 1
            #print('Batch {}\tloss: {:.3f}'.format(j,loss.item()))
            #print('inference',time()-t)
        xx_ = torch.cat([x[0][0],x_[0][0]],dim=1).unsqueeze(0)
        writer.add_image('_frames_true_dec{}'.format(i),xx_)
        #writer.add_image('_frames_dec_{}'.format(i),x_[0])
        #writer.add_image('_img_emb_{}'.format(i),\
        #                 z[:seq_len].unsqueeze(0))
        model.eval()
        v_losses = []
        for j,x in enumerate(valid_loader):
            x = x.to(device)
            x_ = model(x)
            loss = loss_fn(x_,x)
            writer.add_scalar('valid_cost',loss.item(),kv)
            v_losses.append(loss.item())
            kv += 1
        mean_train, mean_valid = np.mean(losses),np.mean(v_losses)
        print('Epoch {} loss\t{:.3f}\tValid loss\t{:.3f}'\
              .format(i,mean_train,mean_valid))
        xx_ = torch.cat([x[0][0],x_[0][0]],dim=1).unsqueeze(0)
        writer.add_image('_frames_true_dec_valid{}'.format(i),xx_)
        if mean_valid < min_loss:
            print('Saving model')
            min_loss = mean_valid
            torch.save({'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'min_loss': min_loss,
                        'epoch': i+1}, model_fn)
    print('Start of test')
    model.eval()
    t_losses = []
    for x in test_loader:
        x = x.to(device)
        x_ = model(x)
        loss = loss_fn(x_,x)
        t_losses.append(loss.item())
    mean_test = np.mean(t_losses)
    print('Test loss:',np.mean(mean_test))
