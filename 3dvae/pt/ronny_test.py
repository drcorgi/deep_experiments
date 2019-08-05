#!/usr/bin/env python

import torch
import torch.nn as nn
import os
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter

import argparse, os, sys, subprocess
import numpy as np
import cv2
#from tqdm import tqdm
from glob import glob
from os.path import *

sys.path.append('/home/ubuntu/flownet2-pytorch/')

import models, losses, datasets
from utils import flow_utils, tools

from matplotlib import pyplot as plt

global offset

def list_split_kitti(seq):
    base = '/home/ubuntu/kitti/dataset/'
    all_seqs = [sorted(glob(base+'sequences/{:02d}/image_0/*.png'\
                .format(i))) for i in range(11)]
    train_seqs = all_seqs[seq]
    return train_seqs

class FramesDataset(Dataset):
    def __init__(self,fnames,transform=None):
        self.fnames = fnames
        self.len = len(self.fnames)-offset
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        try:
            frame = cv2.imread(self.fnames[idx])
            frame = cv2.resize(frame,(256,64)).transpose(2,0,1)
            frame2 = cv2.imread(self.fnames[min(idx+offset,self.len-1)])
            frame2 = cv2.resize(frame2,(256,64)).transpose(2,0,1)
            frame = torch.from_numpy(frame).float().unsqueeze(0).transpose(1,0)
            frame2 = torch.from_numpy(frame2).float().unsqueeze(0).transpose(1,0)
            frame = torch.cat([frame,frame2],dim=1)
            return frame
        except Exception as e:
            print(e,'frame not loaded')

def plotflow(f,fig,sub,title=''):
    ax = fig.add_subplot(sub)
    ax.title.set_text(title)
    scale = 1
    X,Y = np.meshgrid(np.arange(0,f.shape[0],scale),np.arange(0,f.shape[1],scale))
    U,V = f[::scale,::scale,0], f[::scale,::scale,1]
    M = np.hypot(U,V)
    Q = ax.quiver(X,Y,U,V,M,units='x',pivot='mid',width=0.5,scale=1/0.15)
    #qk = plt.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',\
    #               coordinates='figure')
    #plt.scatter(X,Y,color='0.5',s=1)

class Arguments:
    def __init__(self):
        self.rgb_max = 255.0

if __name__ == '__main__':
    global offset
    offset = 1

    model_fn = '/home/ubuntu/models/FlowNet2-S_checkpoint.pth'
    device = torch.device('cuda:0')
    args = Arguments()
    model = models.FlowNet2S(args)
    if os.path.isfile(model_fn):
        print('Loading existing model')
        checkpoint = torch.load(model_fn)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print('Model checkpoint not found')
        raise FileNotFoundError()
    model.train()

    for seq in range(11):
        frames_dir = list_split_kitti(seq)
        frames_dataset = FramesDataset(frames_dir)
        frames_loader = DataLoader(frames_dataset,batch_size=512,shuffle=False)

        seq_dir = '/home/ubuntu/kitti/flow/64x256_flownet_{}/{:02d}'.format(offset,seq)
        if not os.path.isdir(seq_dir):
            os.mkdir(seq_dir)

        i = 0
        for x in frames_loader:
            f = model(x)[0]
            print(f.size())
            for fr in f:
                fr = fr.permute(1,2,0).detach().cpu().numpy()
                #print(x.size(),f.shape)
                fname = seq_dir+'/{}.npy'.format(i)
                print('Saving {} of shape {}'.format(fname,fr.shape))
                np.save(fname,fr)
                i += 1

        #x0 = x[0,0,0].detach().cpu().numpy()
        #x1 = x[0,0,1].detach().cpu().numpy()
        #fcv = cv2.calcOpticalFlowFarneback(x0,x1,None,0.5,3,15,3,5,1.2,0)
        #fig = plt.figure()
        #plt.title('Flow comparison. Frame offset = {}'.format(offset))
        #plotflow(f,fig,121,title='flownet 2.0')
        #plotflow(fcv,fig,122,title='cv2\'s Farneback')
        #plt.show()
