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
from glob import glob
from os.path import *

sys.path.append('/home/ubuntu/flownet2-pytorch/')

import models, losses, datasets
from utils import flow_utils, tools

from matplotlib import pyplot as plt

def list_split_raw_kitti(h=32,w=128):
    basedir = '/home/ubuntu/kitti/raw/' #.format(h,w)
    debug_msg = 'load_raw_kitti_odom_imu'
    dates = [d for d in os.listdir(basedir) if os.path.isdir(basedir+d)]
    print(debug_msg,'dates:',dates)
    dates_drives = []
    for d in dates:
        dates_drives += [(d,drv[17:-5]) for drv in\
                         os.listdir(basedir+'/'+d+'/') if os.path.isdir(basedir+d+'/'+drv)]
    print(debug_msg,'dates_drives:',dates_drives)
    return dates_drives

class FramesDataset(Dataset):
    def __init__(self,fdir,new_shape=[16,64],offset=1,transform=None):
        self.fnames = glob(fdir+'/*.png') #sorted(glob(fdir+'/*.png'),key=lambda x:int(x[-10:-4]))
        print(self.fnames)
        self.len = len(self.fnames)-offset
        self.transform = transform
        self.offset = offset
        self.new_shape = new_shape

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        try:
            #h,w = self.new_shape
            h,w = 128,512
            offset = self.offset
            frame = cv2.imread(self.fnames[idx])
            frame = cv2.resize(frame,(w,h)).transpose(2,0,1)
            frame2 = cv2.imread(self.fnames[min(idx+offset,self.len-1)])
            frame2 = cv2.resize(frame2,(w,h)).transpose(2,0,1)
            frame = torch.from_numpy(frame).float().unsqueeze(0).transpose(1,0)
            frame2 = torch.from_numpy(frame2).float().unsqueeze(0).transpose(1,0)
            frame = torch.cat([frame,frame2],dim=1)
            return frame, torch.tensor(int(self.fnames[idx][-10:-4]))
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
    offset = 1
    h,w = 32,128
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
    #model.eval()
    model.train()

    dest_dir = '/home/ubuntu/kitti/flow/raw/{}x{}_flownet_{}/'.format(h,w,offset)
    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)

    basedir = '/home/ubuntu/kitti/raw/'.format(h,w)
    dds = list_split_raw_kitti() ## dates, drives
    for dd in dds:
        frames_dir = basedir+dd[0]+'/{}_drive_{}_sync/image_00/data/'.format(dd[0],dd[1])
        frames_dataset = FramesDataset(frames_dir,new_shape=[h,w],offset=offset)
        frames_loader = DataLoader(frames_dataset,batch_size=512,shuffle=False)

        seq_dir = '/home/ubuntu/kitti/flow/raw/{}x{}_flownet_{}/{}_{}/'.format(h,w,offset,dd[0],dd[1])
        if not os.path.isdir(seq_dir):
            os.mkdir(seq_dir)

        i = 0
        for x,idx in frames_loader:
            f = model(x)[0]
            #for y in f: print(y.size())
            for fr,id in zip(f,idx):
                fr = fr.permute(1,2,0).detach().cpu().numpy()
                id = id.detach().cpu().numpy()
                id = int(id)
                #print(x.size(),f.shape)
                fname = seq_dir+'{:06d}.npy'.format(id)
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
