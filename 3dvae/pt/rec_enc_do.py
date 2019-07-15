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
from pt_ae import DirectOdometry, FastDirectOdometry, Conv1dRecMapper,\
VanillaAutoencoder, MLPAutoencoder, VanAE, Conv1dMapper, seq_pose_loss
from datetime import datetime
from plotter import c3dto2d, abs2relative, plot_eval, plot_yy
from odom_loader import load_kitti_odom
from tensorboardX import SummaryWriter
from time import time
from seq_datasets import FastFluxSeqDataset, FastSeqDataset, FluxSeqDataset,\
list_split_kitti_flux, list_split_kitti_, my_collate, ToTensor

if __name__=='__main__':
    enc_fn = sys.argv[1]
    model_fn = sys.argv[2]
    h_dim = int(sys.argv[3])
    new_dim = (int(sys.argv[4]),int(sys.argv[5]))
    seq_len = int(sys.argv[6])
    batch_size = int(sys.argv[7])
    num_epochs = int(sys.argv[8])
    stride = 1
    assert seq_len%stride == 0
    strided_seq_len = seq_len//stride
    tipo = 'flux' #'flux' or 'img'
    loading = 'lazy' #'cached' or 'lazy'
    #transf = transforms.Compose([Rescale(new_dim),ToTensor()])
    #transf = [Rescale(new_dim),ToTensor()]
    transf = ToTensor()

    if tipo == 'flux':
        frshape = (2,) + new_dim
        train_dir,valid_dir,test_dir = list_split_kitti_flux(new_dim[0],new_dim[1])
        if loading == 'cached':
            FrSeqDataset = FastFluxSeqDataset
        else:
            FrSeqDataset = FluxSeqDataset
    else:
        frshape = (1,) + new_dim
        train_dir,valid_dir,test_dir = list_split_kitti_(new_dim[0],new_dim[1])
        FrSeqDataset = FastSeqDataset

    train_dataset = FrSeqDataset(train_dir[0],train_dir[1],seq_len,transf,stride=stride)
    valid_dataset = FrSeqDataset(valid_dir[0],valid_dir[1],seq_len,transf,stride=stride)
    test_dataset = FrSeqDataset(test_dir[0],test_dir[1],seq_len,transf,stride=stride)

    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=0,collate_fn=my_collate)
    valid_loader = DataLoader(valid_dataset,batch_size=batch_size,shuffle=False,num_workers=0,collate_fn=my_collate)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=0,collate_fn=my_collate)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(device)

    model = VanAE(frshape,h_dim).to(device)
    if os.path.isfile(enc_fn):
        print('Encoder found')
        checkpoint = torch.load(enc_fn)
        model.load_state_dict(checkpoint['model_state'])
        '''for param in model.enc.parameters():
            param.requires_grad = False'''
    else:
        print('Encoder not found. Creating new one')

    vo = Conv1dRecMapper((h_dim,strided_seq_len),(strided_seq_len,12)).to(device)
    #vo = Conv1dMapper((h_dim,strided_seq_len),(strided_seq_len,12)).to(device)
    model.dec = vo

    ##model = VanillaAutoencoder((1,)+new_dim).to(device)
    #model = VanillaAutoencoder((2,)+new_dim,h_dim).to(device)
    #model = MLPAutoencoder((2,)+new_dim,h_dim).to(device)
    #model = FastDirectOdometry((1,)+new_dim,(12,)).to(device)
    params = model.parameters()
    optimizer = optim.Adam(params,lr=3e-4)
    min_loss = 1e15
    epoch = 0
    writer = SummaryWriter('/home/ubuntu/log/exp_h{}_l{}_s{}_{}x{}'\
                           .format(h_dim,seq_len,stride,new_dim[0],new_dim[1]))

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
    #loss_fn = seq_pose_loss
    k,kv = 0,0
    for i in range(epoch,num_epochs):
        print('Epoch',i)
        model.train()
        losses = []
        for j,xy in enumerate(train_loader):
            x,y = xy[0].to(device), xy[1].to(device)
            optimizer.zero_grad()
            y_ = model(x)
            loss = loss_fn(y,y_)
            loss.backward()
            optimizer.step()
            writer.add_scalar('train_cost',loss.item(),k)
            losses.append(loss.item())
            k += 1
            #print('Epoch {} Batch {} loss: {:.4f}'.format(i,j,loss.item()))
        #x_ = x[0].view(-1,x.size(-1)).unsqueeze(0)
        #print(x_.size())
        #writer.add_image('_img_seq_{}'.format(i),x_)
        #writer.add_image('_img_emb_{}'.format(i),\
        #                 z[:seq_len].unsqueeze(0))
        plot_yy(y,y_,device,writer) ###
        model.eval()
        v_losses = []
        for j,xy in enumerate(valid_loader):
            x,y = xy[0].to(device), xy[1].to(device)
            y_ = model(x)
            loss = loss_fn(y_,y)
            v_losses.append(loss.item())
            writer.add_scalar('valid_cost',loss.item(),kv)
            kv += 1
        #writer.add_embedding(y[0,:,[3,7,11]],tag='gt_pts_{}'.format(i),global_step=1)
        #writer.add_embedding(y_[0,:,[3,7,11]],tag='est_pts_{}'.format(i),global_step=1)
        mean_train, mean_valid = np.mean(losses),np.mean(v_losses)
        print('Epoch {} loss\t{:.4f}\tValid loss\t{:.4f}'\
              .format(i,mean_train,mean_valid))
        if mean_valid < min_loss:
            min_loss = mean_valid
            torch.save({'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'min_loss': min_loss,
                        'epoch': i+1}, model_fn)
    model.eval()
    print('Start of plot_eval')
    plot_eval(model,test_loader,strided_seq_len,device,logger=writer)
    writer.close()
    print('End of plot_eval')
