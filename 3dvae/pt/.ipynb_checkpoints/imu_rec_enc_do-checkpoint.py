import os
import sys
import re
import cv2
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/topos')
sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/preprocess')
sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/misc')
sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/sample')
sys.path.append('/home/ubuntu/flownet2-pytorch/')

import models, losses, datasets
from utils import flow_utils, tools

from glob import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pt_ae import DirectOdometry, FastDirectOdometry, Conv1dRecMapper, ImgFlowOdom, DummyFlow,\
VanillaAutoencoder, MLPAutoencoder, VanAE, Conv1dMapper, seq_pose_loss, VanillaEncoder,\
seq_pose_loss_se2, seq_pose_loss_SE2, FlatIMUEncoder, StatXIMU, VanAE_,\
seq_pose_loss_se3
from datetime import datetime
from plotter import c3dto2d, abs2relative, plot_eval_, plot_yy_
from odom_loader import load_kitti_odom
from tensorboardX import SummaryWriter
from time import time
from seq_datasets import FluxSeqDataset, SeqDataset, RawKITTIDataset,\
list_split_kitti_flow, list_split_kitti_flux, list_split_kitti_, my_collate, ToTensor,\
list_split_raw_kitti
from load_flow import Arguments
from custom_optimizers import RAdam

if __name__=='__main__':
    enc_fn = sys.argv[1]
    model_fn = sys.argv[2]
    h_dim = int(sys.argv[3])
    new_dim = (int(sys.argv[4]),int(sys.argv[5]))
    seq_len = int(sys.argv[6])
    batch_size = int(sys.argv[7])
    num_epochs = int(sys.argv[8])
    enc_type = None if len(sys.argv) < 10 else sys.argv[9]
    flow_fn = '/home/ubuntu/models/FlowNet2-S_checkpoint.pth'
    fine_tune_flow = False
    stride = 1
    delay = None if len(sys.argv) < 11 else int(sys.argv[10]) ## 8
    assert seq_len%stride == 0
    strided_seq_len = seq_len//stride
    tipo = 'flow' #'flux' or 'img'
    loading = 'lazy' #'cached' or 'lazy'
    #transf = transforms.Compose([Rescale(new_dim),ToTensor()])
    #transf = [Rescale(new_dim),ToTensor()]
    transf = ToTensor()

    if tipo == 'flow':
        flshape = (2,) + new_dim
        train_dd,valid_dd,test_dd = list_split_raw_kitti() #(new_dim[0],new_dim[1])
        if loading == 'cached':
            FrSeqDataset = FastFluxSeqDataset
        else:
            FrSeqDataset = RawKITTIDataset
    else:
        frshape = (3,) + new_dim
        flshape = (2,) + new_dim
        train_dir,valid_dir,test_dir = list_split_kitti_(new_dim[0],new_dim[1])
        FrSeqDataset = SeqDataset

    '''# Sequence buffers to avoid dataloader replication
    train_buffer = SeqBuffer(train_dir[0],train_dir[1],seq_len,stride=stride)
    valid_buffer = SeqBuffer(valid_dir[0],valid_dir[1],seq_len,stride=stride)
    test_buffer = SeqBuffer(test_dir[0],test_dir[1],seq_len,stride=stride)

    # Buffered datasets
    train_dataset = FrSeqDataset(train_dir[0],train_dir[1],seq_len,\
                      seq_buffer=train_buffer,transform=transf,stride=stride)
    valid_dataset = FrSeqDataset(valid_dir[0],valid_dir[1],seq_len,\
                      seq_buffer=valid_buffer,transform=transf,stride=stride)
    test_dataset = FrSeqDataset(test_dir[0],test_dir[1],seq_len,\
                      seq_buffer=test_buffer,transform=transf,stride=stride)'''

    # Datasets
    basedir = '/home/ubuntu/kitti/raw/'
    flowdir = '/home/ubuntu/kitti/flow/raw/'
    train_dataset = FrSeqDataset(basedir,flowdir,train_dd,seq_len,transform=transf,\
                     stride=stride,train=True,delay=delay)
    valid_dataset = FrSeqDataset(basedir,flowdir,valid_dd,seq_len,transform=transf,\
                     stride=stride,delay=delay)
    test_dataset = FrSeqDataset(basedir,flowdir,test_dd,seq_len,transform=transf,\
                     stride=stride,delay=delay)

    # Data loaders
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,\
                     num_workers=4,pin_memory=False) #,collate_fn=my_collate)
    valid_loader = DataLoader(valid_dataset,batch_size=batch_size,shuffle=False,\
                     num_workers=1,pin_memory=False) #,collate_fn=my_collate)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,\
                     num_workers=1,pin_memory=False) #,collate_fn=my_collate)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(device)

    '''args = Arguments()
    flow = models.FlowNet2S(args)
    if os.path.isfile(flow_fn):
        print('Loading existing flow model')
        checkpoint = torch.load(flow_fn)
        flow.load_state_dict(checkpoint['state_dict'])
        if not fine_tune_flow:
            flow.training = False
            for param in flow.parameters():
                param.requires_grad = False
        for layer in flow.modules():
            if isinstance(layer,nn.BatchNorm2d):
                layer.float()
    else:
        print('Flow model checkpoint not found')
        raise FileNotFoundError()
    flow = ImgFlowOdom(flow,flshape,h_dim,device=device).to(device)
    #flow = DummyFlow(flow,flshape,h_dim,device=device)'''

    model = VanAE_(flshape,h_dim)
    if enc_type == 'flat':
        enc = FlatIMUEncoder()
        h_dim = (2*flshape[1]*flshape[2])//4 + 17
        vo = Conv1dRecMapper((h_dim,strided_seq_len+delay),(strided_seq_len+delay,6),delay=delay)
    elif enc_type == 'imu':
        enc = StatXIMU()
        h_dim = 4*(flshape[1]+flshape[2])+17
        vo = Conv1dRecMapper((h_dim,strided_seq_len+delay),(strided_seq_len+delay,6),delay=delay)
    else:
        enc = VanillaEncoder((2,flshape[1],flshape[2]),h_dim)
        vo = Conv1dRecMapper((h_dim,strided_seq_len+delay),(strided_seq_len+delay,3),delay=delay) ###
    #vo = Conv1dMapper((h_dim,strided_seq_len),(strided_seq_len,6)).to(device)
    model.enc = enc
    model.dec = vo
    model.to(device)

    ##model = VanillaAutoencoder((1,)+new_dim).to(device)
    #model = VanillaAutoencoder((2,)+new_dim,h_dim).to(device)
    #model = MLPAutoencoder((2,)+new_dim,h_dim).to(device)
    #model = FastDirectOdometry((1,)+new_dim,(6,)).to(device)
    params = model.parameters()
    #optimizer = optim.Adam(params,lr=5e-5)
    optimizer = RAdam(params,lr=5e-5)
    min_loss = 1e15
    epoch = 0
    writer = SummaryWriter('/home/ubuntu/log/exp_flow_net_h{}_l{}_s{}_{}x{}'\
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

    #loss_fn = torch.nn.MSELoss()
    loss_fn = seq_pose_loss_se3
    k,kv = 0,0
    #flow.train()
    #flow.training = True #False
    #model.to(device)
    for i in range(epoch,num_epochs):
        print('Epoch',i)
        model.train()
        losses = []
        for x,imu,y,_ in train_loader:
            torch.cuda.empty_cache()
            x,imu,y = x.to(device), imu.to(device), y.to(device)
            print(x.size(),imu.size(),y.size())
            optimizer.zero_grad()
            #x = flow(x)[0]
            #print(x.size())
            y_ = model(x,imu)
            loss = loss_fn(y,y_,delay)
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
        model.eval()
        v_losses = []
        for x,imu,y,_ in valid_loader:
            torch.cuda.empty_cache()
            x,imu,y = x.to(device), imu.to(device), y.to(device)
            print(x.size(),imu.size(),y.size())
            #x = flow(x)[0]
            #print(x.size())
            y_ = model(x,imu)
            loss = loss_fn(y_,y,delay)
            v_losses.append(loss.item())
            writer.add_scalar('valid_cost',loss.item(),kv)
            kv += 1
        #writer.add_embedding(y[0,:,[3,7,11]],tag='gt_pts_{}'.format(i),global_step=1)
        #writer.add_embedding(y_[0,:,[3,7,11]],tag='est_pts_{}'.format(i),global_step=1)
        wid = np.random.randint(len(y))
        plot_yy_(y[wid][delay:],y_[wid][delay:],device,writer) ###
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
    plot_eval_(model,test_loader,strided_seq_len,delay=delay,device=device,logger=writer)
    writer.close()
    print('End of plot_eval')