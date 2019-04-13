import sys
import signal
import numpy as np
import glob, os
import pickle
import torch
import torch.optim as optim

from pt_ae import Conv1dMapper
from plotter import __get_3d_points
from plotter import *

batch_size = 128
wlen = 64
stride = wlen
seq_len = 16
valid_ids = 1024
num_epochs = 900
__flag = sys.argv[1]

input_fn = '/home/ronnypetson/Documents/deep_odometry/kitti/dataset_frames/sequences/emb0_128x128/emb0_flows_128x128_00-10.pck'
#input_fn_poses = '/home/ronnypetson/Documents/deep_odometry/kitti/dataset_frames/sequences/flows_128x128/poses_flat_26-30.npy'
#input_fn = '/home/ronnypetson/Documents/deep_odometry/kitti/dataset_frames/sequences/flows_00-10_128x128.pck'
input_fn_poses = '/home/ronnypetson/Documents/deep_odometry/kitti/dataset_frames/sequences/poses_00-10.pck'
model_fn = '/home/ronnypetson/models/pt/mapper_.pth'

min_loss = 1e15
epoch = 0

def plot_eval(model,data_x,data_y,abs_,device):
    model.eval()
    rel_poses = []
    for i in range(0,len(data_x),batch_size):
        x = data_x[i:i+batch_size].to(device)
        y_ = model(x)
        if len(y_) > 0:
            rel_poses += y_.cpu().detach().numpy().tolist()
    rel_poses = np.array(rel_poses).transpose(0,2,1)
    pts = __get_3d_points(rel_poses,seq_len)
    gt = data_y.cpu().detach().numpy().transpose(0,2,1)
    gt = __get_3d_points(gt,seq_len)
    print(gt.shape,pts.shape,abs_.shape)
    plot_3d_points_(gt,pts)
    plot_abs(abs_,pts)

def evaluate(model,data_x,data_y,loss_fn,device):
    model.eval()
    losses = []
    for i in range(0,len(data_x),batch_size):
        x = data_x[i:i+batch_size].to(device)
        y = data_y[i:i+batch_size].to(device)
        y_ = model(x)
        loss = loss_fn(y_,y)
        losses.append(loss.item())
    mean_loss = np.mean(losses)
    print('Validation: {}'.format(mean_loss))
    return mean_loss

if __name__ == '__main__':
    # Load the data
    #frames = np.load(input_fn) #[:4540+valid_ids]
    #abs_poses = np.load(input_fn_poses) #[:4540+valid_ids]
    with open(input_fn,'rb') as f:
        frames = pickle.load(f)
    with open(input_fn_poses,'rb') as f:
        abs_poses = pickle.load(f)

    # Group the data
    '''frames = np.array([frames[i:i+seq_len] for i in range(len(frames)-seq_len+1) if i not in range(4540-seq_len+1,4540)])
    frames = frames.transpose(0,2,1)
    rel_poses_t = abs2relative(abs_poses[:4540],seq_len,1)
    rel_poses_v = abs2relative(abs_poses[4540:4540+valid_ids],seq_len,1)
    rel_poses = np.concatenate((rel_poses_t,rel_poses_v),axis=0)
    rel_poses = rel_poses.transpose(0,2,1)
    print(frames.shape,rel_poses.shape)'''
    frames = [[s[i:i+seq_len] for i in range(len(s)-seq_len+1)] for s in frames]
    frames = [np.array(s) for s in frames]
    frames = np.concatenate(frames,axis=0).transpose(0,2,1)
    abs_poses = [p[:-1] for p in abs_poses]
    rel_poses = [abs2relative(s,seq_len,1) for s in abs_poses]
    rel_poses = np.concatenate(rel_poses,axis=0).transpose(0,2,1)
    print(frames.shape,rel_poses.shape)

    device = torch.device('cuda:0')
    print(device)
    frames = torch.tensor(frames).float()
    frames_valid = frames[-valid_ids:]
    frames = frames[:-valid_ids]
    rel_poses = torch.tensor(rel_poses).float()
    rel_poses_valid = rel_poses[-valid_ids:]
    rel_poses = rel_poses[:-valid_ids]

    model = Conv1dMapper(frames.size()[1:],rel_poses.size()[1:]).to(device)
    params = model.parameters()
    optimizer = optim.Adam(params,lr=1e-3) # optim.SGD(params,lr=1e-3,momentum=0.1)

    if os.path.isfile(model_fn):
        checkpoint = torch.load(model_fn)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        min_loss = checkpoint['min_loss']
        epoch = checkpoint['epoch']
    else:
        print('Creating new model')

    if __flag == '1': # Train
        model.train()
        loss_fn = torch.nn.MSELoss()
        num_iter = len(frames)//batch_size
        all_ids = np.random.choice(len(frames),[num_epochs,num_iter,batch_size])
        for j in range(epoch,num_epochs):
            loss = evaluate(model,frames_valid,rel_poses_valid,loss_fn,device)
            model.train()
            if loss < min_loss:
                min_loss = loss
                torch.save({'model_state': model.state_dict(),
                            'optimizer_state': optimizer.state_dict(),
                            'min_loss': min_loss,
                            'epoch': j}, model_fn)
            losses = []
            for i in range(num_iter):
                optimizer.zero_grad()
                ids = all_ids[j][i]
                x = frames[ids].to(device)
                y = rel_poses[ids].to(device)
                y_ = model(x)
                loss = loss_fn(y_,y)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            print('Epoch {}: {}'.format(j,np.mean(losses)))
    else: # Evaluate
        loss_fn = torch.nn.MSELoss()
        evaluate(model,frames_valid,rel_poses_valid,loss_fn,device)
        abs_poses = abs_poses[-1][-valid_ids-seq_len+1:]
        abs_poses = abs2relative(abs_poses,len(abs_poses),1)[0]
        #abs_poses = np.concatenate(abs_poses,axis=0).transpose(0,2,1)
        print(abs_poses.shape)
        plot_eval(model,frames_valid,rel_poses_valid,abs_poses,device)
