import sys
import signal
import numpy as np
import glob, os
import pickle
import torch
import torch.optim as optim
from pt_ae import Conv1dMapper

batch_size = 128
wlen = 128
stride = wlen
seq_len = 64
valid_ids = 128
num_epochs = 50
__flag = sys.argv[1]

input_fn = '/home/ronnypetson/Documents/deep_odometry/kitti/dataset_frames/sequences/emb0_128x128/emb0_128x128_26_30.npy'
input_fn_poses = '/home/ronnypetson/Documents/deep_odometry/kitti/dataset_frames/sequences/flows_128x128/poses_flat_26-30.npy'
model_fn = '/home/ronnypetson/models/pt/mapper_.pth'

min_loss = 1e15
epoch = 0

def evaluate(model,data_x,data_y,loss_fn,device):
    model.eval()
    losses = []
    for i in range(0,len(data_x),batch_size):
        x = data_x[i:i+batch_size].to(device)
        y = data_y[i:i+batch_size].to(device)
        y_ = model(x)
        #print(y_[0].transpose(0,1)[8])
        #print(y[0].transpose(0,1)[8])
        loss = loss_fn(y_,y)
        losses.append(loss.item())
    mean_loss = np.mean(losses)
    print('Validation: {}'.format(mean_loss))
    return mean_loss

if __name__ == '__main__':
    # Load the data
    frames = np.load(input_fn)
    abs_poses = np.load(input_fn_poses)

    # Group the data
    frames = np.array([frames[i:i+seq_len] for i in range(len(frames)-seq_len+1)])
    frames = frames.transpose(0,2,1)
    rel_poses = np.array([[abs_poses[i+j]-abs_poses[i] for j in range(seq_len)]\
                           for i in range(len(abs_poses)-seq_len+1)])
    rel_poses = rel_poses.transpose(0,2,1)
    print(frames.shape,rel_poses.shape)

    device = torch.device('cuda:0')
    print(device)
    frames = torch.tensor(frames).float()
    frames_valid = frames[:valid_ids]
    frames = frames[valid_ids:]
    rel_poses = torch.tensor(rel_poses).float()
    rel_poses_valid = rel_poses[:valid_ids]
    rel_poses = rel_poses[valid_ids:]

    model = Conv1dMapper(frames.size()[1:],rel_poses.size()[1:]).to(device)
    params = model.parameters()
    optimizer = optim.Adam(params,lr=3e-4) # optim.SGD(params,lr=1e-3,momentum=0.1)

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
            for i in range(num_iter):
                optimizer.zero_grad()
                ids = all_ids[j][i]
                x = frames[ids].to(device)
                y = rel_poses[ids].to(device)
                y_ = model(x)
                loss = loss_fn(y_,y)
                loss.backward()
                optimizer.step()
            print('Epoch {}: {}'.format(j,loss.item()))
            loss = evaluate(model,frames_valid,rel_poses_valid,loss_fn,device)
            model.train()
            if loss < min_loss:
                min_loss = loss
                torch.save({'model_state': model.state_dict(),
                            'optimizer_state': optimizer.state_dict(),
                            'min_loss': min_loss,
                            'epoch': j+1}, model_fn)
    else: # Evaluate
        loss_fn = torch.nn.MSELoss()
        evaluate(model,frames_valid,rel_poses_valid,loss_fn,device)
