import sys
import signal
import numpy as np
import glob, os
import pickle
import torch
import torch.optim as optim
from pt_ae import VanillaAutoencoder

batch_size = 32
wlen = 128
stride = wlen
seq_len = 128
valid_ids = 128
num_classes = 5
num_epochs = 10
__train = sys.argv[1]

input_fn = '/home/ronnypetson/Documents/deep_odometry/kitti/dataset_frames/sequences/flows_128x128/flows_128x128_26_30.npy'
#input_fn_poses = '/home/ronnypetson/Documents/deep_odometry/kitti/dataset_frames/sequences/flows_128x128/poses_flat_26-30.npy'
model_fn = '/home/ronnypetson/models/pt/ae0_.pth'

min_loss = 0.0
epoch = 0

def evaluate(model,data_x,loss_fn,device):
    model.eval()
    losses = []
    for i in range(0,len(data_x),batch_size):
        x = data_x[i:i+batch_size].to(device)
        y_ = model(x)
        loss = loss_fn(y_,x)
        losses.append(loss.item())
    mean_loss = np.mean(losses)
    print('Validation: {}'.format(mean_loss))
    return mean_loss

if __name__ == '__main__':
    # Load the data
    frames = np.load(input_fn).transpose(0,3,1,2)
    print(frames.shape)

    # Group the data
    # ...

    device = torch.device('cuda:0')
    print(device)
    frames = torch.tensor(frames,requires_grad=False).float()
    frames_valid = frames[:valid_ids]
    frames = frames[valid_ids:]

    model = VanillaAutoencoder(frames.size()[1:]).to(device)
    params = model.parameters()
    optimizer = optim.Adam(params,lr=3e-4) # optim.SGD(params,lr=1e-3,momentum=0.1)

    if os.path.isfile(model_fn):
        checkpoint = torch.load(model_fn)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        min_loss = checkpoint['min_loss']
        epoch = checkpoint['epoch']

    if __train != '0': # Train
        model.train()
        loss_fn = torch.nn.MSELoss()
        num_iter = len(frames)//batch_size
        all_ids = np.random.choice(len(frames),[num_epochs,num_iter,batch_size])
        for j in range(epoch,num_epochs):
            for i in range(num_iter):
                optimizer.zero_grad()
                ids = all_ids[j][i]
                x = frames[ids].to(device)
                y_ = model(x)
                loss = loss_fn(y_,x)
                loss.backward()
                optimizer.step()
                if i%50 == 0:
                    print('Epoch {}: {}'.format(j,loss.item()))
                    loss = evaluate(model,frames_valid,loss_fn,device)
                    model.train()
                    if loss < min_loss:
                        min_loss = loss
                        torch.save({'model_state': model.state_dict(),
                                    'optimizer_state': optimizer.state_dict(),
                                    'min_loss': min_loss,
                                    'epoch': j}, model_fn)
    else: # Evaluate
        loss_fn = torch.nn.MSELoss()
        evaluate(model,frames_valid,loss_fn,device)
