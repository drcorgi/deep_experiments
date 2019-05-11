import sys
import signal
import time
import numpy as np
import glob, os
import pickle
import torch
import torch.optim as optim

from pt_ae import Conv1dMapper
from plotter import get_3d_points__, c3dto2d, abs2relative, plot_3d_points_, plot_abs

class MapTrainer():
    def __init__(self,model,model_fn,batch_size,valid_ids,device):
        self.model_fn = model_fn
        self.batch_size = batch_size
        self.valid_ids = valid_ids
        self.min_loss = 1e15
        self.epoch = 0
        self.device = device
        self.loss_fn = torch.nn.MSELoss()
        self.model = model
        params = self.model.parameters()
        self.optimizer = optim.Adam(params,lr=3e-4)
        if os.path.isfile(model_fn):
            print('Loading existing model')
            checkpoint = torch.load(model_fn)
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.min_loss = checkpoint['min_loss']
            self.epoch = checkpoint['epoch']
        else:
            print('Creating new model')

    def plot_eval(self,data_x,data_y,abs_,seq_len):
        assert len(data_x) == len(data_y)
        self.model.eval()
        rel_poses = []
        for i in range(0,len(data_x),self.batch_size):
            x = data_x[i:i+self.batch_size].to(device)
            if len(x) > 0:
                y_ = self.model(x)
                rel_poses += y_.cpu().detach().numpy().tolist()
        rel_poses = np.array(rel_poses).transpose(0,2,1)
        pts = get_3d_points__(rel_poses,seq_len)
        gt = data_y.cpu().detach().numpy().transpose(0,2,1)
        gt = get_3d_points__(gt,seq_len)
        print(gt.shape,pts.shape,abs_.shape)
        if not os.path.isdir('tmp'):
            os.mkdir('tmp')
        t = time.time()
        plot_3d_points_(gt,pts,'tmp/{}_projections_xyz.png'.format(t))
        plot_abs(abs_,pts,'tmp/{}_absolute_gt_3d.png'.format(t))

    def evaluate(self,data_x,data_y):
        assert len(data_x) == len(data_y)
        self.model.eval()
        losses = []
        for i in range(0,len(data_x),self.batch_size):
            x = data_x[i:i+self.batch_size].to(self.device)
            y = data_y[i:i+self.batch_size].to(self.device)
            y_ = self.model(x)
            loss = self.loss_fn(y_,y)
            losses.append(loss.item())
        mean_loss = np.mean(losses)
        print('Evaluation:\t{}\t'.format(mean_loss),end='')
        return mean_loss

    def train(self,frames,poses,num_epochs):
        assert len(frames) == len(poses)
        frames_valid = frames[:self.valid_ids]
        frames = frames[self.valid_ids:]
        poses_valid = poses[:self.valid_ids]
        poses = poses[self.valid_ids:]

        self.model.train()
        num_iter = len(frames)//self.batch_size
        all_ids = np.random.choice(len(frames),[num_epochs,num_iter,self.batch_size])
        for j in range(self.epoch,num_epochs):
            loss = self.evaluate(frames_valid,poses_valid)
            self.model.train()
            if loss < self.min_loss:
                self.min_loss = loss
                torch.save({'model_state': self.model.state_dict(),
                            'optimizer_state': self.optimizer.state_dict(),
                            'min_loss': self.min_loss,
                            'epoch': j}, self.model_fn)
            losses = []
            for i in range(num_iter):
                self.optimizer.zero_grad()
                ids = all_ids[j][i]
                x = frames[ids].to(self.device)
                y = poses[ids].to(self.device)
                y_ = self.model(x)
                loss = self.loss_fn(y_,y)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            print('Epoch {}:\t{}'.format(j,np.mean(losses)))

if __name__ == '__main__':
    if len(sys.argv) != 10:
        print('Input:',sys.argv)
        print('Usage: input_fn input_fn_poses model_fn batch_size valid_ids test_ids epochs seq_len device')
        exit()

    input_fn = sys.argv[1] #'/home/ronnypetson/Documents/deep_odometry/kitti/dataset_frames/sequences/emb0_128x128/emb0_flows_128x128_00-10.pck'
    input_fn_poses = sys.argv[2] #'/home/ronnypetson/Documents/deep_odometry/kitti/dataset_frames/sequences/poses_00-10.pck'
    model_fn = sys.argv[3] #'/home/ronnypetson/models/pt/test_mapper_.pth'
    batch_size = int(sys.argv[4]) #8
    valid_ids = int(sys.argv[5]) #256
    test_ids = int(sys.argv[6]) #256
    epochs = int(sys.argv[7])
    seq_len = int(sys.argv[8]) # 16
    device = sys.argv[9] #'cuda:0'

    # Load the data
    with open(input_fn,'rb') as f:
        frames = pickle.load(f)
    with open(input_fn_poses,'rb') as f:
        abs_poses = pickle.load(f)
    abs_poses = [[c3dto2d(p) for p in s] for s in abs_poses]

    # Group the data
    frames = [[s[i:i+seq_len] for i in range(len(s)-seq_len+1)] for s in frames]
    frames = [np.array(s) for s in frames]
    frames = np.concatenate(frames,axis=0).transpose(0,2,1)
    #abs_poses = [s for s in abs_poses] #[s[:-1] for s in abs_poses]
    rel_poses = [abs2relative(s,seq_len,1) for s in abs_poses]
    rel_poses = np.concatenate(rel_poses,axis=0).transpose(0,2,1)
    print(frames.shape,rel_poses.shape)

    device = torch.device(device)
    model = Conv1dMapper(frames.shape[1:],rel_poses.shape[1:]).to(device).half()
    t = MapTrainer(model=model,model_fn=model_fn,batch_size=batch_size\
                   ,valid_ids=valid_ids,device=device)

    # Split data into test/train
    frames = torch.tensor(frames).float().half()
    frames_test = frames[:test_ids]
    frames = frames[test_ids:]
    rel_poses = torch.tensor(rel_poses).float().half()
    rel_poses_test = rel_poses[:test_ids]
    rel_poses = rel_poses[test_ids:]

    # Train and test
    t.train(frames,rel_poses,epochs)
    t.evaluate(frames_test,rel_poses_test)
    abs_poses = abs_poses[0][:test_ids]
    abs_poses = abs2relative(abs_poses,len(abs_poses),1)[0]
    t.plot_eval(frames_test,rel_poses_test,abs_poses,seq_len)
