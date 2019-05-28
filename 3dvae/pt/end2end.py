import sys
import signal
import time
import numpy as np
import glob, os
import pickle
import torch
import torch.optim as optim

from pt_ae import Conv1dMapper, Conv1dRecMapper, MLPMapper, DirectOdometry
from plotter import get_3d_points__, get_3d_points_t, c3dto2d, abs2relative, abs2relative_, plot_3d_points_, plot_abs
from odom_dataset import Rescale, FluxToTensor, my_collate

class FluxH5OdomDataset(Dataset):
    def __init__(self, file_path, odom_path, seq_len, chunk_size, transform=None):
        super(FluxH5OdomDataset, self).__init__()
        h5_file = h5py.File(file_path)
        self.data = h5_file.get('frames')
        with open(self.odom_path,'rb') as f:
            self.abs_odom = pickle.load(f)
        self.seq_len = seq_len
        self.transform = transform
        self.chunk_size = chunk_size

    def __getitem__(self, index):
        try:
            if index + self.seq_len + 1 > self.__len__():
                raise Exception('index out of range for sequence')

            x = self.data[index//self.chunk_size][index%self.chunk_size]
            y = self.abs_odom[index]

            next = index+1

            x_ = self.data[next//self.chunk_size][next%self.chunk_size]

            if self.transform[0]:
                x = self.transform[0](x)
                x_ = self.transform[0](x_)

            x = cv2.calcOpticalFlowFarneback(x,x_,None,0.5,3,15,3,5,1.2,0)

            if self.transform[1]:
                x = self.transform[1](x)

            return x,y

        except Exception as e:

            print(e)

    def __len__(self):
        return self.chunk_size*self.data.shape[0]

if __name__ == '__main__':
    if len(sys.argv) != 10:
        print('Input:',sys.argv)
        print('Usage: input_fn input_fn_poses model_fn batch_size valid_ids test_ids epochs seq_len device')
        exit()

    input_fn = sys.argv[1] #'/home/ronnypetson/Documents/deep_odometry/kitti/frames_odom_train.h5'
    input_fn_poses = sys.argv[2] #

    model_fn = sys.argv[3] #'/home/ronnypetson/models/pt/test_mapper_.pth'
    batch_size = int(sys.argv[4]) #8
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
    rel_poses = [abs2relative(s,seq_len,1) for s in abs_poses] ## abs2relative
    rel_poses = np.concatenate(rel_poses,axis=0).transpose(0,2,1)
    print(frames.shape,rel_poses.shape)

    device = torch.device(device)
    #model = Conv1dMapper(frames.shape[1:],rel_poses.shape[1:]).to(device) #.half()
    #model = MLPMapper(frames.shape[1:],rel_poses.shape[1:]).to(device)
    #model = Conv1dRecMapper(frames.shape[1:],rel_poses.shape[1:]).to(device)
    model = DirectOdometry((2,)+new_dim,(12,),h_dim)
