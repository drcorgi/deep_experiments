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
from plotter import c3dto2d, abs2relative, plot_eval, SE3tose3, se3toSE3, homogen,\
SE2tose2, se2toSE2, flat_homogen, c3dto2d_
from odom_loader import load_kitti_odom, load_raw_kitti_odom_imu, load_raw_kitti_img_odom_imu
from tensorboardX import SummaryWriter
from time import time
from liegroups import SE3, SE2

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

def my_collate_(batch):
    batch_x = []
    batch_imu = []
    batch_y = []
    batch_abs = []
    for b in batch:
        if b is not None:
            batch_x.append(b[0])
            batch_imu.append(b[1])
            batch_y.append(b[2])
            batch_abs.append(b[3])
    return torch.stack(batch_x), torch.stack(batch_imu), torch.stack(batch_y), batch_abs

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

def list_split_kitti():
    base = '/home/ubuntu/kitti/dataset/'
    all_seqs = [sorted(glob(base+'sequences/{:02d}/image_0/*.png'\
                .format(i))) for i in range(11)]
    all_poses = [base+'poses/{:02d}.txt'.format(i) for i in range(11)]
    train_seqs, train_poses = all_seqs[:8], all_poses[:8]
    valid_seqs, valid_poses = all_seqs[8:], all_poses[8:]
    test_seqs, test_poses = all_seqs[-1:], all_poses[-1:]
    return (train_seqs,train_poses), (valid_seqs,valid_poses), (test_seqs,test_poses)

def list_split_kitti_(h,w):
    base = '/home/ubuntu/kitti/{}x{}/'.format(h,w)
    pbase = '/home/ubuntu/kitti/dataset/'
    all_seqs = [sorted(glob(base+'{:02d}/*.png'\
                .format(i))) for i in range(11)]
    all_poses = [pbase+'poses/{:02d}.txt'.format(i) for i in range(11)]
    train_seqs, train_poses = all_seqs[:8], all_poses[:8]
    valid_seqs, valid_poses = all_seqs[8:], all_poses[8:]
    test_seqs, test_poses = all_seqs[-1:], all_poses[-1:]
    return (train_seqs,train_poses), (valid_seqs,valid_poses), (test_seqs,test_poses)

def list_split_raw_kitti():
    basedir = '/home/ubuntu/kitti/raw/'
    debug_msg = 'load_raw_kitti_odom_imu'
    dates = [d for d in os.listdir(basedir) if os.path.isdir(basedir+d)]
    print(debug_msg,'dates:',dates)
    dates_drives = []
    for d in dates:
        dates_drives += [(d,drv[17:-5]) for drv in\
                         os.listdir(basedir+'/'+d+'/')\
                         if os.path.isdir(basedir+d+'/'+drv)]
    print(debug_msg,'dates_drives:',dates_drives)
    valid_ids = [8,9,10,11,12,13,14,15,16] #[8,9,10]
    train_ids = [i for i in range(len(dates_drives)) if i not in valid_ids] #[0,1,2,3,4,5,6,7]
    test_ids = [9] #[10]
    '''odom_imu = []
    for dd in dates_drives:
        data = pykitti.raw(basedir,dd[0],dd[1])
        odom_imu.append([(o.T_w_imu,o.packet[6:23]) for o in data.oxts])
    return odom_imu'''
    train_seqs = [dates_drives[i] for i in train_ids]
    valid_seqs = [dates_drives[i] for i in valid_ids]
    test_seqs = [dates_drives[i] for i in test_ids]
    return train_seqs, valid_seqs, test_seqs

def list_split_kitti_flow(h,w):
    base = '/home/ubuntu/kitti/flow/{}x{}_flownet_1/'.format(h,w)
    pbase = '/home/ubuntu/kitti/dataset/'

    all_seqs = []
    for i in range(11):
        fns = base+'{:02d}/*.npy'.format(i)
        fns = sorted(glob(fns),key=lambda x:int(x[44:-4]))
        all_seqs.append(fns)

    all_poses = [pbase+'poses/{:02d}.txt'.format(i) for i in range(11)]
    train_ids = [0,2,8,9]
    valid_ids = [1,3,4,5,6,7,10]

    train_seqs = [all_seqs[i] for i in train_ids]
    train_poses = [all_poses[i] for i in train_ids]
    valid_seqs = [all_seqs[i] for i in valid_ids]
    valid_poses = [all_poses[i] for i in valid_ids]
    test_seqs, test_poses = all_seqs[10:], all_poses[10:]

    return (train_seqs,train_poses), (valid_seqs,valid_poses), (test_seqs,test_poses)

def list_split_kitti_flux(h,w):
    base = '/home/ubuntu/kitti/flux/{}x{}/'.format(h,w)
    pbase = '/home/ubuntu/kitti/dataset/'
    all_seqs = [sorted(glob(base+'{:02d}/*.npy'\
                .format(i))) for i in range(11)]
    all_poses = [pbase+'poses/{:02d}.txt'.format(i) for i in range(11)]
    train_seqs, train_poses = all_seqs[:8], all_poses[:8] # 0,2,8,9
    valid_seqs, valid_poses = all_seqs[8:], all_poses[8:] # 0:1
    test_seqs, test_poses = all_seqs[9:10], all_poses[9:10] # 1:2, 8:9
    return (train_seqs,train_poses), (valid_seqs,valid_poses), (test_seqs,test_poses)

class FluxSeqDataset(Dataset):
    def __init__(self, fnames, pfnames, seq_len, transform=None, stride=1, train=False, delay=1):
        ''' fnames is a list of lists of file names
            pfames is a list of file names (one for each entire sequence)
        '''
        super().__init__()
        self.fnames = fnames
        self.pfnames = pfnames
        self.len = sum([max(0,len(fns)-seq_len+1) for fns in fnames])
        self.frames_len = sum([len(fns) for fns in fnames])
        self.sids = []
        for i,fns in enumerate(fnames):
            for j in range(len(fns)-seq_len+1):
                self.sids.append(i)
        self.fsids = []
        for fns in fnames:
            for i in range(len(fns)-seq_len+1):
                self.fsids.append(i)
        self.seq_len = seq_len
        self.transform = transform # Transform at the frame level
        self.buffer = [] # index -> (x,abs)
        self.aposes = [load_kitti_odom(fn) for fn in self.pfnames]
        self.fshape = np.load(self.fnames[0][0]).transpose(2,0,1).shape
        self.load()
        self.train = train
        self.delay = delay

    def load(self):
        try:
            print('Cacheing frames')
            for s,fns in enumerate(self.fnames):
                seq_ = []
                for id,fn in enumerate(fns):
                    frame = np.load(fn)
                    frame = frame.transpose(2,0,1)
                    if self.transform:
                        frame = self.transform(frame)
                    abs = self.aposes[s][id]
                    p = c3dto2d(abs)
                    p = SE2tose2([p])[0]
                    seq_.append((frame,p,p))
                self.buffer.append(seq_)
        except RuntimeError as re:
            print('frames missed',re)
        except Exception as e:
            print('frames not loaded',e)

    def __getitem__(self, index):
        try:
            d = self.delay
            #print(d,end=' ')
            s,id = self.sids[index], self.fsids[index]
            s_ = s #max(0,s-1) if id == 0 else s
            aug = (np.random.randint(2) == 0) and self.train
            #id__ = min(len(self.buffer[s_])-1,id+self.seq_len) if aug else max(0,id-1)
            id_ = [min(len(self.buffer[s_])-1,id+self.seq_len+i)\
                   if aug else max(0,id-i-1) for i in range(d-1,-1,-1)]

            x = torch.zeros((self.seq_len+d,)+self.fshape)
            y = np.zeros((self.seq_len+d,3))
            abs = np.zeros((self.seq_len,3))

            #print(id_,id__,end=' ')
            for j,i in enumerate(id_):
                x[j] = self.buffer[s_][i][0]
            #y[:d] = self.buffer[s_][id+self.seq_len-1 if aug else id][1] #np.zeros(3)
            #x[0] = self.buffer[s_][id__][0]
            i_ = id
            for i in range(id,id+self.seq_len): ### Fix sampling
                if self.train:
                    i_ = min(i_+np.random.randint(3),len(self.buffer[s])-1)
                else:
                    i_ = i
                x[i-id+d],y[i-id+d],abs[i-id] = self.buffer[s][i_]

            # Data aug
            if aug:
                x[d:] = -torch.flip(x[d:],dims=[0])
                y[d:] = np.flip(y[d:],axis=0)

            inert_ = SE2.exp(y[d]).inv()
            #inert_ = SE2.from_matrix(homogen(y[1]),normalize=True).inv()
            #inert_ = np.linalg.inv(homogen(y[1]))
            #y[1:] = np.array([flat_homogen(np.dot(inert_,homogen(p))) for p in y[1:]])
            y[d:] = np.array([inert_.dot(SE2.exp(p)).log() for p in y[d:]])
            #y[1:] = np.array([flat_homogen(\
            #                  inert_.dot(SE2.from_matrix(homogen(p),\
            #                             normalize=True))\
            #                             .as_matrix()) for p in y[1:]])
            #y[1:] = abs2relative(y[1:],self.seq_len,1)[0]
            #y[1:,[0,1]] /= np.linalg.norm(y[-1,[0,1]])+1e-12
            # Normalize translation
            #y = SE3tose3(y) ###
            y = torch.from_numpy(y).float()
            return x,y,abs
        except RuntimeError as re:
            print('-',re)
        except Exception as e:
            print('--',index,e)

    def __len__(self):
        return self.len

class RawKITTIDataset(Dataset):
    ''' basedir -> flow files names
    '''
    def __init__(self, basedir, flowdir, dates_drives,\
                 seq_len, transform=None, stride=1,\
                 train=False, delay=1):
        super().__init__()
        fnames = self.get_ffnames(basedir,dates_drives)
        self.fnames = fnames # create fnames as list of lists of flow file names
        self.len = sum([max(0,len(fns)-seq_len+1) for fns in fnames])
        self.frames_len = sum([len(fns) for fns in fnames])
        self.sids = []
        for i,fns in enumerate(fnames):
            for j in range(len(fns)-seq_len+1):
                self.sids.append(i)
        self.fsids = []
        for fns in fnames:
            for i in range(len(fns)-seq_len+1):
                self.fsids.append(i)
        self.seq_len = seq_len
        self.transform = transform # Transform at the frame level
        self.buffer = [] # index -> (x,abs)
        self.imgs, self.aposes, self.imu =\
                     load_raw_kitti_img_odom_imu(basedir,dates_drives)  ###
        #exit()
        #print(len(self.aposes), len(self.imu))
        self.fshape = (2,32,128) #cv2.imread(self.fnames[0][0]).transpose(2,0,1).shape
        self.load()
        self.train = train
        self.delay = delay

    def get_ffnames(self,flowdir,dates_drives):
        fns = []
        for dd in dates_drives:
            bdd = flowdir+'/'+dd[0]+'/'\
                  +dd[0]+'_drive_'+dd[1]+'_sync/image_00/data/*.png'
            fns.append(sorted(glob(bdd),key=lambda x:int(x[-10:-4])))
        #print(fns)
        return fns

    def load(self):
        try:
            print('Cacheing frames')
            for s,fns in enumerate(self.fnames):
                seq_ = []
                for id,fn in enumerate(fns):
                    id_ = min(id+1,len(self.imgs[s])-1)
                    frame, frame_ = self.imgs[s][id], self.imgs[s][id_] #np.load(fn)
                    frame = cv2.calcOpticalFlowFarneback\
                            (frame,frame_,None,0.5,3,15,3,5,1.2,0)
                    frame = frame.transpose(2,0,1)
                    if self.transform:
                        frame = self.transform(frame)
                    abs = self.aposes[s][id]
                    p = c3dto2d_(abs) ###
                    p = SE3tose3([p])[0] ### # dim 6
                    imu = self.imu[s][id] # dim 17
                    seq_.append((frame,p,p,imu))
                self.buffer.append(seq_)
        except RuntimeError as re:
            print('frames missed',re)
        except Exception as e:
            print('frames not loaded',e)
            raise e

    def __getitem__(self, index):
        try:
            d = self.delay
            s,id = self.sids[index], self.fsids[index]
            s_ = s
            aug = (np.random.randint(2) == 0) and self.train and False
            id_ = [min(len(self.buffer[s_])-1,id+self.seq_len+i)\
                   if aug else max(0,id-i-1) for i in range(d-1,-1,-1)]

            x = torch.zeros((self.seq_len+d,)+self.fshape)
            y = np.zeros((self.seq_len+d,6))
            abs = np.zeros((self.seq_len,6))
            imu = np.zeros((self.seq_len+d,17))

            for j,i in enumerate(id_):
                x[j] = self.buffer[s_][i][0]

            i_ = id
            for i in range(id,id+self.seq_len):
                if self.train:
                    i_ = min(i_+np.random.randint(3),len(self.buffer[s])-1)
                else:
                    i_ = i
                x[i-id+d],y[i-id+d],abs[i-id],imu[i-id+d] = self.buffer[s][i_]
                #_,y[i-id+d],abs[i-id],imu[i-id+d] = self.buffer[s][i_]

            # Data aug
            if aug:
                x[d:] = -torch.flip(x[d:],dims=[0])
                y[d:] = np.flip(y[d:],axis=0)
                imu[d:] = -np.flip(imu[d:],axis=0) # ??

            inert_ = SE3.exp(y[d]).inv()
            y[d:] = np.array([inert_.dot(SE3.exp(p)).log() for p in y[d:]])
            imu[d:] = np.cumsum(imu[d:],axis=0)
            
            y = torch.from_numpy(y).float()
            imu = torch.from_numpy(imu).float()
            abs = torch.from_numpy(abs).float()
            #yimu = torch.cat([y,imu],dim=1)
            return x,imu,y,abs
        except RuntimeError as re:
            print('-',re)
        except Exception as e:
            print('--',index,e)
            raise e

    def __len__(self):
        return self.len

class SeqDataset(Dataset):
    def __init__(self, fnames, pfnames, seq_len, transform=None, stride=1):
        ''' fnames is a list of lists of file names
            pfames is a list of file names (one for each entire sequence)
        '''
        super().__init__()
        self.fnames = fnames
        self.pfnames = pfnames
        self.len = sum([max(0,len(fns)-seq_len+1) for fns in fnames])
        self.frames_len = sum([len(fns) for fns in fnames])
        self.sids = []
        for i,fns in enumerate(fnames):
            for j in range(len(fns)-seq_len+1):
                self.sids.append(i)
        self.fsids = []
        for fns in fnames:
            for i in range(len(fns)-seq_len+1):
                self.fsids.append(i)
        self.seq_len = seq_len
        self.transform = transform # Transform at the frame level
        self.buffer = [] # index -> (x,abs)
        self.aposes = [load_kitti_odom(fn) for fn in self.pfnames]
        self.fshape = cv2.imread(self.fnames[0][0]).transpose(2,0,1).shape
        self.load()

    def load(self):
        try:
            print('Cacheing frames')
            for s,fns in enumerate(self.fnames):
                seq_ = []
                for id,fn in enumerate(fns):
                    frame = cv2.imread(fn) #np.load(fn)
                    frame = frame.transpose(2,0,1)
                    if self.transform:
                        frame = self.transform(frame)
                    abs = self.aposes[s][id]
                    p = c3dto2d(abs)
                    seq_.append((frame,p,abs))
                self.buffer.append(seq_)
        except RuntimeError as re:
            print('frames missed',re)
        except Exception as e:
            print('frames not loaded',e)

    def __getitem__(self, index):
        try:
            s,id = self.sids[index], self.fsids[index]
            x = torch.zeros((self.seq_len,)+self.fshape)
            y = np.zeros((self.seq_len,12))
            abs = np.zeros((self.seq_len,12))
            for i in range(id,id+self.seq_len):
                x[i-id],y[i-id],abs[i-id] = self.buffer[s][i]
            y = abs2relative(y,self.seq_len,1)[0]
            y = torch.from_numpy(y).float()
            return x,y,abs
        except RuntimeError as re:
            print('-',re)
        except Exception as e:
            print('--',i,e)

    def __len__(self):
        return self.len

if __name__ == '__main__':
    train_split, valid_split, test_split = list_split_raw_kitti()
    dset = RawKITTIDataset('/home/ubuntu/kitti/raw/',\
                           '/home/ubuntu/kitti/flow/raw/',\
                           train_split,2,transform=ToTensor())
    dload = DataLoader(dset,batch_size=512,shuffle=True,\
                     num_workers=1,pin_memory=False) #,collate_fn=my_collate_)
    for x,imu,y,_ in dload:
        print(x.size(),imu.size(),y.size())
