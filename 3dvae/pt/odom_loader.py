import numpy as np
import gym
import cv2
import os
import re
import time
from copy import deepcopy

img_shape = (128,128,1)
h, w, _ = img_shape

def homogen(x):
    assert len(x) == 12
    return np.array(x.tolist()+[0.0,0.0,0.0,1.0]).reshape((4,4))

def flat_homogen(x):
    assert x.shape == (4,4)
    return np.array(x.reshape(16)[:-4])

def save_opt_flows(frames):
    flows = []
    for i in range(len(frames)-1):
        flow = cv2.calcOpticalFlowFarneback(frames[i],frames[i+1],None,0.5,3,15,3,5,1.2,0)
        flows.append(flow)
    np.save('/home/ronnypetson/Documents/deep_odometry/kitti/dataset_frames/sequences/flows_00-10_128x128.npy'\
            ,np.array(flows))

def log_run_kitti(fdir='/home/ronnypetson/Documents/deep_odometry/kitti/dataset_frames/sequences/02/image_0'):
    frames = []
    fnames = os.listdir(fdir)
    fnames = [f for f in fnames if os.path.isfile(fdir+'/'+f)]
    fnames = sorted(fnames,key=lambda x: int(x[:-4]))
    imgs = [cv2.imread(fdir+'/'+fname,0) for fname in fnames]
    #with Pool(5) as p:
    #    imgs = p.map(imread_0,[fdir+'/'+fname for fname in fnames])
    for f in imgs:
        #f = cv2.resize(f,img_shape[:-1],interpolation=cv2.INTER_LINEAR)#.reshape(img_shape[:-1])
        frames.append(f) # /255.0
    return np.array(frames)

def log_run_kitti_all(re_dir='/home/ronnypetson/Documents/deep_odometry/kitti/dataset_frames/sequences/{}/image_0/128x128'):
    seqs = ['00','01','02','03','04','05','06','07','08','09','10'] #,'11',\
             #'12','13','14','15','16','17','18','19','20','21']
    frames = log_run_kitti(re_dir.format(seqs[0]))
    for s in seqs[1:]:
        print('Loading sequence from '+s)
        sframes = log_run_kitti(re_dir.format(s))
        frames = np.concatenate((frames,sframes),axis=0)
    return frames

def load_kitti_odom(fdir='/home/ronnypetson/Documents/deep_odometry/kitti/dataset/poses/02.txt',wsize=32,stride=1):
    '''
        Returns relative poses (window-wise) and absolute poses (frame-wise) in flat homogen form
    '''
    with open(fdir) as f:
        content = f.readlines()
    poses = [l.split() for l in content]
    poses = np.array([ [ float(p) for p in l ] for l in poses ])
    poses_ = [homogen(p) for p in poses]
    rposes = []
    for i in range(len(poses_)-(stride*wsize-1)):
        #rposes.append([flat_homogen(np.matmul(poses_[j],np.linalg.inv(poses_[i]))) for j in range(i,i+wsize,1)])
        rposes.append([flat_homogen(np.matmul(np.linalg.inv(poses_[i]),poses_[j])) for j in range(i,i+stride*wsize,stride)])
    return np.array(rposes), poses

def load_kitti_odom_all(fdir='/home/ronnypetson/Documents/deep_odometry/kitti/dataset/poses',wsize=32,stride=1):
    fns = os.listdir(fdir)
    fns = sorted(fns,key=lambda x: int(x[:-4]))
    rposes, aposes = load_kitti_odom(fdir+'/'+fns[0],wsize,stride)
    limits = [len(aposes)]
    for fn in fns[1:]:
        rp, ap = load_kitti_odom(fdir+'/'+fn,wsize,stride)
        rposes = np.concatenate((rposes,rp),axis=0)
        aposes = np.concatenate((aposes,ap),axis=0)
        limits.append(len(aposes))
    return rposes, aposes, np.reshape([range(l-(stride*wsize-1),l,1) for l in limits],(-1,))

if __name__ == '__main__':
    frames = log_run_kitti_all()
    _, odoms, _ = load_kitti_odom_all()
    #print(frames.shape, odoms.shape)
    save_opt_flows(frames)
    np.save('/home/ronnypetson/Documents/deep_odometry/kitti/dataset/poses/00_10_poses.npy',np.array(odoms[1:]))
