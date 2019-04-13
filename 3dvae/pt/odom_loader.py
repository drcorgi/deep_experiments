import numpy as np
import gym
import cv2
import os
import re
import time
import pickle
from copy import deepcopy

img_shape = (128,128,1)
h, w, _ = img_shape

def homogen(x):
    assert len(x) == 12
    return np.array(x.tolist()+[0.0,0.0,0.0,1.0]).reshape((4,4))

def flat_homogen(x):
    assert x.shape == (4,4)
    return np.array(x.reshape(16)[:-4])

def save_opt_flows(frames,fname):
    flows = []
    for s in frames:
        seq_flows = []
        for i in range(len(s)-1):
            flow = cv2.calcOpticalFlowFarneback(s[i],s[i+1],None,0.5,3,15,3,5,1.2,0)
            seq_flows.append(flow)
        flows.append(np.array(seq_flows))
    with open(fname,'wb') as f:
        pickle.dump(flows,f)

def log_run_kitti(fdir='/home/ronnypetson/Documents/deep_odometry/kitti/dataset_frames/sequences/02/image_0'):
    frames = []
    fnames = os.listdir(fdir)
    fnames = [f for f in fnames if os.path.isfile(fdir+'/'+f)]
    fnames = sorted(fnames,key=lambda x: int(x[:-4]))
    #print(fnames)
    imgs = [cv2.imread(fdir+'/'+fname,0) for fname in fnames]
    #with Pool(5) as p:
    #    imgs = p.map(imread_0,[fdir+'/'+fname for fname in fnames])
    for f in imgs:
        f = cv2.resize(f,(128,128)) #.reshape(img_shape[:-1])
        frames.append(f) # /255.0
    return np.array(frames)

def log_run_kitti_all(re_dir='/home/ronnypetson/Documents/deep_odometry/kitti/dataset_frames/sequences/{}/image_0/128x128'):
    seqs = ['00','01','02','03','04','05','06','07','08','09','10'] #,'11',\
             #'12','13','14','15','16','17','18','19','20','21']
    frames = []
    for s in seqs:
        print('Loading sequence from '+s)
        f = log_run_kitti(re_dir.format(s))
        frames.append(f)
    return frames

def load_kitti_odom(fdir='/home/ronnypetson/Documents/deep_odometry/kitti/dataset/poses/02.txt'):
    with open(fdir) as f:
        content = f.readlines()
    poses = [l.split() for l in content]
    poses = np.array([ [ float(p) for p in l ] for l in poses ])
    return poses

def load_kitti_odom_all(fdir='/home/ronnypetson/Documents/deep_odometry/kitti/dataset/poses'):
    fns = os.listdir(fdir)
    fns = sorted(fns,key=lambda x: int(x[:-4]))
    poses = []
    for fn in fns:
        print('Loading poses from '+fn)
        p = load_kitti_odom(fdir+'/'+fn)
        poses.append(p)
    return poses

if __name__ == '__main__':
    frames = log_run_kitti_all()
    #frames = frames.reshape(-1,1,128,128)
    #np.save('/home/ronnypetson/Documents/deep_odometry/kitti/dataset_frames/sequences/frames_00-10_128x128.npy',frames)
    odoms = load_kitti_odom_all()
    #print(frames.shape, odoms.shape)
    save_opt_flows(frames,'/home/ronnypetson/Documents/deep_odometry/kitti/dataset_frames/sequences/flows_00-10_128x128.pck')
    #np.save('/home/ronnypetson/Documents/deep_odometry/kitti/dataset_frames/sequences/poses_00-10.npy',np.array(odoms))
    with open('/home/ronnypetson/Documents/deep_odometry/kitti/dataset_frames/sequences/poses_00-10.pck','wb') as f:
        pickle.dump(odoms,f)
