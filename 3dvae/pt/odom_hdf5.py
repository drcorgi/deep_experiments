import numpy as np
import gym
import cv2
import os, sys
import re
import time
import pickle
import h5py

from copy import deepcopy
from glob import glob

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
    '''with open(fname,'wb') as f:
        pickle.dump(flows,f)'''
    np.save(fname,flows)

def get_opt_flows(frames):
    seq_flows = []
    for i in range(len(frames)-1):
        flow = cv2.calcOpticalFlowFarneback(frames[i],frames[i+1],None,0.5,3,15,3,5,1.2,0)
        seq_flows.append(flow)
    return np.array(seq_flows)

def log_run_kitti(fdir,fshape):
    frames = []
    fnames = os.listdir(fdir)
    fnames = [f for f in fnames if os.path.isfile(fdir+'/'+f)]
    fnames = sorted(fnames,key=lambda x: int(x[:-4]))
    for fname in fnames:
        f = cv2.imread(fdir+'/'+fname,0)
        f = cv2.resize(f,fshape)
        frames.append(f)
    return np.array(frames)

def log_run_kitti_all(re_dir,fshape):
    seqs = [p for p in glob(re_dir) if os.path.isdir(p)]
    seqs = sorted(seqs)[:11]
    print(seqs)
    frames = []
    for s in seqs:
        print('Loading sequence from '+s)
        f = log_run_kitti(s,fshape)
        frames.append(f)
    return frames

def kitti_save_all(re_dir,fshape):
    seqs = [p for p in glob(re_dir) if os.path.isdir(p)]
    seqs = sorted(seqs)[:11]
    print(seqs)
    print('Saving .npy versions of frames and flows')
    for s in seqs:
        print('Loading sequence from '+s)
        f = log_run_kitti(s,fshape)
        np.save(s+'/../frames_{}_{}.npy'.format(fshape[0],fshape[1]),f)
        f = get_opt_flows(f)
        np.save(s+'/../flows_{}_{}.npy'.format(fshape[0],fshape[1]),f)

def load_kitti_odom(fdir):
    with open(fdir) as f:
        content = f.readlines()
    poses = [l.split() for l in content]
    poses = np.array([ [ float(p) for p in l ] for l in poses ])
    return poses

def load_kitti_odom_all(fdir):
    fns = os.listdir(fdir)
    fns = sorted(fns,key=lambda x: int(x[:-4]))
    print(fns)
    poses = []
    for fn in fns:
        print('Loading poses from '+fn)
        p = load_kitti_odom(fdir+'/'+fn)
        poses.append(p)
    return poses

if __name__ == '__main__':
    ''' Metadados da base
        Tipo: lista de dicionários (id_sub_base id_sequencia id_frame arquivo_frame arquivo_odometria linha_odometria)
    '''
    meta_fn = sys.argv[1]

    kitti_save_all(frames_dir,frame_shape)
    odoms = load_kitti_odom_all(odom_dir)
    print('Odom',len(odoms))
    np.save(odom_dir+'/abs_poses_{}_{}'.format(frame_shape[0],frame_shape[1]),odoms)
