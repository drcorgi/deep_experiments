import numpy as np
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
    ''' Metadados da base de Odometria visual
        Tipo: lista de dicionários ('sub_base' 'sequence' 'sid_frame' 'frame_fn' 'odom_fn')
    '''
    meta_fn = sys.argv[1] # 'visual_odometry_database.meta'
    h5_fn = sys.argv[2] # '/home/ronnypetson/Documents/deep_odometry/kitti/frames_odom.h5'

    with open(meta_fn,'rb') as f:
        meta = pickle.load(f)

    chunk_size = 100
    n_segments = len(meta)//chunk_size
    included = chunk_size*n_segments
    height,width = 376,1241

    h5_file = h5py.File(h5_fn,'w')
    frames = h5_file.create_dataset('frames',shape=(n_segments,chunk_size,height,width),dtype=np.uint8)
    # odometry can be loaded all at once; no need for hdf5
    frame_chunk = []
    for i in range(100*102,included):
        img = cv2.imread(meta[i]['frame_fn'],0)
        img = cv2.resize(img,(width,height))
        if img is not None:
            frame_chunk.append(img)
        else:
            print('None frame')
        if (i+1)%chunk_size == 0:
            frame_chunk = np.array(frame_chunk,dtype=np.uint8)
            print('Writing chunk {} of shape {}'.format(i//chunk_size,frame_chunk.shape))
            frames[i//chunk_size] = frame_chunk
            frame_chunk = []
    h5_file.close()
