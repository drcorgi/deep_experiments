import numpy as np
import cv2
import os, sys
import re
import time
import pickle
import h5py
from glob import glob
from odom_loader import load_kitti_odom

if __name__ == '__main__':
    ''' Metadados da base de Odometria visual
        Tipo: lista de dicionários ('sub_base' 'sequence' 'sid_frame' 'frame_fn' 'odom_fn')
    '''
    meta_fn = sys.argv[1] # 'visual_odometry_database.meta'
    h5_fn = sys.argv[2] # '/home/ronnypetson/Documents/deep_odometry/kitti/frames_odom'

    with open(meta_fn,'rb') as f:
        meta = pickle.load(f)

    valid_count = 5640
    test_count = 1100

    chunk_size = 10
    n_segments = len(meta)//chunk_size if len(meta)%chunk_size == 0 else len(meta)//chunk_size + 1
    valid_segments = valid_count//chunk_size
    test_segments = test_count//chunk_size
    train_segments = n_segments - valid_segments - test_segments
    included = len(meta) #chunk_size*n_segments
    height,width = 376,1241

    h5_valid = h5py.File(h5_fn+'_valid.h5','w') # First 5640 frames
    h5_test = h5py.File(h5_fn+'_test.h5','w') # Next 1100 frames
    h5_train = h5py.File(h5_fn+'_train.h5','w') # Remaining frames

    grp_valid_frames = h5_valid.create_group('frames')
    grp_valid_poses = h5_valid.create_group('poses')
    grp_test_frames = h5_test.create_group('frames')
    grp_test_poses = h5_test.create_group('poses')
    grp_train_frames = h5_train.create_group('frames')
    grp_train_poses = h5_train.create_group('poses')

    frames_valid = grp_valid_frames.create_dataset('frames',shape=(valid_segments,chunk_size,height,width),dtype=np.uint8)
    frames_test = grp_test_frames.create_dataset('frames',shape=(test_segments,chunk_size,height,width),dtype=np.uint8)
    frames_train = grp_train_frames.create_dataset('frames',shape=(train_segments,chunk_size,height,width),dtype=np.uint8)

    poses_valid = grp_valid_poses.create_dataset('poses',shape=(valid_segments,chunk_size,12),dtype=np.float32)
    poses_test = grp_test_poses.create_dataset('poses',shape=(test_segments,chunk_size,12),dtype=np.float32)
    poses_train = grp_train_poses.create_dataset('poses',shape=(train_segments,chunk_size,12),dtype=np.float32)

    # odometry can be loaded all at once; no need for hdf5
    frame_chunk = []
    poses_chunk = []
    for i in range(included):
        img = cv2.imread(meta[i]['frame_fn'],0)
        img = cv2.resize(img,(width,height))
        odom = load_kitti_odom(meta[i]['odom_fn'])[meta[i]['sid_frame']]
        if img is not None and odom is not None:
            frame_chunk.append(img)
            poses_chunk.append(odom)
        else:
            print('None frame')
        if (i+1)%chunk_size == 0:
            frame_chunk = np.array(frame_chunk,dtype=np.uint8)
            poses_chunk = np.array(poses_chunk,dtype=np.float32)
            print('Writing chunk {} of shape {}, poses {}'.format(i//chunk_size,frame_chunk.shape,poses_chunk.shape))
            if i < valid_count:
                frames_valid[i//chunk_size] = frame_chunk
                poses_valid[i//chunk_size] = poses_chunk
            elif i < valid_count + test_count:
                frames_test[(i-valid_count)//chunk_size] = frame_chunk
                poses_test[(i-valid_count)//chunk_size] = poses_chunk
            else:
                frames_train[(i-valid_count-test_count)//chunk_size] = frame_chunk
                poses_train[(i-valid_count-test_count)//chunk_size] = poses_chunk
            frame_chunk = []
            poses_chunk = []
    h5_valid.close()
    h5_test.close()
    h5_train.close()
