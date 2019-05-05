import numpy as np
import cv2
import os, sys
import re
import time
import pickle
import h5py

from glob import glob

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

    frames_valid = h5_valid.create_dataset('frames',shape=(valid_segments,chunk_size,height,width),dtype=np.uint8)
    frames_test = h5_test.create_dataset('frames',shape=(test_segments,chunk_size,height,width),dtype=np.uint8)
    frames_train = h5_train.create_dataset('frames',shape=(train_segments,chunk_size,height,width),dtype=np.uint8)

    # odometry can be loaded all at once; no need for hdf5
    frame_chunk = []
    for i in range(included):
        img = cv2.imread(meta[i]['frame_fn'],0)
        img = cv2.resize(img,(width,height))
        if img is not None:
            frame_chunk.append(img)
        else:
            print('None frame')
        if (i+1)%chunk_size == 0:
            frame_chunk = np.array(frame_chunk,dtype=np.uint8)
            print('Writing chunk {} of shape {}'.format(i//chunk_size,frame_chunk.shape))
            if i < valid_count:
                frames_valid[i//chunk_size] = frame_chunk
            elif i < valid_count + test_count:
                frames_test[(i-valid_count)//chunk_size] = frame_chunk
            else:
                frames_train[(i-valid_count-test_count)//chunk_size] = frame_chunk
            frame_chunk = []
    h5_valid.close()
    h5_test.close()
    h5_train.close()
