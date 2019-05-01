import os
import sys
import pickle
import numpy as np

from datetime import datetime
from odom_loader import load_kitti_odom

if __name__=='__main__':
    odom_fn = sys.argv[1] #'/home/ronnypetson/Documents/deep_odometry/kitti/abs_poses.pck'
    meta_fn = sys.argv[2] #'visual_odometry_database.meta'

    ''' Metadados da base de Odometria visual
        Tipo: lista de dicionários ('sub_base' 'sequence' 'sid_frame' 'frame_fn' 'odom_fn')
    '''
    with open(meta_fn,'rb') as f:
        meta = pickle.load(f)

    # Works for KITTI
    prev_s = -1
    all_odom = [[],[],[],[],[],[],[],[],[],[],[]]
    for i in range(len(meta)):
        s = int(meta[i]['sequence'])
        print(s)
        sid = meta[i]['sid_frame']
        if s != prev_s:
            odoms = load_kitti_odom(meta[i]['odom_fn'])
        odom = odoms[sid]
        all_odom[s].append(odom)
        prev_s = s

    with open(odom_fn,'wb') as f:
        pickle.dump(all_odom,f)
