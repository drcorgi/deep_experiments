import numpy as np
import cv2
import os, sys
import re
import time
import pickle
import pykitti

#sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/misc')

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
    #print(fnames)
    #imgs = [cv2.imread(fdir+'/'+fname,0) for fname in fnames]
    #with Pool(5) as p:
    #    imgs = p.map(imread_0,[fdir+'/'+fname for fname in fnames])
    for fname in fnames:
        f = cv2.imread(fdir+'/'+fname,0)
        f = cv2.resize(f,fshape) #.reshape(img_shape[:-1])
        frames.append(f) # /255.0
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

def load_raw_kitti_odom_imu(basedir,dates_drives):
    ''' OxtsData(packet=OxtsPacket(lat=49.030737883859,\
        lon=8.3398878482103, alt=114.77355194092, roll=0.035521,\
        pitch=0.006243, yaw=-0.9161816732051, vn=-6.3349919699448,\
        ve=4.8542886072513, vf=7.980990183777, vl=0.006629911301955,\
        vu=0.043791248142715, ax=-0.0034208888097007, ay=0.61793453350163,\
        az=9.325595250268, af=0.050762056308803, al=0.2773333895807,\
        au=9.3426054822303, wx=-0.017105540779655, wy=0.0057239717836089,\
        wz=-0.00081830538408185, wf=-0.017108767059082, wl=0.0057499843741701,\
        wu=-0.00048516742865752, pos_accuracy=0.11469088891451, vel_accuracy=0.019849433241279,\
        navstat=4, numsats=7, posmode=5, velmode=5, orimode=6), T_w_imu
    '''
    '''debug_msg = 'load_raw_kitti_odom_imu'
    dates = [d for d in os.listdir(basedir) if os.path.isdir(d)]
    print(debug_msg,'dates:',dates)
    dates_drives = []
    for d in dates:
        dates_drives += [(d,drv[11:-5]) for drv in\
                         os.listdir(basedir+'/'+d+'/') if os.path.isdir(drv)]
    print(debug_msg,'dates_drives:',dates_drives)'''
    odom, imu = [], []
    for dd in dates_drives:
        data = pykitti.raw(basedir,dd[0],dd[1])
        odom.append([flat_homogen(o.T_w_imu) for o in data.oxts])
        imu.append([np.array(o.packet[6:23]) for o in data.oxts])
    return odom, imu

def load_raw_kitti_img_odom_imu(basedir,dates_drives,h=32,w=128):
    img, odom, imu = [], [], []
    for dd in dates_drives:
        data = pykitti.raw(basedir,dd[0],dd[1])
        #print(np.array(next(data.cam0)).shape)
        img.append([cv2.resize(np.array(im),(128,32)) for im in data.cam0])
        odom.append([flat_homogen(o.T_w_imu) for o in data.oxts])
        imu.append([np.array(o.packet[6:23]) for o in data.oxts])
    return img, odom, imu

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
    if len(sys.argv) != 5:
        print('Input:',sys.argv)
        print('Usage: frames_dir h w odom_dir')
        exit()

    frames_dir = sys.argv[1] #'/home/ronnypetson/Documents/deep_odometry/kitti/dataset_frames/sequences/*/image_0/'
    #frames_out = sys.argv[2] #'/home/ronnypetson/Documents/deep_odometry/kitti/dataset_frames/sequences/frames_00-10_128x128.npy'
    #flows_out = sys.argv[3] #'/home/ronnypetson/Documents/deep_odometry/kitti/dataset_frames/sequences/flows_00-10_128x128.npy'
    frame_shape = (int(sys.argv[2]),int(sys.argv[3])) # 128 128
    odom_dir = sys.argv[4] #'/home/ronnypetson/Documents/deep_odometry/kitti/dataset/poses'
    #odom_out = sys.argv[7] #'/home/ronnypetson/Documents/deep_odometry/kitti/dataset/poses/poses_00-10.npy'

    #frames = log_run_kitti_all(frames_dir,frame_shape)
    #print('Saving numpy binary version of frames')
    #np.save(frames_out,frames)
    kitti_save_all(frames_dir,frame_shape)
    odoms = load_kitti_odom_all(odom_dir)
    print('Odom',len(odoms))
    np.save(odom_dir+'/abs_poses_{}_{}'.format(frame_shape[0],frame_shape[1]),odoms)
    #print('Frames {} odoms {}'.format(len(frames),len(odoms)))
    #print('Saving .npy versions of flows and odoms')
    #save_opt_flows(frames,flows_out)
    #with open(odom_out,'wb') as f:
    #    pickle.dump(odoms,f)
