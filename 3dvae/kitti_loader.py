import pykitti
import cv2
import numpy as np

basedir = '/home/ronnypetson/Downloads/'
date = '2011_09_30'
drive = '0028'
data = pykitti.raw(basedir, date, drive)

def save_flows():
    frames = [cv2.resize(np.array(f),(128,128)) for f in data.cam0]
    save_opt_flows(frames)

def get_test_poses():
    poses = [d[1] for d in data.oxts]
    poses = [[np.matmul(np.linalg.inv(poses[i]),poses[j]) for j in range(i,i+wsize,1)] for i in range(len(poses)-wsize+1)]
    return np.array([[flat_homogen(p) for p in seq] for seq in poses])

def get_test_imu():
    imu = [d[0][6:-7] for d in data.oxts]
    return np.array([imu[i:i+wsize] for i in range(len(imu)-wsize+1)])

def save_flows_for_ae():
    frames = log_run_kitti_all()
    save_opt_flows(frames)

