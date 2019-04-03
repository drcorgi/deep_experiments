import numpy as np
import gym
import cv2
import os
import re
import time
from copy import deepcopy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool

img_shape = (128,128,1)
batch_size = 64
latent_dim = 128
h, w, _ = img_shape

def plot_abs(gt,rec,ddir='/home/ronnypetson/models'):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d') # ,projection='3d'
    gt = np.array([[p[3],p[7],p[11]] for p in gt])
    ax.plot(gt[:,0],gt[:,1],gt[:,2],'g')
    ax.plot(rec[:,0],rec[:,1],rec[:,2],'b')
    plt.savefig(ddir+'/3d_abs_plot.png')
    plt.close(fig)

def plot_3d_points_(gt,est,ddir='/home/ronnypetson/models'):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot(gt[:,0],gt[:,1],gt[:,2],'g')
    ax.plot(est[:,0],est[:,1],est[:,2],'b')
    plt.savefig(ddir+'/3d_path_plot.png')
    plt.close(fig)

def plot_2d_points_(gt,est,ign=1,ddir='/home/ronnypetson/models'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if ign == 1:
        ax.plot(gt[:,0],gt[:,1],'g')
        ax.plot(est[:,0],est[:,1],'b')
    plt.savefig(ddir+'/2d_path_plot.png')
    plt.close(fig)

def homogen(x):
    assert len(x) == 12
    return np.array(x.tolist()+[0.0,0.0,0.0,1.0]).reshape((4,4))

def flat_homogen(x):
    assert x.shape == (4,4)
    return np.array(x.reshape(16)[:-4])

def abs2relative(abs_poses,wsize,stride):
    poses = [homogen(p) for p in abs_poses]
    rposes = []
    for i in range(len(poses)-(stride*wsize-1)):
        rposes.append([flat_homogen(np.matmul(np.linalg.inv(poses[i]),poses[j])) for j in range(i,i+stride*wsize,stride)])
    return np.array(rposes)

def get_3d_points_fast(rposes,wlen=32):
    rposes = [[homogen(p) for p in r] for r in rposes]
    aposes = [rposes[0][0]]
    for i in range(1,len(rposes),1):
        pos_seq_idx = max(0,i-wlen//2)
        pos_subseq_idx = wlen//2
        if i < pos_subseq_idx:
            pos_subseq_idx = i
        abs_pose = np.matmul(aposes[pos_seq_idx],rposes[pos_seq_idx][pos_subseq_idx])
        aposes.append(abs_pose)
    return np.array([[p[0,3],p[1,3],p[2,3]] for p in aposes])

def get_3d_points_(rposes,wlen=32):
    rposes = [[homogen(p) for p in r] for r in rposes]
    aposes = [rposes[0]]
    for i in range(1,len(rposes),1):
        p = []
        for j in range(max(0,i-(wlen-1)),i,1):
            p.append(aposes[j][i-j])
        in_p = np.mean(p,axis=0)
        new_p = [np.matmul(in_p,rposes[i][j]) for j in range(wlen)]
        aposes.append(new_p)
    poses_ = []
    for i in range(len(aposes)+wlen-1):
        p = []
        for j in range(max(0,i-(wlen-1)),min(len(aposes),i+1),1):
            p.append(aposes[j][i-j])
        poses_.append(np.mean(p,axis=0))
    return np.array([[p[0,3],p[1,3],p[2,3]] for p in poses_])

def __get_3d_points(rposes,wlen):
    rposes = [[homogen(p) for p in r] for r in rposes]
    aposes = rposes[0]
    for i in range(wlen,len(rposes),wlen):
        in_p = aposes[-1]
        aposes += [np.matmul(in_p,rposes[i][j]) for j in range(wlen)]
    return np.array([[p[0,3],p[1,3],p[2,3]] for p in aposes])
