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

def c3dto2d(p):
    p[[1,4,6,7,9]] = np.zeros(5,dtype=np.float32)
    p[5] = 1.0
    det = np.linalg.det([p[[0,2]],p[[8,10]]])
    if det < 5e-1:
        print(p)
    p[[0,2,8,10]] = p[[0,2,8,10]]/(det+1e-7)
    return p

def plot_abs(gt,rec,out_fn):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    gt = np.array([[p[3],p[7],p[11]] for p in gt])
    #rec = np.array([[p[3],p[7],p[11]] for p in rec])
    ax.plot(gt[:,0],gt[:,1],gt[:,2],'g.')
    ax.plot(rec[:,0],rec[:,1],rec[:,2],'b.')
    plt.savefig(out_fn)
    plt.close(fig)

def plot_3d_points_(gt,est,out_fn,wlen):
    fig = plt.figure()
    ax = fig.add_subplot(131)
    ax.plot(gt[:,0],gt[:,1],'g')
    ax.plot(est[:,0],est[:,1],'b')
    ax = fig.add_subplot(132)
    ax.plot(gt[:,0],gt[:,2],'g')
    #ax.plot(est[:,0],est[:,2],'b')

    ax.plot(est[::wlen,0],est[::wlen,2],'r.')

    ax = fig.add_subplot(133)
    ax.plot(gt[:,1],gt[:,2],'g')
    ax.plot(est[:,1],est[:,2],'b')
    '''ax = fig.add_subplot(111)
    ax.plot(gt[:,0],gt[:,1],'g.')
    ax.plot(est[:,0],est[:,1],'b.')'''
    plt.savefig(out_fn)
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

def abs2relative_(abs_poses,wsize,stride):
    rposes = []
    for i in range(len(abs_poses)-(stride*wsize-1)):
        rposes.append([-abs_poses[i]+abs_poses[j] for j in range(i,i+stride*wsize,stride)])
    return np.array(rposes)

def abs2relative(abs_poses,wsize,stride):
    poses = [homogen(p) for p in abs_poses]
    rposes = []
    for i in range(len(poses)-(stride*wsize-1)):
        rposes.append([flat_homogen(np.matmul(np.linalg.inv(poses[i]),poses[j])) for j in range(i,i+stride*wsize,stride)])
        #rposes.append([flat_homogen(np.matmul(np.linalg.inv(poses[min(i,j-1)]),poses[j])) for j in range(i,i+stride*wsize,stride)])
    return np.array(rposes)

def relative2abs(rel_poses,wsize):
    ''' rel_poses: array de poses relativas (contíguo)
    '''
    poses = [homogen(p) for p in rel_poses]
    abs_poses = poses[:wsize]
    for i in range(wsize,len(poses)):
        in_p = abs_poses[wsize*(i//wsize)-1]
        #print(in_p)
        try:
            in_p = np.linalg.inv(in_p)
            abs_poses.append(np.matmul(in_p,poses[i]))
        except np.linalg.LinAlgError:
            abs_poses.append(abs_poses[-1])
    abs_poses = [flat_homogen(p) for p in abs_poses]
    return abs_poses

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

def get_3d_points_(rposes,wlen):
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

def get_3d_points__(rposes,wlen):
    rposes = [[homogen(p) for p in r] for r in rposes]
    aposes = rposes[0]
    for i in range(wlen,len(rposes),wlen):
        #in_p = np.matmul(aposes[-1],rposes[i-1][1])
        in_p = aposes[-1]
        aposes += [np.matmul(in_p,rposes[i][j]) for j in range(wlen)]
        #for j in range(wlen):
        #    aposes.append(np.matmul(aposes[-1],rposes[i][j]))
    return np.array([[p[0,3],p[1,3],p[2,3]] for p in aposes])

def get_3d_points_t(rposes,wlen,gt_poses):
    rposes = [[homogen(p) for p in r] for r in rposes]
    aposes = rposes[0]
    for i in range(wlen,len(rposes),wlen):
        #in_p = aposes[-1]
        #in_p = np.matmul(aposes[-wlen+1],rposes[i-wlen+1][-1])
        #in_p = np.matmul(aposes[-1],rposes[i-1][1])
        in_p = homogen(gt_poses[i])
        aposes += [np.matmul(in_p,rposes[i][j]) for j in range(wlen)]
        #in_p = aposes[-wlen+1]+rposes[i-wlen+1][-1]
        #aposes += [in_p+rposes[i][j] for j in range(wlen)]
    return np.array([[p[0,3],p[1,3],p[2,3]] for p in aposes])

def get_3d_points_t2(rposes,wlen,gt_poses):
    rposes = [homogen(p) for p in rposes]
    aposes = rposes[0]
    for i in range(wlen,len(rposes),wlen):
        in_p = homogen(gt_poses[i])
        aposes += [np.matmul(in_p,rposes[i][j]) for j in range(wlen)]
    return np.array([[p[0,3],p[1,3],p[2,3]] for p in aposes])

def plot_eval(model,test_loader,seq_len,device='cuda:0'):
    rel_poses = []
    data_y = []
    for xy in test_loader:
        x,y = xy[0].to(device), xy[1].to(device)
        y_ = model(x)
        data_y += y.cpu().detach().numpy().reshape(-1,12).tolist()
        rel_poses += y_.cpu().detach().numpy().reshape(-1,12).tolist()

    rel_poses = np.array(rel_poses) #.transpose(0,2,1)
    gt = np.array(data_y) #.transpose(0,2,1)
    print(rel_poses.shape)
    abs_ = np.array(relative2abs(rel_poses,seq_len))
    print(abs_.shape)

    pts = [[p[3],p[7],p[11]] for p in abs_] #get_3d_points_t2(rel_poses,seq_len,abs_)
    pts_ = get_3d_points__(rel_poses,seq_len)
    gt = get_3d_points__(gt,seq_len)

    print(pts.shape,pts_.shape,gt.shape)
    if not os.path.isdir('tmp'):
        os.mkdir('tmp')
    t = time.time()
    plot_3d_points_(pts,pts_,'tmp/{}_projections_xyz.png'.format(t),wlen=seq_len) #gt
    plot_abs(abs_,pts_,'tmp/{}_absolute_gt_3d.png'.format(t))
