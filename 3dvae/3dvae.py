import sys
import signal
import numpy as np
import gym
import cv2
import matplotlib.pyplot as plt

from vae import *
from transition import *
from utilities import *

'''
img_shape = [64,64,1]
batch_size = 64
latent_dim = 128
h, w, _ = img_shape
'''

wsize = 32
seq_len = 32

if __name__ == '__main__':
    # Loading the data
    # Penn COSYVIO
    #frames, tstamps = log_run_video(num_it=10000) # ,fdir='/home/ronnypetson/Videos/Webcam'
    #tstamps, poses = load_penn_odom(tstamps) # Supposing all frame timestamps have a correspondent pose
    # KITTI

    '''frames, sdivs = log_run_kitti_all()
    poses, poses_abs = load_kitti_odom_all()
    print(frames.shape, len(sdivs))
    print(poses.shape, poses_abs.shape)
    exit()'''

    frames = log_run_kitti_all()
    poses, poses_abs, avoid = load_kitti_odom_all(seq_len=wsize)
    # Loading the encoder models
    aes = [Vanilla2DAutoencoder([None,h,w],1e-3,batch_size,128,'/home/ronnypetson/models/VanillaAE2D_128x128_kitti'),\
           Vanilla1DAutoencoder([None,seq_len,128],1e-3,batch_size,256,'/home/ronnypetson/models/VanillaAE1D_kitti_256',False),\
           Vanilla1DAutoencoder([None,seq_len,256],1e-3,batch_size,512,'/home/ronnypetson/models/VanillaAE1D_kitti_512',False)]
    train_last_ae(aes[:1],frames,30)
    encode_decode_sequence(aes[:1],frames[:32])
    # Mapping from state to pose
    #t = Matcher([None,512],[None,1024,12],model_fname='/home/ronnypetson/models/Matcher_kitti_512_1024')
    '''t = Matcher([None,256],[None,wsize,12],model_fname='/home/ronnypetson/models/Matcher_kitti_256_32')
    data_x = up_(aes[:2],frames,training=True)
    data_x = np.array([data_x[i] for i in range(len(data_x)) if i not in avoid])
    tf.reset_default_graph()
    print(len(data_x),len(poses))
    data_x_train = data_x[1024:]
    data_x_test = data_x[:1024]
    poses_train = poses[1024:]
    poses_test = poses[:1024]
    train_transition(t,data_x_train,poses_train,300)
    # Checking the estimated poses
    rmse, estimated = test_transition(t,data_x_test,poses_test)
    #gt_points = get_3d_points(poses[:2048], poses_abs[:2048]) # get_3d_points_(poses[:2048],seq_len=1024)
    est_points = get_3d_points_(estimated,wsize)#get_3d_points_(estimated,seq_len=1024)
    #plot_3d_points_(poses_abs,poses_abs)
    plot_abs(poses_abs[:1024],estimated)
    plot_2d_points_(est_points,est_points)
    print(rmse)'''

