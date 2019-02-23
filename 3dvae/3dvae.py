import sys
import signal
import numpy as np
import gym
import cv2
import matplotlib.pyplot as plt

from vae import *
from transition import *
from utilities import *

wsize = 256
seq_len = 16

if __name__ == '__main__':
    # Loading the data
    frames = log_run_kitti_all() # log_run_video(fdir='/home/ronnypetson/Videos/pikachu')
    poses, poses_abs, avoid = load_kitti_odom_all(wsize=16)
    # Loading the encoder models
    aes = [Vanilla2DAutoencoder([None,h,w],1e-3,batch_size,128,'/home/ronnypetson/models/VanillaAE2D_128x128_kitti'),\
           Vanilla1DAutoencoder([None,seq_len,128],1e-3,batch_size,256,'/home/ronnypetson/models/VanillaAE1D_kitti_256_16',False),\
           Vanilla1DAutoencoder([None,seq_len,256],1e-3,batch_size,512,'/home/ronnypetson/models/VanillaAE1D_kitti_512_16',False)]
    #train_last_ae(aes[:1],frames,40,seq_len=seq_len)
    #train_last_ae(aes[:3],frames,40,seq_len=seq_len)
    #encode_decode_sequence(aes[:3],frames[:256],seq_len=seq_len)
    # Mapping from state to pose
    #t = Matcher([None,512],[None,1024,12],model_fname='/home/ronnypetson/models/Matcher_kitti_512_1024')
    #t = Matcher([None,256],[None,wsize,12],model_fname='/home/ronnypetson/models/Matcher_kitti_256_32')
    #t = Matcher([None,512],[None,wsize,12],model_fname='/home/ronnypetson/models/Matcher_kitti_512_256')
    #t = Matcher([None,256],[None,wsize,12],model_fname='/home/ronnypetson/models/Matcher_kitti_256_16')

    t = Conv1DTransition([None,seq_len,128],[None,seq_len,12],model_fname='/home/ronnypetson/models/Matcher_kitti_16x128_16x12')
    data_x = up_(aes[:1],frames,seq_len=seq_len,training=True)
    data_x = [[data_x[j] for j in range(i,i+seq_len,1)] for i in range(len(data_x)-seq_len+1)] #
    data_x = np.array([data_x[i] for i in range(len(data_x)) if i not in avoid])
    print(len(data_x),len(poses)) #
    tf.reset_default_graph()
    data_x_train = data_x[1024:]
    data_x_test = data_x[:1024]
    poses_train = poses[1024:]
    poses_test = poses[:1024]
    train_transition(t,data_x_train,poses_train,300)
    # Checking the estimated poses
    rmse, estimated = test_transition(t,data_x_test,poses_test)
    gt_points = get_3d_points_(poses_test,16)
    est_points = get_3d_points_(estimated,16)
    plot_2d_points_(gt_points,est_points)
    plot_3d_points_(gt_points,est_points)
    print(rmse)

