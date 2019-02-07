import sys
import signal
import numpy as np
import gym
import cv2
import matplotlib.pyplot as plt

from vae import *
from transition import *
from utilities import *

img_shape = [64,64,1]
batch_size = 64
latent_dim = 128
h, w, _ = img_shape

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
    poses, poses_abs = load_kitti_odom_all()
    # Loading the encoder models
    aes = [VanillaAutoencoder([None,h,w,1],1e-3,batch_size,128,'/home/ronnypetson/models/Vanilla_AE_64x64_kitti'),\
           MetaVanillaAutoencoder([None,32,128,1],1e-3,batch_size,256,'/home/ronnypetson/models/Vanilla_Meta1_AE_kitti'),\
           MetaVanillaAutoencoder([None,32,256,1],1e-3,batch_size,256,'/home/ronnypetson/models/Vanilla_Meta2_AE_kitti',False)]
    #train_last_ae(aes[:2],frames,30)
    #encode_decode_sequence(aes[:2],frames[1000:1128])
    # Mapping from state to pose
    t = Transition([None,256],[None,12],model_fname='/home/ronnypetson/models/Vanilla_transition_kitti_')
    data_x = up_(aes[:2],frames,training=True)
    tf.reset_default_graph()
    print(len(data_x),len(poses[31:]))
    #train_transition(t,data_x,poses[31:],200)
    # Checking the estimated poses
    rmse, estimated = test_transition(t,data_x[256:256+128],poses[31+256:31+256+128])
    gt_points = get_3d_points(poses[31+256:31+256+128], poses_abs[256:256+128])
    est_points = get_3d_points(estimated, poses_abs[256:256+128])
    plot_3d_points_(gt_points,est_points)
    plot_2d_points_(gt_points,est_points)
    print(rmse)

