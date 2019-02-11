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
    poses, poses_abs, avoid = load_kitti_odom_all()
    # Loading the encoder models
    aes = [VanillaAutoencoder([None,h,w,1],1e-3,batch_size,128,'/home/ronnypetson/models/Vanilla_AE_64x64_kitti'),\
           MetaVanillaAutoencoder([None,32,128,1],1e-3,batch_size,512,'/home/ronnypetson/models/Vanilla_Meta1_AE_kitti_512'),\
           MetaVanillaAutoencoder([None,32,256,1],1e-3,batch_size,256,'/home/ronnypetson/models/Vanilla_Meta2_AE_kitti',False)]
    #train_last_ae(aes[:2],frames,30)
    #encode_decode_sequence(aes[:2],frames[1000:1128])
    # Mapping from state to pose
    t = Matcher([None,512],[None,32,12],model_fname='/home/ronnypetson/models/Matcher_kitti_512_32')
    data_x = up_(aes[:2],frames,training=True)
    data_x = np.array([data_x[i] for i in range(len(data_x)) if i not in avoid])
    tf.reset_default_graph()
    print(len(data_x),len(poses))
    #train_transition(t,data_x,poses,500)
    # Checking the estimated poses
    rmse, estimated = test_transition(t,data_x[:2048],poses[:2048])
    gt_points = get_3d_points_(poses[:2048]) #get_3d_points(poses[:2048], poses_abs[:2048])
    est_points = get_3d_points_(estimated) #get_3d_points(estimated, poses_abs[:2048])
    plot_3d_points_(gt_points,est_points)
    plot_2d_points_(gt_points,est_points)
    print(rmse)

