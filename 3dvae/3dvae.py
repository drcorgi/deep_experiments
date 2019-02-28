import sys
import signal
import numpy as np
import gym
import cv2
import matplotlib.pyplot as plt

from vae import *
from transition import *
from utilities import *

wsize = 16
seq_len = 16

if __name__ == '__main__':
    # Loading the data
    frames = log_run_kitti_all() # log_run_video(fdir='/home/ronnypetson/Videos/pikachu')
    poses, poses_abs, avoid = load_kitti_odom_all(wsize=wsize)
    # Loading the encoder models
    aes = [Vanilla2DAutoencoder([None,h,w],1e-3,batch_size,128,'/home/ronnypetson/models/VanillaAE2D_128x128_kitti'),\
           Vanilla1DAutoencoder([None,seq_len,128],1e-3,batch_size,256,'/home/ronnypetson/models/VanillaAE1D_kitti_256_16'),\
           Vanilla1DAutoencoder([None,seq_len,256],1e-3,batch_size,512,'/home/ronnypetson/models/VanillaAE1D_kitti_512_16',False)]
    #train_last_ae(aes[:2],frames,40,seq_len=seq_len)
    #train_last_ae(aes[:3],frames,40,seq_len=seq_len)
    #encode_decode_sequence(aes[:3],frames[:256],seq_len=seq_len)
    # Mapping from state to poses
    #t = Matcher([None,512],[None,1024,12],model_fname='/home/ronnypetson/models/Matcher_kitti_512_1024')
    #t = Conv1DTransition([None,seq_len,128+256],[None,wsize,12],
    #            model_fname='/home/ronnypetson/models/Matcher_kitti_{}x(128+256)_{}x12'.format(seq_len,wsize))
    t = ContextConv1DTransition([None,seq_len,128],[None,wsize,12],[None,256],
                model_fname='/home/ronnypetson/models/ContextTransition_kitti_{}x128_{}x256_{}x12'.format(seq_len,1,wsize))
    #t = Conv1DTransition([None,seq_len,256],[None,wsize,12],
    #            model_fname='/home/ronnypetson/models/Matcher_kitti_{}x256_{}x12'.format(seq_len,wsize))
    data_x = up_(aes[:1],frames,seq_len=seq_len,training=True)
    context = up_(aes[:2],frames,seq_len=seq_len,training=True) # True
    #data_x = np.concatenate((data_x,emb_frames[seq_len//2:-seq_len//2+1]),axis=1)
    # TODO: Augment data_x: forwards and backwards
    data_x = [data_x[i:i+seq_len] for i in range(len(data_x)-seq_len+1)] # for level-1
    #data_x = [data_x[i:i+seq_len*seq_len:seq_len] for i in range(len(data_x)-seq_len*seq_len+1)]
    #data_x = [data_x[i:i+seq_len*seq_len:seq_len] for i in range(len(data_x)-seq_len*seq_len+1)]
    #poses = poses[:-seq_len+1]
    # Align sequences with poses
    data_x = np.array([data_x[i] for i in range(len(data_x)) if i not in avoid])
    context = np.array([context[i] for i in range(len(context)) if i not in avoid])
    print(len(data_x),len(context))
    assert len(data_x) == len(context)
    print(len(data_x),len(poses)) #
    assert len(data_x) == len(poses)
    #
    tf.reset_default_graph()
    data_x_train = data_x[1024:]
    data_x_test = data_x[:1024]
    poses_train = poses[1024:]
    poses_test = poses[:1024]
    cont_train = context[1024:]
    cont_test = context[:1024]
    train_transition_(t,data_x_train,cont_train,poses_train,300)
    # Checking the estimated poses
    rmse, estimated = test_transition_(t,data_x_test,cont_test,poses_test)
    gt_points = get_3d_points_(poses_test,wsize)
    est_points = get_3d_points_(estimated,wsize)
    plot_2d_points_(gt_points,est_points)
    plot_3d_points_(gt_points,est_points)
    print(rmse)

