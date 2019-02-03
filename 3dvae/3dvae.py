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

def simulate(state_pairs,simulator,ae,k=1):
    for j in range(k):
        x = state_pairs[np.random.randint(len(state_pairs)),0]
        sim_x = [x]
        for i in range(1024):
            x = simulator.forward([x])[0]
            sim_x.append(x)
        for i in range(1024):
            frame = ae.generator(np.array([sim_x[i][128:,0]]))[0] # sim_x[i][128:]
            fig = plt.figure()
            plt.imshow(frame.reshape((h,w)), cmap='gray')
            plt.savefig('/home/ronnypetson/models/sim_frames_{}_{}.png'.format(j,i))
            plt.close(fig)

def train_simulator(num_epochs):
    frames = log_run()
    ae = VanillaAutoencoder([None,h,w,1], 1e-3, batch_size, latent_dim)
    # ae = VariationalAutoencoder([None,64,64,1], 1e-3, batch_size, latent_dim)
    # ae = ConvAutoencoder([None,h,w,1], 1e-3, batch_size)
    meta_ae = MetaVanillaAutoencoder([None,32,128,1], 1e-3, batch_size, latent_dim, '/home/ronnypetson/models/Vanilla_MetaAE')
    # meta_ae = VariationalAutoencoder(input_dim=[None,32,128,1], model_fname='/home/ronnypetson/models/Meta_VAE')
    # meta_ae = ConvAutoencoder([None,32,64,1], 1e-3, batch_size, model_fname='/home/ronnypetson/models/Conv_MetaAE')

    state_pairs = get_state_pairs_(frames,ae,meta_ae).reshape((-1,2,256,1))
    num_sample = len(state_pairs)

    # close the inner-most sessions first
    meta_ae.close_session()
    # ae.close_session()

    # simulator = Transition(model_fname='/home/ronnypetson/models/Vanilla_transition')
    # simulator = TransitionWGAN(model_fname='/home/ronnypetson/models/Transition_WGAN')
    # simulator = ConvTransition()
    simulator = Conv1DTransition()
    for epoch in range(num_epochs):
        for _ in range(num_sample // batch_size):
            # Obtina a batch
            batch = get_batch(state_pairs)
            x = [b[0] for b in batch] # + np.random.normal(0.0,1e-3,(batch_size,256))
            x_ = [b[1] for b in batch]
            # Execute the forward and the backward pass and report computed losses
            loss = simulator.run_single_step(x,x_) # , d_loss
        if epoch%30==29:
            simulator.save_model()
        print('[Epoch {}] Loss: {}'.format(epoch, loss))

    simulate(state_pairs,simulator,ae)
    simulator.close_session()
    ae.close_session()
    print('Done!')

def decode_seq():
    frames = log_run(500)[-32:] #.reshape((-1,1,h,w,1))
    ae = VanillaAutoencoder([None,h,w,1], 1e-3, batch_size, latent_dim)
    # ae = VariationalAutoencoder([None,64,64,1], 1e-3, batch_size, latent_dim)
    # ae = ConvAutoencoder([None,h,w,1], 1e-3, batch_size)
    # ae = Conv3DAutoencoder([None,1,h,w,1], 1e-3, batch_size)

    encodings = encode_(frames,ae) #get_encodings(frames,ae) # [1,128,32,1]
    encodings = stack_(encodings)

    meta_ae = MetaVanillaAutoencoder([None,32,128,1], 1e-3, batch_size, latent_dim, '/home/ronnypetson/models/Vanilla_MetaAE')
    # meta_ae = VariationalAutoencoder(input_dim=[None,32,128,1], model_fname='/home/ronnypetson/models/Meta_VAE')
    # meta_ae = ConvAutoencoder([None,32,64,1], 1e-3, batch_size, model_fname='/home/ronnypetson/models/Conv_MetaAE')
    # meta_ae = Conv3DAutoencoder([None,32,8,8,1], 1e-3, batch_size, model_fname='/home/ronnypetson/models/Conv3D_MetaAE')

    meta_enc = meta_ae.transformer(encodings) # shape [-1,256,1]
    rec_enc = meta_ae.generator(meta_enc) # shape([-1,32,256,1])
    rec_enc = rec_enc.reshape((32,128)) #.transpose((2,0,1)).reshape((32,16,16,1))
    rec_frames = ae.generator(rec_enc) # .reshape([32,128]) .reshape([32,64,64])

    # close the inner-most sessions first
    meta_ae.close_session()
    ae.close_session()

    fig = plt.figure()
    rec_imgs = np.empty((4*h,2*4*w))
    for i in range(4):
        for j in range(4):
            rec_imgs[i*h:(i+1)*h,2*j*w:(2*j+1)*w] = frames[4*i+j].reshape((h,w))
            rec_imgs[i*h:(i+1)*h,(2*j+1)*w:(2*j+2)*w] = rec_frames[4*i+j].reshape((h,w))
    plt.imshow(rec_imgs, cmap='gray')
    plt.savefig('/home/ronnypetson/models/rec_frames.png')
    plt.close(fig)

def train_meta_ae(num_epochs):
    frames = log_run() #.reshape((-1,1,h,w,1))
    # ae = VariationalAutoencoder([None,64,64,1], 1e-3, batch_size, latent_dim)
    ae = VanillaAutoencoder([None,h,w,1], 1e-3, batch_size, latent_dim)
    # ae = ConvAutoencoder([None,h,w,1], 1e-3, batch_size)
    # ae = Conv3DAutoencoder([None,1,h,w,1], 1e-3, batch_size)

    encodings = encode_(frames,ae)
    encodings = stack_(encodings)
    ae.close_session()
    meta_ae = None

    def save_reproductions():
        batch = get_batch(encodings)
        x_reconstructed = meta_ae.reconstructor(batch) #.reshape((128,batch_size))
        n = 16 #meta_ae.batch_size
        I_reconstructed = np.empty((128,(32+4)*n)) # Vertical strips
        for i in range(n):
            #x = np.concatenate((x_reconstructed[i].reshape(128,32),batch[i].reshape(128,32)),axis=1)
            x = x_reconstructed[i].reshape(128,32) - batch[i].reshape(128,32)
            x = np.concatenate((x,np.zeros((128,4))),axis=1)
            I_reconstructed[:,(32+4)*i:(32+4)*(i+1)] = x
        fig = plt.figure()
        plt.imshow(I_reconstructed, cmap='gray')
        plt.savefig('/home/ronnypetson/models/rec_encodings.png')
        plt.close(fig)

    num_sample=len(encodings)
    meta_ae = MetaVanillaAutoencoder([None,32,128,1], 1e-3, batch_size, latent_dim, '/home/ronnypetson/models/Vanilla_MetaAE')
    #meta_ae = VariationalAutoencoder(input_dim=[None,32,128,1], model_fname='/home/ronnypetson/models/Meta_VAE')
    #meta_ae = ConvAutoencoder([None,32,64,1], 1e-3, batch_size, model_fname='/home/ronnypetson/models/Conv_MetaAE')
    #meta_ae = Conv3DAutoencoder([None,32,8,8,1], 1e-3, batch_size, model_fname='/home/ronnypetson/models/Conv3D_MetaAE')
    for epoch in range(num_epochs):
        for _ in range(num_sample // batch_size):
            # Obtain a batch
            batch = get_batch(encodings)
            # Execute the forward and the backward pass and report computed losses
            loss = meta_ae.run_single_step(batch)
        if epoch%10==9:
            meta_ae.save_model()
            save_reproductions()
        print('[Epoch {}] Loss: {}'.format(epoch, loss))
    meta_ae.close_session()
    print('Done!')

def train_ae(num_epochs):
    frames = log_run() #.reshape((-1,1,h,w,1))
    num_sample = len(frames)
    model = None

    def sig_handler(sig,frame):
        model.save_model()
        sys.exit(0)

    signal.signal(signal.SIGINT,sig_handler)

    def save_reproductions():
        batch = get_batch(log_run(256)[-batch_size:]) #.reshape((-1,1,h,w,1)))
        x_reconstructed = model.reconstructor(batch)
        n = np.sqrt(model.batch_size).astype(np.int32)//2
        I_reconstructed = np.empty((h*n, 2*w*n))
        for i in range(n):
            for j in range(n):
                x = np.concatenate((x_reconstructed[i*n+j].reshape(h, w),batch[i*n+j].reshape(h, w)),axis=1)
                I_reconstructed[i*h:(i+1)*h, j*2*w:(j+1)*2*w] = x
        fig = plt.figure()
        plt.imshow(I_reconstructed, cmap='gray')
        plt.savefig('/home/ronnypetson/models/I_reconstructed.png')
        plt.close(fig)

    #model = VariationalAutoencoder([None,64,64,1], 1e-3, batch_size, latent_dim)
    model = VanillaAutoencoder([None,h,w,1], 1e-3, batch_size, latent_dim)
    #model = ConvAutoencoder([None,h,w,1], 1e-3, batch_size)
    #model = Conv3DAutoencoder([None,1,h,w,1], 1e-3, batch_size)
    for epoch in range(num_epochs):
        for _ in range(num_sample // batch_size):
            # Obtina a batch
            batch = get_batch(frames)
            # Execute the forward and the backward pass and report computed losses
            loss = model.run_single_step(batch)
        if epoch%5==4:
            save_reproductions()
        if epoch%10==9:
            model.save_model()
        print('[Epoch {}] Loss: {}'.format(epoch, loss))
    model.close_session()
    print('Done!')

def text_test():    
    paes = [DenseAutoencoder([None,8,27,1],1e-3,batch_size,32,'/home/ronnypetson/models/Dense_AE_text32_','dense_8_27_1__32'),\
            DenseAutoencoder([None,32,32,1],1e-3,batch_size,64,'/home/ronnypetson/models/Dense_AE2_text64_','dense_32_32_1__64'),\
            DenseAutoencoder([None,32,64,1],1e-3,batch_size,128,'/home/ronnypetson/models/Dense_AE3_text128__','dense_32_64_1__128_')]
    text_data = log_run_text('/home/ronnypetson/Documents/lusiadas.txt',wlen=8)
    #text_data = log_run_text('/home/ronnypetson/Documents/corruption',wlen=8)
    print(text_data.shape)
    #train_last_ae(paes,text_data[:90000],20)
    encode_decode_sequence(paes,text_data[-32*32:],data_type='text')

    '''
    tf.reset_default_graph()
    
    eaes = [DenseAutoencoder([None,8,27,1],1e-3,batch_size,32,'/home/ronnypetson/models/text/Dense_AE_eng_text32__','dense_eng_8_27_1__32')]#,\
           #DenseAutoencoder([None,32,32,1],1e-3,batch_size,128,'/home/ronnypetson/models/text/Dense_AE2_eng_text32_','dense_eng_32_32_1__128',False),\
           #DenseAutoencoder([None,32,32,1],1e-3,batch_size,64,'/home/ronnypetson/models/text/Dense_AE3_eng_text64_','dense_eng_32_32_1__64',False)]
    eng_text_data = log_run_text('/home/ronnypetson/Documents/eng_lusiads.txt',wlen=8)
    print(eng_text_data.shape)
    #train_last_ae(eaes[:1],eng_text_data[:90000],30)
    #encode_decode_sequence(eaes[:1],eng_text_data[-256:],data_type='text')
    '''

    #pt_text = text_data[-512:-480] # log_run_text('/home/ronnypetson/Documents/corruption',wlen=8)
    #t = Transition([None,32],[None,32],model_fname='/home/ronnypetson/models/Vanilla_transition_text__')
    #train_translator(t,paes,paes,text_data[:-32],text_data[32:],120)
    #data = up_(paes,pt_text)
    #data = t.forward(data)
    #down_(paes,data,pt_text)

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
    assert len(poses_abs)-len(sdivs)*31 == len(poses)
    exit()'''

    frames, sdivs = log_run_kitti_all()
    poses, poses_abs = load_kitti_odom_all()
    # Loading the encoder models
    aes = [VanillaAutoencoder([None,h,w,1],1e-3,batch_size,128,'/home/ronnypetson/models/Vanilla_AE_64x64_kitti'),\
           MetaVanillaAutoencoder([None,32,128,1],1e-3,batch_size,256,'/home/ronnypetson/models/Vanilla_Meta1_AE_kitti',False),\
           MetaVanillaAutoencoder([None,32,256,1],1e-3,batch_size,256,'/home/ronnypetson/models/Vanilla_Meta2_AE_kitti',False)]
    train_last_ae(aes[:1],frames,30)
    #encode_decode_sequence(aes[:1],frames[:129])
    # Mapping from state to pose
    '''t = Transition([None,256],[None,12],model_fname='/home/ronnypetson/models/Vanilla_transition_kitti')
    data_x = up_(aes[:2],frames,training=True)
    tf.reset_default_graph()
    print(len(data_x),len(poses))
    train_transition(t,data_x,poses,200)
    # Checking the estimated poses
    rmse, estimated = test_transition(t,data_x,poses)
    gt_points = get_3d_points(poses, poses_abs)
    est_points = get_3d_points(estimated, poses_abs)
    plot_3d_points_(gt_points,est_points)
    plot_2d_points_(gt_points,est_points)
    print(rmse)'''

