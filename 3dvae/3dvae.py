import sys
import signal
import numpy as np
import gym
import cv2
import matplotlib.pyplot as plt

from vae import *
from transition import *
from utilities import *

img_shape = [32,32,1]
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
            frame = ae.generator_(np.array([sim_x[i][:,16:]]))[0] # sim_x[i][128:]
            fig = plt.figure()
            plt.imshow(frame.reshape((h,w)), cmap='gray')
            plt.savefig('/home/ronnypetson/models/sim_frames_{}_{}.png'.format(j,i))
            plt.close(fig)

def train_simulator(num_epochs):
    frames = log_run()
    # ae = VanillaAutoencoder([None,64,64,1], 1e-3, batch_size, latent_dim)
    # ae = VariationalAutoencoder([None,64,64,1], 1e-3, batch_size, latent_dim)
    ae = ConvAutoencoder([None,h,w,1], 1e-3, batch_size)
    # meta_ae = MetaVanillaAutoencoder([None,32,128,1], 1e-3, batch_size, latent_dim, '/home/ronnypetson/models/meta_encoder')
    # meta_ae = VariationalAutoencoder(input_dim=[None,32,128,1], model_fname='/home/ronnypetson/models/Meta_VAE')
    meta_ae = ConvAutoencoder([None,32,64,1], 1e-3, batch_size, model_fname='/home/ronnypetson/models/Conv_MetaAE')

    state_pairs = get_state_pairs_(frames,ae,meta_ae)
    num_sample = len(state_pairs)

    # close the inner-most sessions first
    meta_ae.close_session()
    # ae.close_session()

    # simulator = Transition(model_fname='/home/ronnypetson/models/Vanilla_transition')
    # simulator = TransitionWGAN(model_fname='/home/ronnypetson/models/Transition_WGAN')
    simulator = ConvTransition()
    for epoch in range(num_epochs):
        for iter in range(num_sample // batch_size):
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
    frames = log_run(500)[-32:]
    # ae = VanillaAutoencoder([None,64,64,1], 1e-3, batch_size, latent_dim)
    # ae = VariationalAutoencoder([None,64,64,1], 1e-3, batch_size, latent_dim)
    ae = ConvAutoencoder([None,h,w,1], 1e-3, batch_size)
    encodings = encode_(frames,ae) #get_encodings(frames,ae) # [1,128,32,1]
    encodings = stack_(encodings)

    # meta_ae = MetaVanillaAutoencoder([None,32,128,1], 1e-3, batch_size, latent_dim, '/home/ronnypetson/models/meta_encoder')
    # meta_ae = VariationalAutoencoder(input_dim=[None,32,128,1], model_fname='/home/ronnypetson/models/Meta_VAE')
    meta_ae = ConvAutoencoder([None,32,64,1], 1e-3, batch_size, model_fname='/home/ronnypetson/models/Conv_MetaAE')

    meta_enc = meta_ae.transformer(encodings) # shape [-1,256,1]
    rec_enc = meta_ae.generator(meta_enc) # shape([-1,32,256,1])
    rec_enc = rec_enc.reshape((rec_enc.shape[1:])) #.transpose((2,0,1)).reshape((32,16,16,1))
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
    frames = log_run()
    # ae = VariationalAutoencoder([None,64,64,1], 1e-3, batch_size, latent_dim)
    # ae = VanillaAutoencoder([None,64,64,1], 1e-3, batch_size, latent_dim)
    ae = ConvAutoencoder([None,h,w,1], 1e-3, batch_size)
    encodings = encode_(frames,ae)
    encodings = stack_(encodings)
    ae.close_session()
    meta_ae = None

    def save_reproductions():
        batch = get_batch(encodings)
        x_reconstructed = meta_ae.reconstructor(batch) #.reshape((128,batch_size))
        n = 16 #meta_ae.batch_size
        I_reconstructed = np.empty((64,(32+4)*n)) # Vertical strips
        for i in range(n):
            #x = np.concatenate((x_reconstructed[i].reshape(128,32),batch[i].reshape(128,32)),axis=1)
            x = x_reconstructed[i].reshape(64,32) - batch[i].reshape(64,32)
            x = np.concatenate((x,np.zeros((64,4))),axis=1)
            I_reconstructed[:,(32+4)*i:(32+4)*(i+1)] = x
        fig = plt.figure()
        plt.imshow(I_reconstructed, cmap='gray')
        plt.savefig('/home/ronnypetson/models/rec_encodings.png')
        plt.close(fig)

    num_sample=len(encodings)
    #meta_ae = MetaVanillaAutoencoder([None,32,128,1], 1e-3, batch_size, latent_dim, '/home/ronnypetson/models/meta_encoder')
    #meta_ae = VariationalAutoencoder(input_dim=[None,32,128,1], model_fname='/home/ronnypetson/models/Meta_VAE')
    meta_ae = ConvAutoencoder([None,32,64,1], 1e-3, batch_size, model_fname='/home/ronnypetson/models/Conv_MetaAE')
    for epoch in range(num_epochs):
        for iter in range(num_sample // batch_size):
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
    frames = log_run()
    num_sample = len(frames)
    model = None

    def sig_handler(sig,frame):
        model.save_model()
        sys.exit(0)

    signal.signal(signal.SIGINT,sig_handler)

    def save_reproductions():
        batch = get_batch(log_run(256)[-batch_size:])
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
    #model = VanillaAutoencoder([None,64,64,1], 1e-3, batch_size, latent_dim)
    model = ConvAutoencoder([None,h,w,1], 1e-3, batch_size)
    for epoch in range(num_epochs):
        for iter in range(num_sample // batch_size):
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

if __name__ == '__main__':
    #train_ae(60)
    #train_meta_ae(80)
    #decode_seq()
    train_simulator(30)

