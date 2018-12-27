import sys
import signal
import numpy as np
import gym
import cv2
import matplotlib.pyplot as plt

from vae import *
from transition import *

img_shape = (64,64,1)
batch_size = 64
latent_dim = 128
h, w, _ = img_shape
#sess = tf.InteractiveSession()

def log_run(num_it=10000):
	env = gym.make('Pong-v0')
	frames = []
	obs = env.reset()
	for t in range(num_it):
		env.render()
		act = env.action_space.sample()
		obs = cv2.cvtColor(obs,cv2.COLOR_BGR2GRAY)
		obs = cv2.resize(obs,img_shape[:-1]).reshape(img_shape)
		frames.append(obs/255.0) # [obs, act]
		obs, rwd, done, _ = env.step(act)
		if done:
			obs = env.reset()
	env.close()
	return np.array(frames)

def get_batch(data):
    inds = np.random.choice(range(data.shape[0]), batch_size, False)
    return np.array([data[i] for i in inds])

def get_encodings(frames,model,meta=False):
    enc = []
    num_batches = (len(frames)+batch_size)//batch_size
    for i in range(num_batches):
        b = frames[i*batch_size:(i+1)*batch_size]
        if len(b) > 0:
            enc += model.transformer(b).tolist()
    if meta:
        return np.array(enc)
    else:
        return np.array([np.array(enc[i:i+32]).reshape([32,128,1]) for i in range(len(frames)-31)])

def get_state_pairs(frames,ae1,ae2):
    enc1 = get_encodings(frames,ae1)
    enc2 = get_encodings(enc1,ae2,True)
    enc1 = enc1.reshape([-1,32,128])
    states = []
    for i in range(enc2.shape[0]):
        states.append(np.concatenate((enc2[i],enc1[i,-1]),axis=0))
    return np.array([[states[i],states[i+1]] for i in range(len(states)-1)])

def train_simulator():
    frames = log_run()
    ae = VanillaAutoencoder([None,64,64,1], 1e-3, batch_size, latent_dim)
    meta_ae = MetaVanillaAutoencoder([None,32,128,1], 1e-3, batch_size, latent_dim, '/home/ronnypetson/models/meta_encoder')
    state_pairs = get_state_pairs(frames,ae,meta_ae)
    num_sample = len(state_pairs)

    # close the inner-most sessions first
    meta_ae.close_session()
    #ae.close_session()

    #simulator = Transition(model_fname='/home/ronnypetson/models/Vanilla_transition')
    simulator = TransitionWGAN(model_fname='/home/ronnypetson/models/Transition_WGAN')
    for epoch in range(480):
        for iter in range(num_sample // batch_size):
            # Obtina a batch
            batch = get_batch(state_pairs)
            x = [b[0] for b in batch]
            x_ = [b[1] for b in batch]
            # Execute the forward and the backward pass and report computed losses
            g_loss, d_loss = simulator.run_single_step(x,x_)
        if epoch%30==29:
            simulator.save_model()
        print('[Epoch {}] Loss: {} {}'.format(epoch, g_loss, d_loss))

    x = state_pairs[100,0]
    sim_x = [x]
    for i in range(32):
        x = simulator.forward([x])[0]
        sim_x.append(x)
        #sim_x.append(state_pairs[i,1])
    simulator.close_session()

    #ae = VanillaAutoencoder([None,64,64,1], 1e-3, batch_size, latent_dim)
    sim_frames = np.empty((4*64,8*64))
    for i in range(32):
        frame = ae.generator([sim_x[i][128:]])[0] # state_pairs[i][0][128:]
        r,c = (i//8),(i%8)
        sim_frames[r*64:(r+1)*64,c*64:(c+1)*64] = frame.reshape((64,64))
    ae.close_session()

    fig = plt.figure()
    plt.imshow(sim_frames, cmap='gray')
    plt.savefig('sim_frames.png')
    plt.close(fig)
    print('Done!')

def decode_seq():
    frames = log_run(500)[-32:]
    ae = VanillaAutoencoder([None,64,64,1], 1e-3, batch_size, latent_dim)
    encodings = get_encodings(frames,ae) # [1,128,32,1]

    meta_ae = MetaVanillaAutoencoder([None,32,128,1], 1e-3, batch_size, latent_dim, '/home/ronnypetson/models/meta_encoder')
    meta_enc = meta_ae.transformer(encodings) # shape [1,128]
    rec_enc = meta_ae.generator(meta_enc) # shape([1,128,32,1])
    rec_frames = ae.generator(rec_enc.reshape([32,128])).reshape([32,64,64])

    # close the inner-most sessions first
    meta_ae.close_session()
    ae.close_session()

    fig = plt.figure()
    rec_imgs = np.empty((4*h,2*4*w))
    for i in range(4):
        for j in range(4):
            rec_imgs[i*h:(i+1)*h,2*j*w:(2*j+1)*w] = frames[4*i+j].reshape((64,64))
            rec_imgs[i*h:(i+1)*h,(2*j+1)*w:(2*j+2)*w] = rec_frames[4*i+j]
    plt.imshow(rec_imgs, cmap='gray')
    plt.savefig('rec_frames.png')
    plt.close(fig)

def train_meta_ae():
    frames = log_run()
    ae = VanillaAutoencoder([None,64,64,1], 1e-3, batch_size, latent_dim)
    encodings = get_encodings(frames,ae)
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
        plt.savefig('rec_encodings.png')
        plt.close(fig)

    num_sample=len(encodings)
    meta_ae = MetaVanillaAutoencoder([None,32,128,1], 1e-3, batch_size, latent_dim, '/home/ronnypetson/models/meta_encoder')
    for epoch in range(100):
        for iter in range(num_sample // batch_size):
            # Obtain a batch
            batch = get_batch(encodings)
            # Execute the forward and the backward pass and report computed losses
            loss = meta_ae.run_single_step(batch)
        if epoch%5==4:
            save_reproductions()
        if epoch%10==9:
            meta_ae.save_model()
        print('[Epoch {}] Loss: {}'.format(epoch, loss))
    meta_ae.close_session()
    print('Done!')

def train_ae():
    frames = log_run()
    num_sample = len(frames)
    model = None

    def sig_handler(sig,frame):
        model.save_model()
        sys.exit(0)

    signal.signal(signal.SIGINT,sig_handler)

    def save_reproductions():
        batch = get_batch(log_run(64))
        x_reconstructed = model.reconstructor(batch)
        n = np.sqrt(model.batch_size).astype(np.int32)//2
        I_reconstructed = np.empty((h*n, 2*w*n))
        for i in range(n):
            for j in range(n):
                x = np.concatenate((x_reconstructed[i*n+j].reshape(h, w),batch[i*n+j].reshape(h, w)),axis=1)
                I_reconstructed[i*h:(i+1)*h, j*2*w:(j+1)*2*w] = x
        fig = plt.figure()
        plt.imshow(I_reconstructed, cmap='gray')
        plt.savefig('I_reconstructed.png')
        plt.close(fig)

    #model = VariantionalAutoencoder([None,64,64,1], 1e-3, batch_size, latent_dim)
    model = VanillaAutoencoder([None,64,64,1], 1e-3, batch_size, latent_dim)
    for epoch in range(60):
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
    #save_reproductions()
    model.close_session()
    print('Done!')

if __name__ == '__main__':
    #train_ae()
    #train_meta_ae()
    #decode_seq()
    train_simulator()

