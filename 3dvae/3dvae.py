import sys
import signal
import numpy as np
import gym
import cv2
import matplotlib.pyplot as plt

from vae import *

img_shape = (64,64,1)
batch_size = 64
latent_dim = 128
h, w, _ = img_shape

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

def get_encodings(frames,model):
    enc = []
    num_batches = len(frames)//batch_size
    remainder = len(frames)%batch_size
    for i in range(num_batches):
        b = frames[i*num_batches:(i+1)*num_batches]
        if len(b) > 0:
            enc += model.transformer(b).tolist()
    #if remainder > 0:
    #    b = frames[-remainder:]
    #    enc += model.transformer(b).tolist()
    return np.array([np.array(enc[i:i+32]).reshape([128,32,1]) for i in range(len(frames)-32)])

def get_meta_encodings():
    frames = log_run()
    ae = VanillaAutoencoder([None,64,64,1], 1e-3, batch_size, latent_dim)
    encodings = get_encodings(frames,ae)
    ae.close_session()

    print(encodings.shape)
    num_sample=len(encodings)
    meta_ae = MetaVanillaAutoencoder([None,128,32,1], 1e-3, batch_size, latent_dim, '/home/ronnypetson/models/meta_encoder')
    for epoch in range(100):
        for iter in range(num_sample // batch_size):
            # Obtina a batch
            batch = get_batch(encodings)
            # Execute the forward and the backward pass and report computed losses
            loss = meta_ae.run_single_step(batch)
        if epoch%10==9:
            meta_ae.save_model()
        print('[Epoch {}] Loss: {}'.format(epoch, loss))
    print('Done!')

def main():
    frames = log_run()
    num_sample = len(frames)
    model = None

    def sig_handler(sig,frame):
        model.save_model()
        sys.exit(0)

    signal.signal(signal.SIGINT,sig_handler)

    def save_reproductions():
        batch = get_batch(frames)
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
    for epoch in range(100):
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
    print('Done!')

if __name__ == '__main__':
    #main()
    get_meta_encodings()

