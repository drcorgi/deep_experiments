import numpy as np
import gym
import cv2
from copy import deepcopy
from matplotlib import pyplot as plt

from vae import *
from transition import *

img_shape = (32,32,1)
batch_size = 64
latent_dim = 128
h, w, _ = img_shape
env_name = 'Pong-v0'

def log_run(num_it=10000):
	env = gym.make(env_name)
	frames = []
	obs = env.reset()
	for t in range(num_it):
		env.render()
		act = env.action_space.sample()
		obs = cv2.cvtColor(obs,cv2.COLOR_BGR2GRAY)
		obs = cv2.resize(obs,img_shape[:-1],interpolation=cv2.INTER_AREA).reshape(img_shape)
		frames.append(obs/255.0) # [obs, act]
		obs, rwd, done, _ = env.step(act)
		if done:
			obs = env.reset()
	env.close()
	return np.array(frames,dtype=np.float32)

def get_batch(data):
    inds = np.random.choice(range(data.shape[0]), batch_size, False)
    return np.array([data[i] for i in inds],dtype=np.float32)

def encode_(data,ae):
    enc = []
    num_batches = (len(data)+batch_size)//batch_size
    for i in range(num_batches):
        b = data[i*batch_size:(i+1)*batch_size]
        if len(b) > 0:
            enc += ae.transformer(b).tolist()
    return np.array(enc,dtype=np.float32)

def stack_(data,seq_len=32,offset=1,blimit=1): # change offset for higher encodings
    sshape = (seq_len,data.shape[1],1)
    stacked = [np.array(data[range(i,i+offset*seq_len,offset)]).reshape(sshape) for i in range(blimit)]
    stacked = np.stack(stacked,axis=0)
    return stacked

def decode_(data,ae,seq_len=32,offset=1,start=0,base=False):
    dec = []
    if not base: data = data[range(start,start+offset*seq_len,offset)]
    num_batches = (len(data)+batch_size)//batch_size
    for i in range(num_batches):
        b = data[i*batch_size:(i+1)*batch_size]
        if len(b) > 0:
            dec += ae.generator(b).tolist() # .reshape(seq_len,data.shape[-1])
    return np.array(dec,dtype=np.float32)

def unstack_(data,seq_len=32): # Ex.: (1,32,128,1) -> (32,128)
    d = []
    for a in data:
        d += a.reshape((seq_len,data.shape[-2])).tolist()
    d = np.array(d,dtype=np.float32)
    return d #data.reshape((-1,data.shape[-2]))

def plot_data(data):
    im_shape = (data.shape[1],data.shape[2])
    for i in range(32): # len(data)
        fig = plt.figure()
        plt.imshow(data[i].reshape(im_shape), cmap='gray')
        plt.savefig('/home/ronnypetson/models/data_plot_{}.png'.format(i))
        plt.close(fig)

def train_last_ae(aes,data,num_epochs,seq_len=32):
    current = aes[-1]
    base = aes[:-1]
    offset = 1
    for ae in base:
        #ae.load()
        data = stack_(encode_(data,ae),offset=offset,blimit=len(data)-offset*(seq_len-1))
        offset *= seq_len
        print('.')
    num_sample=len(data)
    #current.load()
    for epoch in range(num_epochs):
        for _ in range(num_sample // batch_size):
            batch = get_batch(data)
            loss = current.run_single_step(batch)
        if epoch%10==9:
            current.save_model()
        print('[Epoch {}] Loss: {}'.format(epoch, loss))
    print('Done!')

def encode_decode_sequence(aes,data,seq_len=32):
    # Separate base data for later comparison with the reconstruction
    base_data = deepcopy(data)
    # Obtain base encodings
    data = encode_(data,aes[0])
    offset = 1
    # Obtain meta-encodings from the middle
    for ae in aes[1:]:
        #ae.load()
        data = stack_(data,offset=offset,blimit=len(data)-offset*(seq_len-1))
        offset *= 32
        data = encode_(data,ae)
    # Reconstruct original data
    aes.reverse()
    for ae in aes[:-1]:
        data = decode_(data,ae,offset=offset)
        offset = offset//32
        data = unstack_(data)
    data = decode_(data,aes[-1],offset=1,base=True)
    # Compare the reconstructions
    im_shape = (data.shape[1],2*data.shape[2])
    for i in range(128): # len(data)
        side_by_side = np.concatenate((base_data[i],data[i]),axis=1)
        fig = plt.figure()
        plt.imshow(side_by_side.reshape(im_shape), cmap='gray')
        plt.savefig('/home/ronnypetson/models/rec_1024_{}.png'.format(i))
        plt.close(fig)

def get_encodings(frames,model,enc_shape=[-1,32,128,1],meta=False):
    enc = []
    num_batches = (len(frames)+batch_size)//batch_size
    for i in range(num_batches):
        b = frames[i*batch_size:(i+1)*batch_size]
        if len(b) > 0:
            enc += model.transformer(b).tolist()
    if meta:
        return np.array(enc,dtype=np.float32)
    else:
        return np.array([ np.array(enc[i:i+32]).reshape(enc_shape[1:]) for i in range(len(frames)-31)],dtype=np.float32)

def get_state_pairs_(frames,ae1,ae2,seq_len=32):
    enc1_ = encode_(frames,ae1) # [b,h,w,c]
    enc1 = stack_(enc1_)
    enc2 = encode_(enc1,ae2)
    states = []
    for i in range(enc2.shape[0]):
        states.append(np.concatenate((enc2[i].reshape(128),enc1_[i+seq_len-1].reshape(128)),axis=0))
    return np.array([[states[i],states[i+1]] for i in range(len(states)-1)],dtype=np.float32)

def get_state_pairs(frames,ae1,ae2):
    enc1 = get_encodings(frames,ae1)
    enc2 = get_encodings(enc1,ae2,True)
    enc1 = enc1.reshape([-1,32,128])
    states = []
    for i in range(enc2.shape[0]):
        states.append(np.concatenate((enc2[i],enc1[i,-1]),axis=0))
    return np.array([[states[i],states[i+1]] for i in range(len(states)-1)],dtype=np.float32)

