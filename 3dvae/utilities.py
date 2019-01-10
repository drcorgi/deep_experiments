import numpy as np
import gym
import cv2

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

def stack_(data,seq_len=32):
    sshape = (seq_len,data.shape[1],1) #(seq_len,data.shape[2],data.shape[3],1)
    stacked = [np.array(data[i:i+seq_len]).reshape(sshape) for i in range(len(data)-(seq_len-1))]
    stacked = np.stack(stacked,axis=0)
    return stacked

def unstack_(data,seq_len=32):
    data = data.transpose((0,2,3,1))
    return [np.split(d,seq_len) for d in data]

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

def train_last_ae(aes,data,num_epochs):
    current = aes[-1]
    base = aes[:-1]
    for ae in base:
        ae.load()
        data = stack_(encode_(data,ae))
        #tf.reset_default_graph()
        print('.')
    num_sample=len(data)
    current.load()
    for epoch in range(num_epochs):
        for _ in range(num_sample // batch_size):
            batch = get_batch(data)
            loss = current.run_single_step(batch)
        if epoch%10==9:
            current.save_model()
        print('[Epoch {}] Loss: {}'.format(epoch, loss))
    #tf.reset_default_graph()
    print('Done!')

