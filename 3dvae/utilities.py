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
env_name = 'Assault-v0' #'Breakout-v0' #'Pong-v0'

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

def treat_string(s):
    s_ = ''
    for c in s:
        if ord(c) < 97 or ord(c) > 122:
            c = ' '
        s_ += c
    return s_

def get_ords(s):
    ords = []
    for c in s:
        if c == ' ':
            ords.append(26)
        else:
            ords.append(ord(c)-97)
    return ords

def get_str(ords):
    s = ''
    for o in ords:
        if o == 26:
            s += ' '
        else:
            s += chr(o+97)
    return s

def log_run_text(fname,max_chars=100000,wlen=32):
    with open(fname,'r',encoding='ISO-8859-1') as source:
        data = source.read().lower()
        data = data[:max_chars]
        data = treat_string(data)
        len_data = len(data)
        ords = get_ords(data)
        np_data = np.zeros((len_data,27))
        np_data[np.arange(len_data),ords] = 1.0
        print(len_data)
        return np.array([np_data[i:i+wlen] for i in range(0,len_data-wlen+1,wlen)]).reshape((-1,wlen,27,1))

def cat2text(data):
    ords = []
    for w in data:
        ords += [np.argmax(c) for c in w]
    return get_str(ords)

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

def stack_(data,seq_len=32,offset=1,blimit=1,training=False): # change offset for higher encodings
    sshape = (seq_len,data.shape[1],1)
    if training:
        stacked = [np.array(data[range(i,i+offset*seq_len,offset)]).reshape(sshape) for i in range(blimit)]
    else:
        stacked = [np.array(data[i:i+seq_len]).reshape(sshape) for i in range(0,blimit,seq_len)]
    stacked = np.stack(stacked,axis=0)
    return stacked

def decode_(data,ae,seq_len=32,offset=1,start=0,base=False):
    dec = []
    #if not base: data = data[range(start,start+offset*seq_len,offset)]
    if not base: data = data[range(start,len(data),offset)]
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
        data = stack_(encode_(data,ae),offset=offset,blimit=len(data)-offset*(seq_len-1),training=True)
        offset *= seq_len
        print('.')
        print(data.shape)
    num_sample=len(data)
    for epoch in range(num_epochs):
        for _ in range(num_sample // batch_size):
            batch = get_batch(data)
            loss = current.run_single_step(batch)
        if epoch%10==9:
            current.save_model()
        print('[Epoch {}] Loss: {}'.format(epoch, loss))
    print('Done!')

def encode_decode_sequence(aes,data,seq_len=32,data_type='image'):
    base_data = deepcopy(data)
    data = encode_(data,aes[0])
    offset = seq_len
    for ae in aes[1:]:
        print(data.shape)
        data = stack_(data,offset=offset,blimit=len(data)-(seq_len-1))
        data = encode_(data,ae)
    offset = 1 #1024
    aes.reverse()
    for ae in aes[:-1]:
        print(data.shape)
        data = decode_(data,ae,offset=offset)
        print(data.shape)
        data = unstack_(data)
    print(data.shape)
    data = decode_(data,aes[-1],offset=1,base=True)
    if data_type=='text':
        base_data = np.reshape(base_data,(-1,64,27))
        data = np.reshape(data,(-1,64,27))
        print(base_data.shape,data.shape)
        base_text = cat2text(base_data)
        decoded_text = cat2text(data)
        print(base_text)
        print()
        print(decoded_text)
    else:
        im_shape = (data.shape[1],2*data.shape[2])
        for i in range(1024): # len(data)
            side_by_side = np.concatenate((base_data[i],data[i]),axis=1)
            fig = plt.figure()
            plt.imshow(side_by_side.reshape(im_shape), cmap='gray')
            plt.savefig('/home/ronnypetson/models/rec_1024_{}.png'.format(i))
            plt.close(fig)

'''
    To be deprecated
'''
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

