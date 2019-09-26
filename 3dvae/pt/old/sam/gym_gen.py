import numpy as np
import gym
import cv2

def env_gen(env_name='Pong-v0',img_shape=(32,32),seq_len=32):
    env = gym.make(env_name)
    frames = []
    obs = env.reset()
    while True:
        try:
            x = []
            for i in range(seq_len):
                env.render()
                act = env.action_space.sample()
                obs = cv2.cvtColor(obs,cv2.COLOR_BGR2GRAY)
                obs = cv2.resize(obs,img_shape,interpolation=cv2.INTER_AREA) #.reshape(img_shape)
                x.append(obs)
                obs, rwd, done, _ = env.step(act)
                if done:
                    obs = env.reset()
            yield np.array(x)
        except Exception as e:
            print(e)
    env.close()

def batchify(gen,batch_size=32):
    while True:
        try:
            x = []
            for i in range(batch_size):
                x.append(next(gen))
            yield np.array(x)
        except StopIteration as e:
            print(e)
            break

if __name__ == '__main__':
    gen = env_gen()
    b = batchify(gen)
    for i in range(1):
        for nxt in b:
            print(nxt.shape)
