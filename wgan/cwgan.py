"""Minimal implementation of Wasserstein GAN for MNIST."""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os
from tensorflow.contrib import layers
from tensorflow.examples.tutorials.mnist import input_data

img_dim = 16
model_fn = './cwgan_16'

session = tf.InteractiveSession()

def get_prev(batch,m):
    prev = []
    for i in range(len(batch[0])):
        y = (batch[1][i]+9)%10
        while True:
            b = m.train.next_batch(1)
            if b[1][0] == y:
                prev.append(b[0][0])
                break
    return np.array(prev)

def process_images(imgs):
    images = imgs.reshape([-1, 28, 28, 1])
    images = [cv2.resize(img,(img_dim,img_dim),interpolation=cv2.INTER_AREA).reshape((img_dim,img_dim,1)) for img in images]
    return np.array(images)

def leaky_relu(x):
    return tf.maximum(x, 0.2 * x)


def generator(z):
    with tf.variable_scope('generator'):
        #z = layers.fully_connected(z, num_outputs=1024) # 4096
        #z = tf.reshape(z, [-1, 2, 2, 256]) # 4, 4

        z = layers.conv2d_transpose(z, num_outputs=128, kernel_size=5, stride=1,padding='valid') # stride=2
        z = layers.conv2d_transpose(z, num_outputs=64, kernel_size=5, stride=1) # stride=2
        z = layers.conv2d_transpose(z, num_outputs=1, kernel_size=5, stride=1,
                                    activation_fn=tf.nn.sigmoid)
        return z[:, 2:-2, 2:-2, :]


def discriminator(x, reuse):
    with tf.variable_scope('discriminator', reuse=reuse):
        x = layers.conv2d(x, num_outputs=64, kernel_size=5, stride=2,
                          activation_fn=leaky_relu)
        x = layers.conv2d(x, num_outputs=128, kernel_size=5, stride=2,
                          activation_fn=leaky_relu)
        x = layers.conv2d(x, num_outputs=256, kernel_size=5, stride=2,
                          activation_fn=leaky_relu)

        x = layers.flatten(x)
        return layers.fully_connected(x, num_outputs=1, activation_fn=None)


with tf.name_scope('placeholders'):
    x_true = tf.placeholder(tf.float32, [None, img_dim, img_dim, 1])
    x_prev = tf.placeholder(tf.float32, [None, img_dim, img_dim, 1])
    z = tf.placeholder(tf.float32, [None, img_dim, img_dim, 1]) # x_prev + noise [None, 128]


x_generated = generator(z)

d_true = discriminator(tf.concat([x_true,x_prev],3), reuse=False)
d_generated = discriminator(tf.concat([x_generated,x_prev],3), reuse=True)

with tf.name_scope('regularizer'):
    epsilon = tf.random_uniform([50, 1, 1, 1], 0.0, 1.0)
    x_hat = epsilon * x_true + (1 - epsilon) * x_generated
    d_hat = discriminator(tf.concat([x_hat,x_prev],3), reuse=True)

    gradients = tf.gradients(d_hat, x_hat)[0]
    ddx = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2]))
    d_regularizer = tf.reduce_mean((ddx - 1.0) ** 2)

with tf.name_scope('loss'):
    g_loss = tf.reduce_mean(d_generated)
    d_loss = (tf.reduce_mean(d_true) - tf.reduce_mean(d_generated) +
              10 * d_regularizer)

with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0, beta2=0.9)

    g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
    g_train = optimizer.minimize(g_loss, var_list=g_vars)
    d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
    d_train = optimizer.minimize(d_loss, var_list=d_vars)

saver = tf.train.Saver()
if os.path.isfile(model_fn+'.meta'):
    saver.restore(session,model_fn) # session
else:
    print('model not found')
    tf.global_variables_initializer().run()

mnist = input_data.read_data_sets('MNIST_data')
plt.ion()
plt.show()
for i in range(10001):
    batch = mnist.train.next_batch(50)
    prev_batch = get_prev(batch,mnist)
    images = process_images(batch[0])
    prev_images = process_images(prev_batch)
    #images = np.concatenate((prev_images,images),axis=3)
    z_train = np.random.randn(50, img_dim, img_dim, 1)*0.06 + prev_images

    session.run(g_train, feed_dict={z: z_train,x_prev: prev_images})
    for j in range(5):
        session.run(d_train, feed_dict={x_true: images, x_prev: prev_images, z: z_train})

    if i % 100 == 0:
        print('iter={}/20000'.format(i))
        z_validate = z_train[0].reshape((-1,img_dim,img_dim,1))
        generated = x_generated.eval(feed_dict={z: z_validate}).squeeze()
        #print(generated.shape)
        #plt.clf()
        plt.close()
        f, axarr = plt.subplots(2,2)
        axarr[0,0].imshow(prev_images[0].squeeze())
        axarr[0,1].imshow(z_validate.squeeze())
        axarr[1,0].imshow(generated)
        axarr[1,1].imshow(images[0].squeeze())
        plt.draw()
        plt.pause(0.01)
        saver.save(session,model_fn)

