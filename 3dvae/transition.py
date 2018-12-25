import os
import numpy as np
import tensorflow as tf

class VanillaAutoencoder(object):
    def __init__(self, input_dim=[None,64,64,1], learning_rate=1e-3, batch_size=64, n_z=128, model_fname='/home/ronnypetson/models/Vanilla_AE_pong'):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_z = n_z
        self.model_fname = model_fname
        self.build()
        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver()
        if os.path.isfile(self.model_fname+'.meta'):
            try:
                self.saver.restore(self.sess,self.model_fname)
            except ValueError:
                self.sess.run(tf.global_variables_initializer())
                print('Cannot restore model')
        else:
            self.sess.run(tf.global_variables_initializer())

    # Build the netowrk and the loss functions
    def build(self):
        self.x = tf.placeholder(name='x', dtype=tf.float32, shape=self.input_dim)
        # Encode
        # x -> z_mean, z_sigma -> z
        conv1 = tf.layers.conv2d(self.x, 32, (5,5), (2,2), padding='same', activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, 64, (5,5), (2,2), padding='same', activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(conv2, 64, (3,3), (1,1), padding='same', activation=tf.nn.relu)
        flat1 = tf.layers.flatten(conv3)
        self.z = tf.layers.dense(flat1,self.n_z)

        # Decode
        # z -> x_hat
        dec1 = tf.layers.dense(self.z,16*16*64,activation=tf.nn.relu) # tf.shape(flat1)
        dec1 = tf.reshape(dec1,[-1,16,16,64]) # tf.shape(conv3)
        dec2 = tf.layers.conv2d_transpose(dec1, 64, (3,3), (1,1), padding='same', activation=tf.nn.relu)
        dec3 = tf.layers.conv2d_transpose(dec2, 64, (5,5), (2,2), padding='same', activation=tf.nn.relu)
        self.x_hat = tf.layers.conv2d_transpose(dec3, 1, (5,5), (2,2), padding='same', activation=tf.nn.relu)
        # Loss
        # Reconstruction loss
        # Minimize the cross-entropy loss
        # H(x, x_hat) = -\Sigma x*log(x_hat) + (1-x)*log(1-x_hat)
        #epsilon = 1e-10
        #recon_loss = -tf.reduce_sum(self.x * tf.log(epsilon+self.x_hat) + (1-self.x) * tf.log(epsilon+1-self.x_hat),axis=[1,2,3])
        self.total_loss = tf.losses.mean_squared_error(self.x,self.x_hat)
        #self.total_loss = tf.reduce_mean(recon_loss)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss)
        return

    # Execute the forward and the backward pass
    def run_single_step(self, x, save=False):
        _, loss = self.sess.run(
            [self.train_op, self.total_loss],
            feed_dict={self.x: x}
        )
        return loss
    # x -> x_hat
    def reconstructor(self, x):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.x: x})
        return x_hat
    # z -> x
    def generator(self, z):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.z: z})
        return x_hat
    # x -> z
    def transformer(self, x):
        z = self.sess.run(self.z, feed_dict={self.x: x})
        return z
    def save_model(self):
        self.saver.save(self.sess, self.model_fname)
    def close_session(self):
        self.sess.close()

class MetaVanillaAutoencoder(object):
    def __init__(self, input_dim=[None,128,32,1], learning_rate=1e-3, batch_size=64, n_z=128, model_fname='/home/ronnypetson/models/meta_encoder'):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_z = n_z
        self.model_fname = model_fname
        self.build()
        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver()
        if os.path.isfile(self.model_fname+'.meta'):
            try:
                self.saver.restore(self.sess,self.model_fname)
            except ValueError:
                self.sess.run(tf.global_variables_initializer())
                print('Cannot restore model')
        else:
            self.sess.run(tf.global_variables_initializer())

    # Build the netowrk and the loss functions
    def build(self):
        self.x = tf.placeholder(name='x', dtype=tf.float32, shape=self.input_dim)
        # Encode
        # x -> z_mean, z_sigma -> z
        conv1 = tf.layers.conv2d(self.x, 32, (5,5), (2,2), padding='same', activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, 64, (5,5), (2,2), padding='same', activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(conv2, 64, (3,3), (1,1), padding='same', activation=tf.nn.relu)
        flat1 = tf.layers.flatten(conv3)
        self.z = tf.layers.dense(flat1,self.n_z)

        # Decode
        # z -> x_hat
        dec1 = tf.layers.dense(self.z,32*8*64,activation=tf.nn.relu) # tf.shape(flat1)
        dec1 = tf.reshape(dec1,[-1,32,8,64]) # tf.shape(conv3)
        dec2 = tf.layers.conv2d_transpose(dec1, 64, (3,3), (1,1), padding='same', activation=tf.nn.relu)
        dec3 = tf.layers.conv2d_transpose(dec2, 64, (5,5), (2,2), padding='same', activation=tf.nn.relu)
        self.x_hat = tf.layers.conv2d_transpose(dec3, 1, (5,5), (2,2), padding='same', activation=tf.nn.relu)
        # Loss
        # Reconstruction loss
        # Minimize the cross-entropy loss
        # H(x, x_hat) = -\Sigma x*log(x_hat) + (1-x)*log(1-x_hat)
        #epsilon = 1e-10
        #recon_loss = -tf.reduce_sum(self.x * tf.log(epsilon+self.x_hat) + (1-self.x) * tf.log(epsilon+1-self.x_hat),axis=[1,2,3])
        self.total_loss = tf.losses.mean_squared_error(self.x,self.x_hat)
        #self.total_loss = tf.reduce_mean(recon_loss)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss)
        return

    # Execute the forward and the backward pass
    def run_single_step(self, x, save=False):
        _, loss = self.sess.run(
            [self.train_op, self.total_loss],
            feed_dict={self.x: x}
        )
        return loss
    # x -> x_hat
    def forward(self, x):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.x: x})
        return x_hat
    def save_model(self):
        self.saver.save(self.sess, self.model_fname)
    def close_session(self):
        self.sess.close()

