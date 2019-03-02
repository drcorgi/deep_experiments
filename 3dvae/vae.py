import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.framework import checkpoint_utils

################################

class VanillaAutoencoder(object):
    def __init__(self, input_dim=[None,64,64,1], learning_rate=1e-3, batch_size=64, n_z=128, model_fname='/home/ronnypetson/models/Vanilla_AE_pong', load=True):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_z = n_z
        self.model_fname = model_fname
        if load: self.load()

    # Build the netowrk and the loss functions
    def build(self):
        self.x = tf.placeholder(name='x', dtype=tf.float32, shape=self.input_dim)
        # Encode
        # x -> z
        conv1 = tf.layers.conv2d(self.x, 32, (5,5), (2,2), padding='same', activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, 64, (3,3), (2,2), padding='same', activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(conv2, 64, (3,3), (1,1), padding='same', activation=tf.nn.relu)
        flat1 = tf.layers.flatten(conv3)
        self.z = tf.layers.dense(flat1,self.n_z)
        # Decode
        # z -> x_hat
        new_dim = [-1,self.input_dim[1]//4,self.input_dim[2]//4,64]
        dec1 = tf.layers.dense(self.z,np.prod(new_dim[1:]),activation=tf.nn.relu) # tf.shape(flat1)
        dec1 = tf.reshape(dec1,new_dim) # tf.shape(conv3)
        dec2 = tf.layers.conv2d_transpose(dec1, 64, (3,3), (1,1), padding='same', activation=tf.nn.relu)
        dec3 = tf.layers.conv2d_transpose(dec2, 64, (3,3), (2,2), padding='same', activation=tf.nn.relu)
        self.x_hat = tf.layers.conv2d_transpose(dec3, self.input_dim[-1], (5,5), (2,2), padding='same', activation=None) # None
        self.total_loss = tf.losses.mean_squared_error(self.x,self.x_hat)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss)

    def load(self):
        self.build()
        self.sess = tf.InteractiveSession() # Interactive
        self.saver = tf.train.Saver()
        if os.path.isfile(self.model_fname+'.meta'):
            try:
                self.saver.restore(self.sess,self.model_fname)
            except ValueError:
                print('Cannot restore model')
        else:
            print('Model file not found')
            self.sess.run(tf.global_variables_initializer())

    # Execute the forward and the backward pass
    def run_single_step(self, x):
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
    def __init__(self, input_dim=[None,32,128,1], learning_rate=1e-3, batch_size=64, n_z=128, model_fname='/home/ronnypetson/models/Vanilla_MetaAE', load=True):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_z = n_z
        self.model_fname = model_fname
        #self.build()
        #self.sess = tf.InteractiveSession() # Interactive
        #self.saver = tf.train.Saver()
        if load: self.load()

    # Build the network and the loss functions
    def build(self):
        self.x = tf.placeholder(name='x', dtype=tf.float32, shape=self.input_dim)
        # Encode
        # x -> z
        conv1 = tf.layers.conv2d(self.x, 32, (5,5), (2,2), padding='same', activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, 64, (3,3), (2,2), padding='same', activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(conv2, 64, (3,3), (1,1), padding='same', activation=tf.nn.relu)
        flat1 = tf.layers.flatten(conv3)
        self.z = tf.layers.dense(flat1,self.n_z) # ,activation=tf.nn.relu
        # Decode
        # z -> x_hat
        new_dim = [-1,self.input_dim[1]//4,self.input_dim[2]//4,64]
        dec1 = tf.layers.dense(self.z,np.prod(new_dim[1:]),activation=tf.nn.relu) # tf.shape(flat1)
        dec1 = tf.reshape(dec1,new_dim) # tf.shape(conv3)
        dec2 = tf.layers.conv2d_transpose(dec1, 64, (3,3), (1,1), padding='same', activation=tf.nn.relu)
        dec3 = tf.layers.conv2d_transpose(dec2, 64, (3,3), (2,2), padding='same', activation=tf.nn.relu)
        self.x_hat = tf.layers.conv2d_transpose(dec3, 1, (5,5), (2,2), padding='same', activation=None)
        self.total_loss = tf.losses.mean_squared_error(self.x,self.x_hat)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss)

    def load(self):
        self.build()
        self.sess = tf.InteractiveSession() # Interactive
        self.saver = tf.train.Saver()
        if os.path.isfile(self.model_fname+'.meta'):
            try:
                self.saver.restore(self.sess,self.model_fname)
            except ValueError:
                print('Cannot restore model')
        else:
            print('Model file not found')
            self.sess.run(tf.global_variables_initializer())

    # Execute the forward and the backward pass
    def run_single_step(self, x):
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

class Vanilla1DAutoencoder(object):
    def __init__(self, input_dim=[None,64,64], learning_rate=1e-3, batch_size=64, n_z=128, model_fname='/home/ronnypetson/models/VanillaAE1D', load=True):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_z = n_z
        self.model_fname = model_fname
        #self.build()
        #self.sess = tf.InteractiveSession() # Interactive
        #self.saver = tf.train.Saver()
        if load: self.load()

    # Build the netowrk and the loss functions
    def build(self):
        self.x = tf.placeholder(name='x', dtype=tf.float32, shape=self.input_dim)
        # Encode
        # x -> z
        conv1 = tf.layers.conv1d(self.x, 32, (3,), (1,), padding='same', activation=tf.nn.relu)
        conv2 = tf.layers.conv1d(conv1, 64, (3,), (1,), padding='same', activation=tf.nn.relu)
        conv3 = tf.layers.conv1d(conv2, 64, (3,), (1,), padding='same', activation=tf.nn.relu)
        flat1 = tf.layers.flatten(conv3)
        self.z = tf.layers.dense(flat1,self.n_z)
        # Decode
        # z -> x_hat
        new_dim = [-1,self.input_dim[1],64]
        dec1 = tf.layers.dense(self.z,np.prod(new_dim[1:]),activation=tf.nn.relu) # tf.shape(flat1)
        dec1 = tf.reshape(dec1,new_dim) # tf.shape(conv3)
        dec2 = tf.layers.conv1d(dec1, 64, (3,), (1,), padding='same', activation=tf.nn.relu)
        dec3 = tf.layers.conv1d(dec2, 64, (3,), (1,), padding='same', activation=tf.nn.relu)
        self.x_hat = tf.layers.conv1d(dec3, self.input_dim[-1], (3,), (1,), padding='same', activation=None) # None
        self.total_loss = tf.losses.mean_squared_error(self.x,self.x_hat)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss)

    def load(self):
        self.build()
        self.sess = tf.InteractiveSession() # Interactive
        self.saver = tf.train.Saver()
        if os.path.isfile(self.model_fname+'.meta'):
            try:
                self.saver.restore(self.sess,self.model_fname)
            except ValueError:
                print('Cannot restore model')
        else:
            print('Model file not found')
            self.sess.run(tf.global_variables_initializer())

    # Execute the forward and the backward pass
    def run_single_step(self, x):
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

class Vanilla2DAutoencoder(object):
    def __init__(self, input_dim=[None,64,64], learning_rate=1e-3, batch_size=64, n_z=128, model_fname='/home/ronnypetson/models/Vanilla_AE_pong', load=True):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_z = n_z
        self.model_fname = model_fname
        #self.build()
        #self.sess = tf.InteractiveSession() # Interactive
        #self.saver = tf.train.Saver()
        if load: self.load()

    # Build the netowrk and the loss functions
    def build(self):
        self.x = tf.placeholder(name='x', dtype=tf.float32, shape=self.input_dim)
        # Encode
        # x -> z
        x_1 = tf.reshape(self.x,[-1,self.input_dim[1],self.input_dim[2],1])
        conv1 = tf.layers.conv2d(x_1, 32, (5,5), (2,2), padding='same', activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, 64, (3,3), (2,2), padding='same', activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(conv2, 64, (3,3), (1,1), padding='same', activation=tf.nn.relu)
        flat1 = tf.layers.flatten(conv3)
        self.z = tf.layers.dense(flat1,self.n_z)
        # Decode
        # z -> x_hat
        new_dim = [-1,self.input_dim[1]//4,self.input_dim[2]//4,64]
        dec1 = tf.layers.dense(self.z,np.prod(new_dim[1:]),activation=tf.nn.relu) # tf.shape(flat1)
        dec1 = tf.reshape(dec1,new_dim) # tf.shape(conv3)
        dec2 = tf.layers.conv2d_transpose(dec1, 64, (3,3), (1,1), padding='same', activation=tf.nn.relu)
        dec3 = tf.layers.conv2d_transpose(dec2, 64, (3,3), (2,2), padding='same', activation=tf.nn.relu)
        x_hat = tf.layers.conv2d_transpose(dec3, 1, (5,5), (2,2), padding='same', activation=None) # None
        self.x_hat = tf.reshape(x_hat,[-1,self.input_dim[1],self.input_dim[2]])
        self.total_loss = tf.losses.mean_squared_error(self.x,self.x_hat)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss)

    def load(self):
        self.build()
        self.sess = tf.InteractiveSession() # Interactive
        self.saver = tf.train.Saver()
        if os.path.isfile(self.model_fname+'.meta'):
            try:
                self.saver.restore(self.sess,self.model_fname)
            except ValueError:
                print('Cannot restore model')
        else:
            print('Model file not found')
            self.sess.run(tf.global_variables_initializer())

    # Execute the forward and the backward pass
    def run_single_step(self, x):
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

class DenseAutoencoder(object):
    def __init__(self, input_dim=[None,32,128,1], learning_rate=1e-3, batch_size=64, n_z=128, model_fname='/home/ronnypetson/models/Dense_AE',var_scope='dense',load=True):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_z = n_z
        self.model_fname = model_fname
        self.var_scope = var_scope
        with tf.name_scope(self.var_scope):
            self.build()
            self.sess = tf.InteractiveSession() # Interactive
            self.saver = tf.train.Saver()
            if load: self.load()

    # Build the network and the loss functions
    def build(self):
        self.x = tf.placeholder(name='x', dtype=tf.float32, shape=self.input_dim)
        # Encode
        # x -> z
        conv1 = tf.layers.conv2d(self.x, 32, (3,3), (1,1), padding='same', activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, 64, (3,3), (1,1), padding='same', activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(conv2, 64, (3,3), (1,1), padding='same', activation=tf.nn.relu)
        flat1 = tf.layers.flatten(conv3)
        self.z = tf.layers.dense(flat1,self.n_z) # ,activation=tf.nn.relu
        # Decode
        # z -> x_hat
        new_dim = [-1,self.input_dim[1],self.input_dim[2],64]
        dec1 = tf.layers.dense(self.z,np.prod(new_dim[1:]),activation=tf.nn.relu) # tf.shape(flat1)
        dec1 = tf.reshape(dec1,new_dim) # tf.shape(conv3)
        dec2 = tf.layers.conv2d_transpose(dec1, 64, (3,3), (1,1), padding='same', activation=tf.nn.relu)
        dec3 = tf.layers.conv2d_transpose(dec2, 64, (3,3), (1,1), padding='same', activation=tf.nn.relu)
        self.x_hat = tf.layers.conv2d_transpose(dec3, 1, (3,3), (1,1), padding='same', activation=None)
        self.total_loss = tf.losses.mean_squared_error(self.x,self.x_hat)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss)

    def load(self):
        if os.path.isfile(self.model_fname+'.meta'):
            try:
                self.saver.restore(self.sess,self.model_fname)
            except ValueError:
                print('Cannot restore model')
        else:
            print('Model file not found')
            self.sess.run(tf.global_variables_initializer())

    # Execute the forward and the backward pass
    def run_single_step(self, x):
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

################################

class VariationalAutoencoder(object):
    def __init__(self, input_dim=[None,64,64,1], learning_rate=3e-4, batch_size=64, n_z=128, model_fname='/home/ronnypetson/models/VAE_pong'):
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
            print('Model file not found')
            self.sess.run(tf.global_variables_initializer())
    # Encode
    # x -> z_mean, z_sigma -> z
    def encoder_(self,x):
        conv1 = tf.layers.conv2d(x, 32, (5,5), (2,2), padding='same', activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, 64, (5,5), (1,1), padding='same', activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(conv2, 64, (3,3), (1,1), padding='same', activation=tf.nn.relu) #
        flat1 = tf.layers.flatten(conv3)
        z_mu = tf.layers.dense(flat1,self.n_z,activation=None) # None 
        z_log_sigma_sq = tf.layers.dense(flat1,self.n_z,activation=None) # None
        eps = tf.random_normal(shape=tf.shape(z_log_sigma_sq),mean=0.0, stddev=0.01, dtype=tf.float32)
        z = z_mu + tf.sqrt(tf.exp(z_log_sigma_sq)) * eps
        return z, z_mu, z_log_sigma_sq
    # Decode
    # z -> x_hat
    def decoder_(self,z):
        new_dim = [-1,self.input_dim[1]//2,self.input_dim[2]//2,64]
        dec1 = tf.layers.dense(z,np.prod(new_dim[1:]),activation=tf.nn.relu) # tf.shape(flat1)
        dec1 = tf.reshape(dec1,new_dim) # tf.shape(conv3)
        dec2 = tf.layers.conv2d_transpose(dec1, 64, (3,3), (1,1), padding='same', activation=tf.nn.relu)
        dec3 = tf.layers.conv2d_transpose(dec2, 64, (5,5), (1,1), padding='same', activation=tf.nn.relu)
        x_hat = tf.layers.conv2d_transpose(dec3, 1, (5,5), (2,2), padding='same', activation=tf.nn.relu) #
        return x_hat
    # Build the netowrk and the loss functions
    def build(self):
        self.x = tf.placeholder(name='x', dtype=tf.float32, shape=self.input_dim)
        self.z, self.z_mu, self.z_log_sigma_sq = self.encoder_(self.x)
        self.x_hat = self.decoder_(self.z)
        # Loss
        # Reconstruction loss
        # Minimize the cross-entropy loss
        # H(x, x_hat) = -\Sigma x*log(x_hat) + (1-x)*log(1-x_hat)
        '''
        epsilon = 1e-10
        recon_loss = -tf.reduce_sum(
            self.x * tf.log(epsilon,self.x_hat) + (1-self.x) * tf.log(epsilon+1-self.x_hat),
            axis=[1,2,3]
        )
        self.recon_loss = tf.reduce_mean(recon_loss)
        '''
        self.recon_loss = tf.losses.mean_squared_error(self.x,self.x_hat)
        #tf.losses.huber_loss(self.x,self.x_hat) #tf.losses.mean_squared_error(self.x,self.x_hat)
        # Latent loss
        # Kullback Leibler divergence: measure the difference between two distributions
        # Here we measure the divergence between the latent distribution and N(0, 1)
        latent_loss = -0.5 * tf.reduce_mean(1.0 + self.z_log_sigma_sq - tf.square(self.z_mu) - tf.exp(self.z_log_sigma_sq), axis=1)
        self.latent_loss = tf.reduce_mean(latent_loss) # tf.distributions.kl_divergence()
        self.total_loss = self.recon_loss + self.latent_loss #tf.reduce_mean(self.recon_loss + self.latent_loss)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss)
        return
    # Execute the forward and the backward pass
    def run_single_step(self, x):
        _, loss, recon_loss, latent_loss, z_mu = self.sess.run(
            [self.train_op, self.total_loss, self.recon_loss, self.latent_loss, self.z_mu],
            feed_dict={self.x: x}
        )
        return loss, recon_loss, latent_loss, z_mu
    # z -> x
    def generator(self, z):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.z: z})
        return x_hat
    # x -> z
    def transformer(self, x):
        z = self.sess.run(self.z, feed_dict={self.x: x})
        return z
        # x -> x_hat
    def reconstructor(self, x):
        #x_hat = self.sess.run(self.x_hat, feed_dict={self.x: x})
        z_mu = self.sess.run(self.z_mu, feed_dict={self.x: x})
        x_hat = self.generator(z_mu)
        return x_hat
    def save_model(self):
        self.saver.save(self.sess, self.model_fname)
    def close_session(self):
        self.sess.close()

class ConvAutoencoder(object):
    def __init__(self, input_dim=[None,32,32,1], learning_rate=1e-3, batch_size=64, z_shape=[64,1], model_fname='/home/ronnypetson/models/Conv_AE'):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.z_shape = z_shape
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
            print('Model file not found')
            self.sess.run(tf.global_variables_initializer())

    # Build the netowrk and the loss functions
    def build(self):
        self.x = tf.placeholder(name='x', dtype=tf.float32, shape=self.input_dim)
        # Encode
        # x -> z_mean, z_sigma -> z
        conv1 = tf.layers.conv2d(self.x, 64, (5,5), (2,2), padding='same', activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, 64, (5,5), (2,2), padding='same', activation=tf.nn.relu)
        self.z_ = tf.layers.conv2d(conv2, 1, (3,3), (1,1), padding='same', activation=None) # tf.nn.relu
        #self.z_ = tf.image.resize_bilinear(conv3, (16,16))
        self.z = tf.reshape(self.z_,[-1,np.prod([d//4 for d in self.input_dim[1:-1]]),1])
        # Decode
        # z -> x_hat
        #dec1 = tf.image.resize_bilinear(self.z_, [dim//4 for dim in self.input_dim[1:-1]])
        dec1 = tf.layers.conv2d_transpose(self.z_, 64, (3,3), (1,1), padding='same', activation=tf.nn.relu)
        dec2 = tf.layers.conv2d_transpose(dec1, 64, (5,5), (2,2), padding='same', activation=tf.nn.relu)
        self.x_hat = tf.layers.conv2d_transpose(dec2, self.input_dim[3], (5,5), (2,2), padding='same', activation=None) # None
        self.total_loss = tf.losses.mean_squared_error(self.x,self.x_hat)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss)

    # Execute the forward and the backward pass
    def run_single_step(self, x):
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
        #x_hat = self.sess.run(self.x_hat, feed_dict={self.z: z})
        x_hat = self.sess.run(self.x_hat, feed_dict={self.z_: z.reshape([-1,self.input_dim[1]//4,self.input_dim[2]//4,1])})
        return x_hat
    def generator_(self, z_):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.z_: z_})
        return x_hat
    # x -> z
    def transformer(self, x):
        z = self.sess.run(self.z, feed_dict={self.x: x})
        return z
    def save_model(self):
        self.saver.save(self.sess, self.model_fname)
    def close_session(self):
        self.sess.close()

class Conv3DAutoencoder(object):
    def __init__(self, input_dim=[None,1,32,32,1], learning_rate=1e-3, batch_size=64, model_fname='/home/ronnypetson/models/Conv3D_AE',load=True):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model_fname = model_fname
        self.build()
        if load:
            self.load()

    # Build the netowrk and the loss functions
    def build(self):
        self.x = tf.placeholder(name='x', dtype=tf.float32, shape=self.input_dim)
        # Encode
        d = min(self.input_dim[1],2)
        h = min(self.input_dim[2]//8,2)
        w = min(self.input_dim[3]//8,2)
        fd = lambda x: min(x,self.input_dim[1])
        # x -> z_mean, z_sigma -> z
        conv1 = tf.layers.conv3d(self.x, 64, (fd(5),5,5), (d,h,w), padding='same', activation=tf.nn.relu)
        conv2 = tf.layers.conv3d(conv1, 64, (fd(3),3,3), (d,h,w), padding='same', activation=tf.nn.relu)
        self.z = tf.layers.conv3d(conv2, 1, (fd(3),3,3), (1,1,1), padding='same', activation=None) # tf.nn.relu
        # Decode
        # z -> x_hat
        dec1 = tf.layers.conv3d_transpose(self.z, 64, (fd(3),3,3), (1,1,1), padding='same', activation=tf.nn.relu)
        dec2 = tf.layers.conv3d_transpose(dec1, 64, (fd(3),3,3), (d,h,w), padding='same', activation=tf.nn.relu)
        self.x_hat = tf.layers.conv3d_transpose(dec2, self.input_dim[-1], (fd(5),5,5), (d,h,w), padding='same', activation=None) # None
        self.total_loss = tf.losses.mean_squared_error(self.x,self.x_hat)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss)

    def load(self):
        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver()
        if os.path.isfile(self.model_fname+'.meta'):
            try:
                self.saver.restore(self.sess,self.model_fname)
            except ValueError:
                print('Cannot restore model')
        else:
            print('Model file not found')
            self.sess.run(tf.global_variables_initializer())

    # Execute the forward and the backward pass
    def run_single_step(self, x):
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

