import os
import numpy as np
import tensorflow as tf

class Transition(object):
    def __init__(self, input_dim=[None,256], output_dim=[None,256], learning_rate=1e-3, batch_size=64, model_fname='/home/ronnypetson/models/Vanilla_transition'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
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
        # Placeholders
        self.x = tf.placeholder(name='x', dtype=tf.float32, shape=self.input_dim)
        self.x_ = tf.placeholder(name='x_', dtype=tf.float32, shape=self.output_dim)
        # Layers
        d1 = tf.layers.dense(self.x, 512, activation=tf.nn.relu)
        self.x_hat = tf.layers.dense(d1, self.output_dim[1], activation=None) # tf.nn.relu
        # Loss and train operations
        self.total_loss = tf.losses.mean_squared_error(self.x_,self.x_hat)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss)
        return

    # Execute the forward and the backward pass
    def run_single_step(self, x, x_):
        _, loss = self.sess.run(
            [self.train_op, self.total_loss],
            feed_dict={self.x: x, self.x_: x_}
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

class TransitionWGAN(object):
    def __init__(self, input_dim=[None,256], output_dim=[None,2*256], learning_rate=1e-3, batch_size=64, model_fname='/home/ronnypetson/models/Transition_WGAN'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
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
        # Generator
        def generator_(x):
            with tf.variable_scope('generator'):
                d1 = tf.layers.dense(x, 512, activation=tf.nn.relu)
                x_hat = tf.layers.dense(d1, self.input_dim[1], activation=tf.nn.sigmoid) # tf.nn.relu
            return x_hat
        # Discriminator
        def discriminator_(x,reuse):
            with tf.variable_scope('discriminator',reuse=reuse):
                d2 = tf.layers.dense(x, 128, activation=tf.nn.relu)
                d2 = tf.layers.dense(x, 32, activation=tf.nn.relu)
                d = tf.layers.dense(d2, 1, activation=None)
            return d
        # Placeholders
        with tf.variable_scope('placeholders'):
            self.x = tf.placeholder(tf.float32, self.input_dim) # add noise
            self.x_ = tf.placeholder(tf.float32, self.input_dim)
        self.x_hat = generator_(self.x)
        self.d_true = discriminator_(tf.concat([self.x_,self.x],1),reuse=False)
        self.d_generated = discriminator_(tf.concat([self.x_hat,self.x],1),reuse=True)
        # Loss and train operations
        with tf.name_scope('regularizer'):
            epsilon = tf.random_uniform([self.batch_size, 1], 0.0, 1.0)
            x_hat_ = epsilon * self.x_ + (1 - epsilon) * self.x_hat
            d_hat_ = discriminator_(tf.concat([x_hat_,self.x],1), reuse=True)
            gradients = tf.gradients(d_hat_, x_hat_)[0]
            ddx = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1]))
            d_regularizer = tf.reduce_mean((ddx - 1.0) ** 2)
        with tf.name_scope('loss'):
            self.g_loss = tf.reduce_mean(self.d_generated)
            self.d_loss = (tf.reduce_mean(self.d_true) - tf.reduce_mean(self.d_generated) + 10 * d_regularizer)
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0, beta2=0.9)
            g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
            self.g_train = optimizer.minimize(self.g_loss, var_list=g_vars)
            d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
            self.d_train = optimizer.minimize(self.d_loss, var_list=d_vars)
    # Execute the forward and the backward pass
    def run_single_step(self, x, x_):
        g_loss, _ = self.sess.run([self.g_loss,self.g_train], feed_dict={self.x: x})
        for j in range(5):
            d_loss, _ = self.sess.run([self.d_loss,self.d_train], feed_dict={self.x_: x_, self.x: x})
        return g_loss, d_loss
    # x -> x_hat
    def forward(self, x):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.x: x})
        return x_hat
    def save_model(self):
        self.saver.save(self.sess, self.model_fname)
    def close_session(self):
        self.sess.close()

