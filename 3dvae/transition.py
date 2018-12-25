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
        self.x_hat = tf.layers.dense(d1, self.output_dim[1], activation=tf.nn.relu)
        # Loss and train operations
        self.total_loss = tf.losses.mean_squared_error(self.x_,self.x_hat)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss)
        return

    # Execute the forward and the backward pass
    def run_single_step(self, x, x_, save=False):
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

