# Basic libraries
import os
import numpy as np
import tensorflow as tf

tf.reset_default_graph()
tf.set_random_seed(2016)
np.random.seed(2016)

# LSTM-autoencoder
from LSTMAutoencoder import *
from synbedding import *

# Constants
batch_num = 128
hidden_num = 256
step_num = max_seq #32 #8
elem_num = 8 #1
iteration = 100000
model_fn = 'lstm_encoder'

# placeholder list
p_input = tf.placeholder(tf.float32, shape=(batch_num, step_num, elem_num))
p_inputs = [tf.squeeze(t, [1]) for t in tf.split(p_input, step_num, 1)]

cell = tf.nn.rnn_cell.LSTMCell(hidden_num, use_peepholes=True)
ae = LSTMAutoencoder(hidden_num, p_inputs, cell=cell, decode_without_input=True) # True

with tf.Session() as sess:

    saver = tf.train.Saver()
    if os.path.isfile(model_fn+'.meta'):
        print('\nModel found')
        saver.restore(sess,model_fn) # session
    else:
        print('\nModel not found')
        sess.run(tf.global_variables_initializer())

    for i in range(iteration):
        tree_seq = get_batch(batch_num)
        (loss_val, _) = sess.run([ae.loss, ae.train], {p_input: tree_seq})
        if i%200 == 0:
            print('iter %d:' % (i + 1), loss_val)
            test_input = get_batch(batch_num)
            (output_,) = sess.run([ae.output_], {p_input: test_input})
            in_tree = onehot2string(test_input[0])
            out_tree = onehot2string(output_[0, :, :])
            print('Input:\t%s\nOutput:\t%s'%(in_tree,out_tree))
        if i%4000 == 1:
            saver.save(sess,'./'+model_fn)

