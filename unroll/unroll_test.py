import numpy as np
import tensorflow as tf
import cv2
import os
from matplotlib import pyplot as plt

batch_size = 12
learning_rate = 0.001
sequence_len = 1 # Odd number
img_h = 65 #121 #65
img_w = 81 #161 #49
rs_dir = '/home/ronnypetson/Downloads/CVPR_Dataset/house_rot1_B0/rolling_shutter/'
gs_dir = '/home/ronnypetson/Downloads/CVPR_Dataset/house_rot1_B0/gt_end/'

def load_data(rs=rs_dir, gs=gs_dir):
	rs_files = os.listdir(rs)
	gs_files = os.listdir(gs)

	rs_imgs = []
	gs_imgs = []
	for rsf, gsf in zip(rs_files, gs_files):
		rs_img = cv2.imread(rs_dir+rsf,0)
		#rs_img = cv2.Canny(rs_img,150,200)
		rs_img = cv2.resize(rs_img,(img_h,img_w))
		rs_imgs.append(rs_img)

		gs_img = cv2.imread(gs_dir+gsf,0)
		#gs_img = cv2.Canny(gs_img,150,200)
		gs_img = cv2.resize(gs_img,(img_h,img_w))
		gs_imgs.append(np.reshape(gs_img,(img_h,img_w,1)))

	return rs_imgs, gs_imgs

def get_batch(rs_imgs, gs_imgs):
	num_imgs = len(rs_imgs)
	inds = np.random.choice(range(num_imgs),batch_size,replace=False)
	rs_batch = []
	for i in range(num_imgs):
		seq = []
		for j in range(-int(sequence_len/2),int(sequence_len/2)+1):
			k = min(num_imgs-1,max(0,i+j))
			seq.append(rs_imgs[k])
		rs_batch.append(seq)
	rs_batch = np.array(rs_batch).transpose((0,3,2,1))
	#print(rs_batch.shape)
	return rs_batch, np.array([gs_imgs[ind] for ind in inds])

X = tf.placeholder(tf.float32,[None,img_h,img_w,sequence_len])
Y = tf.placeholder(tf.float32,[None,img_h,img_w,1])

conv1 = tf.layers.conv2d(X,32,(3,3),padding='same',activation=tf.nn.relu)				# [32,32,4] -> [28,28,64]
pool1 = tf.layers.max_pooling2d(conv1,(3,3),strides=(2,2),padding='same')				# [28,28,64] -> [12,12,64]
conv2 = tf.layers.conv2d(pool1,64,(3,3),padding='same',activation=tf.nn.relu)			# [12,12,64] -> [5,5,128]
pool2 = tf.layers.max_pooling2d(conv2,(3,3),strides=(2,2),padding='same')
conv3 = tf.layers.conv2d(pool2,64,(3,3),padding='same',activation=tf.nn.relu)
pool3 = tf.layers.max_pooling2d(conv3,(3,3),strides=(2,2),padding='same')

deconv1 = tf.layers.conv2d_transpose(pool3,64,(3,3),padding='same',activation=tf.nn.relu)	# [5,5,128] -> [11,11,64]
upsamp1 = tf.keras.layers.UpSampling2D(deconv1,(3,3),strides=(2,2),padding='same')
deconv2 = tf.layers.conv2d_transpose(upsamp1,32,(5,5),padding='same',activation=tf.nn.relu)# [11,11,64] -> [25,25,64]
upsamp2 = tf.keras.layers.UpSampling2D(deconv2,(3,3),strides=(2,2),padding='same')
deconv3 = tf.layers.conv2d_transpose(upsamp2,1,(3,3),padding='same',activation=tf.nn.relu) # [25,25,64] -> [29,29,1]

loss = tf.losses.mean_squared_error(Y,deconv3)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

rs,gs = load_data()

for x,y in zip(rs,gs):
	plt.imshow(x)
	plt.show()
	plt.imshow(y.reshape((img_h,img_w)))
	plt.show()
exit()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for t in range(10000):
		b_x, b_y = get_batch(rs,gs)
		loss_, train_ = sess.run([loss,train], feed_dict={X:b_x,Y:b_y})
		print(loss_)
	b_x, _ = get_batch(rs,gs)
	pred_gs = sess.run(deconv3, feed_dict={X:b_x})
	for i in range( len( pred_gs)):
		cv2.imwrite('pred_gs/pred_gs_{}.png'.format(i),pred_gs[i])

