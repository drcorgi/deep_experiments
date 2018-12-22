import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model
from keras import backend as K
from keras.callbacks import TensorBoard

batch_size = 12
learning_rate = 0.001
sequence_len = 3 # Odd number
img_h = 252 #121 #65
img_w = 252 #161 #49
rs_dir = '/home/ronnypetson/Downloads/CVPR_Dataset/house_rot1_B0/rolling_shutter/'
gs_dir = '/home/ronnypetson/Downloads/CVPR_Dataset/house_rot1_B0/gt_end/'
model_fn = 'rs2gs_252x252.h5'

def load_data(rs=rs_dir, gs=gs_dir):
	rs_files = os.listdir(rs)
	gs_files = os.listdir(gs)

	rs_imgs = []
	gs_imgs = []
	for rsf, gsf in zip(rs_files, gs_files):
		rs_img = cv2.imread(rs_dir+rsf,0)
		rs_img = cv2.resize(rs_img,(img_h,img_w))
		rs_imgs.append(rs_img/255.0)

		gs_img = cv2.imread(gs_dir+gsf,0)
		gs_img = cv2.resize(gs_img,(img_h,img_w))/255.0
		gs_imgs.append(np.reshape(gs_img,(img_h,img_w,1)))

	num_imgs = len(rs_imgs)
	rs_data = []
	for i in range(num_imgs):
		seq = []
		for j in range(-int(sequence_len/2),int(sequence_len/2)+1):
			k = min(num_imgs-1,max(0,i+j))
			seq.append(rs_imgs[k])
		rs_data.append(seq)
	rs_data = np.array(rs_data).transpose((0,2,3,1))

	return rs_data, np.array(gs_imgs)

'''
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

	return rs_batch, np.array([gs_imgs[ind] for ind in inds])
'''

input_seq = Input(shape=(img_h, img_w, sequence_len))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_seq)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# Load or create model
autoencoder = None
if os.path.isfile(model_fn):
	autoencoder = load_model(model_fn)
else:
	autoencoder = Model(input_seq, decoded)
	autoencoder.compile(optimizer='adam', loss='binary_crossentropy') # adadelta

x_train, y_train = load_data()

autoencoder.fit(x_train, y_train,\
                epochs=10000,\
                batch_size=batch_size,\
                shuffle=True,\
                validation_data=(x_train, y_train),\
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

decoded_imgs = autoencoder.predict(x_train)

n = 12
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_train[i,:,:,1].reshape(img_h, img_w))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow(decoded_imgs[i].reshape(img_h, img_w))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

autoencoder.save(model_fn)

