import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

batch_size = 16
learning_rate = 0.001
sequence_len = 3 # Odd number
img_h = 224 #121 #65
img_w = 224 #161 #49
rs_dir = '/home/ronnypetson/Documents/penncosyvio/data/tango_bottom/af/frames/' #'~/Downloads/CVPR_Dataset/house_rot1_B0/rolling_shutter/'
gs_dir = '/home/ronnypetson/Documents/penncosyvio/data/tango_top/af/frames/' #'~/Downloads/CVPR_Dataset/house_rot1_B0/gt_end/'
model_fn = 'rs2gs_resnet.h5'

def fn2int(fn):
    return int(fn[:-4])

def load_data(rs=rs_dir, gs=gs_dir):
	rs_files = os.listdir(rs)
	rs_files.sort(key=fn2int)
	gs_files = os.listdir(gs)
	gs_files.sort(key=fn2int)

	model = ResNet50(weights='imagenet')
	model = Model(inputs=model.input, outputs=model.get_layer('activation_49').output)	# None,7,7,2048, avg_pool, activation_49

	rs_imgs = []
	gs_imgs = []
	test_rs = []
	i = 0
	for rsf, gsf in zip(rs_files, gs_files):
		rs_img = cv2.imread(rs_dir+rsf,0)
		rs_img = cv2.resize(rs_img,(img_h,img_w))
		rs_img = preprocess_input(rs_img)
		
		gs_img = cv2.imread(gs_dir+gsf,0)
		gs_img = cv2.resize(gs_img,(440,440))	# 440,440
		gs_img = preprocess_input(gs_img)/255.0
		
		if i < 300:
			rs_imgs.append(rs_img)
			gs_imgs.append(np.reshape(gs_img,(440,440,1)))
		else:
			test_rs.append(rs_img)
		if i == 312:
			break
		i += 1

	num_imgs = len(rs_imgs)
	rs_data = []
	for i in range(num_imgs):
		seq = []
		for j in range(-int(sequence_len/2),int(sequence_len/2)+1):
			k = min(num_imgs-1,max(0,i+j))
			seq.append(rs_imgs[k])
		rs_data.append(seq)

	num_imgs = len(test_rs)
	test_rs_ = []
	for i in range(num_imgs):
		seq = []
		for j in range(-int(sequence_len/2),int(sequence_len/2)+1):
			k = min(num_imgs-1,max(0,i+j))
			seq.append(test_rs[k])
		test_rs_.append(seq)

	rs_data = np.array(rs_data).transpose((0,2,3,1))
	test_rs_ = np.array(test_rs_).transpose((0,2,3,1))
	rs_features = model.predict(rs_data)
	test_features = model.predict(test_rs_)
	del model

	return rs_features, np.array(gs_imgs), test_features, np.array(test_rs)

input_seq = Input(shape=(7, 7, 2048))  # adapt this if using `channels_first` image data format

x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_seq)
x = UpSampling2D((4, 4))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((4, 4))(x)
x = Conv2D(32, (3, 3), activation='relu')(x)
x = UpSampling2D((4, 4))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# Load or create model
autoencoder = None
if os.path.isfile(model_fn):
	autoencoder = load_model(model_fn)
else:
	autoencoder = Model(input_seq, decoded)
	autoencoder.compile(optimizer='adam', loss='binary_crossentropy') # adadelta

x_train, y_train, x_test_f, x_test = load_data()

'''
autoencoder.fit(x_train, y_train,\
                epochs=100,\
                batch_size=batch_size,\
                shuffle=True,\
                validation_data=(x_train, y_train),\
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
'''

decoded_imgs = autoencoder.predict(x_test_f[:12])

n = 12
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(224, 224)) # y_train
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow(decoded_imgs[i].reshape(440, 440))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

autoencoder.save(model_fn)

