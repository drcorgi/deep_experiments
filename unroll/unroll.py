import numpy as np
#import tensorflow as tf
import cv2
from matplotlib import pyplot as plt

train_dir = 'train/'
test_dir = 'test/'

img = cv2.imread(train_dir+'bogdangrad.png',0)
rows,cols = img.shape
center = rows//2,cols//2
dst = np.zeros((rows,cols))

def gen_wave_params(ranges=[100,30,10]):
	freq = np.random.randint(ranges[0])+10
	phase = np.random.randint(ranges[1])
	amp = np.random.rand()*ranges[2]
	return freq, phase, amp

# x wave parameters
freq_x, phase_x, amp_x = gen_wave_params()
# y wave parameters
freq_y, phase_y, amp_y = gen_wave_params()
# theta wave parameters
#freq_theta, phase_theta, amp_theta = gen_wave_params([15,30,0.0])

for i in range(rows):
	tx = amp_x*np.sin((i-phase_x)/freq_x)
	ty = amp_y*np.sin((i-phase_y)/freq_y)
	#y_skew = 1.0+0.15*np.random.rand()
	#theta = amp_theta*np.sin((i-phase_theta)/freq_theta)
	for j in range(cols):
		#ti = i-ty #-center[0]
		#tj = j-tx #-center[1]
		#j_ = np.cos(theta)*tj+np.sin(theta)*ti
		#i_ = -np.sin(theta)*tj+np.cos(theta)*ti
		#i_ = int(i_)+center[0]
		#j_ = int(j_)+center[1]
		i_ = int(i-ty)
		j_ = int(j-tx)
		if(i_ >= 0 and j_ >= 0 and i_ < rows and j_ < cols):
			dst[i][j] = img[i_][j_]

plt.figure(1)
plt.imshow(img,cmap='gray')
plt.figure(2)
plt.imshow(dst,cmap='gray')
plt.title('[fx,px,ax] = '+str([freq_x,phase_x,int(amp_x)])+'; [fy,py,ay] = '+str([freq_y,phase_y,int(amp_y)]))
plt.show()

# DY077344532BR

