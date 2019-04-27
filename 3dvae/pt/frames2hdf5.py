import h5py
import numpy as np

def gen_int():
    i = 0
    while i < 5:
        yield i
        i += 1

'''h5f = h5py.File('data.h5', 'w')
h5f.create_dataset('dataset_1',data=np.array([1,2,3]))
h5f.close()'''

h5f = h5py.File('data.h5','a')
print(h5f)
b = h5f['dataset_1'][:]
b = b.mean()
print(b)

for i in range(5):
    h5f['dataset_1'] = h5f['dataset_1'].concatenate([b,np.array([4,5,6])])

print(h5f)
h5f.close()
