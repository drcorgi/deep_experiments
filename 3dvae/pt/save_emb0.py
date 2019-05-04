import os
import sys
import re
import cv2
import h5py
import pickle
import numpy as np
import torch
import torch.optim as optim

from glob import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pt_ae import VanillaAutoencoder
from datetime import datetime
from odom_dataset import my_collate, Rescale, ToTensor, H5Dataset, FramesDataset, FluxToTensor, FluxH5Dataset
from odom_loader import load_kitti_odom

if __name__=='__main__':
    train_dir = sys.argv[1] #'/home/ronnypetson/Documents/deep_odometry/kitti/frames_odom_train.h5'
    valid_dir = sys.argv[2] #'/home/ronnypetson/Documents/deep_odometry/kitti/frames_odom_valid.h5'
    test_dir = sys.argv[3] #'/home/ronnypetson/Documents/deep_odometry/kitti/frames_odom_test.h5'

    emb_fn = sys.argv[4] #'/home/ronnypetson/Documents/deep_odometry/kitti/frames_emb0.pck'
    meta_fn = sys.argv[5] #'visual_odometry_database.meta'
    model_fn = sys.argv[6] #'/home/ronnypetson/models/pt/model.pth'
    new_dim = (int(sys.argv[7]),int(sys.argv[8]))
    batch_size = int(sys.argv[9])
    ##transf = transforms.Compose([Rescale(new_dim),ToTensor()])
    transf = [Rescale(new_dim),FluxToTensor()]

    ''' Metadados da base de Odometria visual
        Tipo: lista de dicionários ('sub_base' 'sequence' 'sid_frame' 'frame_fn' 'odom_fn')
    '''
    with open(meta_fn,'rb') as f:
        meta = pickle.load(f)

    ##train_dataset = H5Dataset(train_dir,10,transf)
    ##valid_dataset = H5Dataset(valid_dir,10,transf)
    ##test_dataset = H5Dataset(test_dir,10,transf)
    valid_dataset = FluxH5Dataset(valid_dir,10,transf)
    test_dataset = FluxH5Dataset(test_dir,10,transf)
    train_dataset = FluxH5Dataset(train_dir,10,transf)

    valid_loader = DataLoader(valid_dataset,batch_size=batch_size,shuffle=False,num_workers=0,collate_fn=my_collate)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=0,collate_fn=my_collate)
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=False,num_workers=0,collate_fn=my_collate)

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(device)
    ##model = VanillaAutoencoder((1,)+new_dim).to(device)
    model = VanillaAutoencoder((2,)+new_dim).to(device)

    if os.path.isfile(model_fn):
        print('Loading existing model')
        checkpoint = torch.load(model_fn)
        model.load_state_dict(checkpoint['model_state'])
    else:
        print('Model not found')
        exit()

    model.eval()
    all_enc = []
    for loader in [valid_loader,test_loader,train_loader]:
        for x in loader:
            x = x.to(device)
            z = model.forward_z(x).detach().cpu().numpy()
            all_enc.append(z)
    all_enc = np.concatenate(all_enc,axis=0)

    # Works for KITTI
    all_seq = [[],[],[],[],[],[],[],[],[],[],[]]
    for i in range(len(all_enc)):
        s = int(meta[i]['sequence'])
        all_seq[s].append(all_enc[i])

    with open(emb_fn,'wb') as f:
        pickle.dump(all_seq,f)
