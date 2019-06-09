import os
import sys
import re
import cv2
import h5py
import numpy as np
import torch
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/topos')
sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/preprocess')
sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/misc')
sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/sample')

from glob import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pt_ae import DirectOdometry, VanillaAutoencoder, MLPAutoencoder
from datetime import datetime
from plotter import c3dto2d, abs2relative, plot_eval

def my_collate(batch):
    batch_x = []
    batch_y = []
    batch_abs = []
    for b in batch:
        if b is not None:
            batch_x.append(b[0])
            batch_y.append(b[1])
            batch_abs.append(b[2])
    return torch.stack(batch_x), torch.stack(batch_y), batch_abs

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size*h/w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size*w/h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h),int(new_w)
        image = cv2.resize(image,(new_h,new_w))
        return image

class FluxRescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size*h/w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size*w/h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h),int(new_w)
        image0 = cv2.resize(image[:,:,0],(new_h,new_w))
        image1 = cv2.resize(image[:,:,1],(new_h,new_w))
        image = np.stack([image0,image1],axis=2)
        return image

class ToTensor(object):
    def __call__(self,image):
        return torch.from_numpy(image).unsqueeze(0).float()

class H5SeqDataset(Dataset):
    def __init__(self, file_path, seq_len, chunk_size, transform=None):
        super().__init__()
        h5_file = h5py.File(file_path)
        self.frames = h5_file.get('frames')
        self.poses = h5_file.get('poses')
        self.sid_len = h5_file.get('sid_len')
        self.seq_len = seq_len
        self.transform = transform
        self.chunk_size = chunk_size

    def __getitem__(self, index):
        i,j = index//self.chunk_size, index%self.chunk_size
        index = max(2,index)
        if self.sid_len[i][j][0] + self.seq_len >= self.sid_len[i][j][1]\
           or index + self.seq_len >= self.__len__():
            index = index - self.seq_len - 1
        try:
            x = []
            for i in range(index,index+self.seq_len):
                frame = self.frames[i//self.chunk_size][i%self.chunk_size]
                if self.transform:
                    frame = self.transform(frame)
                x.append(frame)
            x = torch.cat(x,dim=0).unsqueeze(1)
            y, abs = [], []
            for i in range(index,index+self.seq_len):
                p = self.poses[i//self.chunk_size][i%self.chunk_size]
                p = c3dto2d(p)
                y.append(p)
                abs.append(p)
            y = abs2relative(y,self.seq_len,1)[0]
            y = torch.from_numpy(y).float()
            return x, y, abs
        except Exception as e:
            print(e)

    def __len__(self):
        return self.chunk_size*self.frames.shape[0]

if __name__=='__main__':
    train_dir = sys.argv[1] #'/home/ronnypetson/Documents/deep_odometry/kitti/joint_frames_odom_train.h5'
    valid_dir = sys.argv[2] #'/home/ronnypetson/Documents/deep_odometry/kitti/joint_frames_odom_valid.h5'
    test_dir = sys.argv[3] #'/home/ronnypetson/Documents/deep_odometry/kitti/joint_frames_odom_test.h5'

    '''train_dir = sorted([fn for fn in glob(train_dir) if os.path.isfile(fn)])
    valid_dir = sorted([fn for fn in glob(valid_dir) if os.path.isfile(fn)])
    test_dir = sorted([fn for fn in glob(test_dir) if os.path.isfile(fn)])'''

    model_fn = sys.argv[4]
    h_dim = int(sys.argv[5])
    log_folder = sys.argv[6]
    new_dim = (int(sys.argv[7]),int(sys.argv[8]))
    batch_size = int(sys.argv[9])
    num_epochs = int(sys.argv[10])
    seq_len = 16
    transf = transforms.Compose([Rescale(new_dim),ToTensor()])
    ##transf = [Rescale(new_dim),ToTensor()] #,FluxToTensor()]

    train_dataset = H5SeqDataset(train_dir,seq_len,10,transf)
    valid_dataset = H5SeqDataset(valid_dir,seq_len,10,transf)
    test_dataset = H5SeqDataset(test_dir,seq_len,10,transf)
    ##train_dataset = FluxH5Dataset(train_dir,10,transf)
    ##valid_dataset = FluxH5Dataset(valid_dir,10,transf)
    ##test_dataset = FluxH5Dataset(test_dir,10,transf)

    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=0,collate_fn=my_collate)
    valid_loader = DataLoader(valid_dataset,batch_size=batch_size//2,shuffle=False,num_workers=0,collate_fn=my_collate)
    test_loader = DataLoader(test_dataset,batch_size=batch_size//2,shuffle=False,num_workers=0,collate_fn=my_collate)

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(device)
    ##model = VanillaAutoencoder((1,)+new_dim).to(device)
    #model = VanillaAutoencoder((2,)+new_dim,h_dim).to(device)
    #model = MLPAutoencoder((2,)+new_dim,h_dim).to(device)
    model = DirectOdometry((1,)+new_dim,(12,),h_dim).to(device)
    params = model.parameters()
    optimizer = optim.Adam(params,lr=3e-4)
    min_loss = 1e15
    epoch = 0

    if os.path.isfile(model_fn):
        print('Loading existing model')
        checkpoint = torch.load(model_fn)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        min_loss = checkpoint['min_loss']
        epoch = checkpoint['epoch']
    else:
        print('Creating new model')

    loss_fn = torch.nn.MSELoss()
    epoch_losses = []
    epoch = num_epochs-1
    for i in range(epoch,num_epochs):
        model.train()
        losses = []
        for j,xy in enumerate(test_loader):
            x,y = xy[0].to(device), xy[1].to(device)
            optimizer.zero_grad()
            y_ = model(x)
            loss = loss_fn(y_,y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            print('Batch {}\tloss: {}'.format(j,loss.item()))
        model.eval()
        v_losses = []
        for xy in test_loader:
            x,y = xy[0].to(device), xy[1].to(device)
            y_ = model(x)
            loss = loss_fn(y_,y)
            v_losses.append(loss.item())
        mean_train, mean_valid = np.mean(losses),np.mean(v_losses)
        epoch_losses.append([i,mean_train,mean_valid])
        print('Epoch {}\tloss\t{:.3f}\tValid loss\t{:.3f}'.format(i,mean_train,mean_valid)) # Mean train loss, mean validation loss
        if mean_valid < min_loss:
            min_loss = mean_valid
            torch.save({'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'min_loss': min_loss,
                        'epoch': i+1}, model_fn)
    model.eval()
    print('Start of plot_eval')
    plot_eval(model,test_loader,seq_len,device)
    print('End of plot_eval')
    '''t_losses = []
    for xy in test_loader:
        x,y = xy[0].to(device), xy[1].to(device)
        #print(x.size(),y.size())
        y_ = model(x)
        loss = loss_fn(y_,y)
        t_losses.append(loss.item())
    mean_test = np.mean(t_losses)
    epoch_losses.append([-1,0.0,mean_test])
    print('Test loss:',np.mean(mean_test))
    # Save training log
    np.save('{}/{}_log.npy'.format(log_folder,datetime.now()),epoch_losses)'''
