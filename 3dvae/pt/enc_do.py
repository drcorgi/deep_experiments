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
from pt_ae import DirectOdometry, FastDirectOdometry,\
VanillaAutoencoder, MLPAutoencoder, VanAE, Conv1dMapper, seq_pose_loss
from datetime import datetime
from plotter import c3dto2d, abs2relative, plot_eval
from odom_loader import load_kitti_odom
from tensorboardX import SummaryWriter
from time import time

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
        image = cv2.resize(image,self.output_size)
        return image

class ToTensor(object):
    def __call__(self,x):
        return torch.from_numpy(x).float()

def list_split_kitti():
    base = '/home/ubuntu/kitti/dataset/'
    all_seqs = [sorted(glob(base+'sequences/{:02d}/image_0/*.png'\
                .format(i))) for i in range(11)]
    all_poses = [base+'poses/{:02d}.txt'.format(i) for i in range(11)]
    train_seqs, train_poses = all_seqs[2:], all_poses[2:]
    valid_seqs, valid_poses = all_seqs[0:1], all_poses[0:1]
    test_seqs, test_poses = all_seqs[1:2], all_poses[1:2]
    return (train_seqs,train_poses), (valid_seqs,valid_poses), (test_seqs,test_poses)

def list_split_kitti_(h,w):
    base = '/home/ubuntu/kitti/{}x{}/'.format(h,w)
    pbase = '/home/ubuntu/kitti/dataset/'
    all_seqs = [sorted(glob(base+'{:02d}/*.png'\
                .format(i))) for i in range(11)]
    all_poses = [pbase+'poses/{:02d}.txt'.format(i) for i in range(11)]
    train_seqs, train_poses = all_seqs[2:], all_poses[2:]
    valid_seqs, valid_poses = all_seqs[0:1], all_poses[0:1]
    test_seqs, test_poses = all_seqs[1:2], all_poses[1:2]
    return (train_seqs,train_poses), (valid_seqs,valid_poses), (test_seqs,test_poses)

def list_split_kitti_flux(h,w):
    base = '/home/ubuntu/kitti/flux/{}x{}/'.format(h,w)
    pbase = '/home/ubuntu/kitti/dataset/'
    all_seqs = [sorted(glob(base+'{:02d}/*.npy'\
                .format(i))) for i in range(11)]
    all_poses = [pbase+'poses/{:02d}.txt'.format(i) for i in range(11)]
    train_seqs, train_poses = all_seqs[2:], all_poses[2:]
    valid_seqs, valid_poses = all_seqs[0:1], all_poses[0:1]
    test_seqs, test_poses = all_seqs[1:2], all_poses[1:2]
    return (train_seqs,train_poses), (valid_seqs,valid_poses), (test_seqs,test_poses)

class FluxSeqDataset(Dataset):
    def __init__(self, fnames, pfnames, seq_len, transform=None):
        ''' fnames is a list of lists of file names
            pfames is a list of file names (one for each entire sequence)
        '''
        super().__init__()
        self.fnames = fnames
        self.pfnames = pfnames
        self.len = sum([max(0,len(fns)-seq_len+1) for fns in fnames])
        self.sids = []
        for i,fns in enumerate(fnames):
            for j in range(len(fns)-seq_len+1):
                self.sids.append(i)
        self.fsids = []
        for fns in fnames:
            for i in range(len(fns)-seq_len+1):
                self.fsids.append(i)
        self.seq_len = seq_len
        self.transform = transform # Transform at the frame level

    def __getitem__(self, index):
        try:
            s,id = self.sids[index], self.fsids[index]
            x = [np.load(fn) for fn in self.fnames[s][id:id+self.seq_len]]
            if self.transform:
                x = [self.transform(img) for img in x]
            x = [img.transpose(0,-1).unsqueeze(0) for img in x]
            x = torch.cat(x,dim=0)
            abs = load_kitti_odom(self.pfnames[s])[id:id+self.seq_len]
            y = []
            for p in abs:
                p = c3dto2d(p)
                y.append(p)
            y = abs2relative(y,self.seq_len,1)[0]
            y = torch.from_numpy(y).float()
            #print('seq loading',x.size(),y.size(),abs.shape)
            return x,y,abs
        except RuntimeError as re:
            print(re)
        except Exception as e:
            print(e)

    def __len__(self):
        return self.len

class SeqDataset(Dataset):
    def __init__(self, fnames, pfnames, seq_len, transform=None):
        ''' fnames is a list of lists of file names
            pfames is a list of file names (one for each entire sequence)
        '''
        super().__init__()
        self.fnames = fnames
        self.pfnames = pfnames
        self.len = sum([max(0,len(fns)-seq_len+1) for fns in fnames])
        self.sids = []
        for i,fns in enumerate(fnames):
            for j in range(len(fns)-seq_len+1):
                self.sids.append(i)
        self.fsids = []
        for fns in fnames:
            for i in range(len(fns)-seq_len+1):
                self.fsids.append(i)
        self.seq_len = seq_len
        self.transform = transform # Transform at the frame level

    def __getitem__(self, index):
        try:
            s,id = self.sids[index], self.fsids[index]
            #t = time()
            x = [cv2.imread(fn,0) for fn in self.fnames[s][id:id+self.seq_len]]
            if self.transform:
                x = [self.transform(img) for img in x]
            #print('batch transf',256*(time()-t))
            x = [img.unsqueeze(0) for img in x]
            x = torch.cat(x,dim=0).unsqueeze(1)
            abs = load_kitti_odom(self.pfnames[s])[id:id+self.seq_len]
            y = []
            for p in abs:
                p = c3dto2d(p)
                y.append(p)
            y = abs2relative(y,self.seq_len,1)[0]
            y = torch.from_numpy(y).float()
            #print('seq loading',x.size(),y.size(),abs.shape)
            return x,y,abs
        except RuntimeError as re:
            print(re)
        except Exception as e:
            print(e)

    def __len__(self):
        return self.len

if __name__=='__main__':
    '''train_dir = sys.argv[1] #'/home/ubuntu/kitti/dataset'
    valid_dir = sys.argv[2] #'/home/ubuntu/kitti/dataset'
    test_dir = sys.argv[3] #'/home/ubuntu/kitti/'
    '''

    enc_fn = sys.argv[1]
    model_fn = sys.argv[2]
    h_dim = int(sys.argv[3])
    new_dim = (int(sys.argv[4]),int(sys.argv[5]))
    seq_len = int(sys.argv[6])
    batch_size = int(sys.argv[7])
    num_epochs = int(sys.argv[8])
    #transf = transforms.Compose([Rescale(new_dim),ToTensor()])
    #transf = [Rescale(new_dim),ToTensor()]
    transf = ToTensor()

    train_dir,valid_dir,test_dir = list_split_kitti_(new_dim[0],new_dim[1])

    train_dataset = SeqDataset(train_dir[0],train_dir[1],seq_len,transf)
    valid_dataset = SeqDataset(valid_dir[0],valid_dir[1],seq_len,transf)
    test_dataset = SeqDataset(test_dir[0],test_dir[1],seq_len,transf)

    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=4,collate_fn=my_collate)
    valid_loader = DataLoader(valid_dataset,batch_size=batch_size,shuffle=False,num_workers=4,collate_fn=my_collate)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=4,collate_fn=my_collate)

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(device)

    model = VanAE((1,)+new_dim,h_dim).to(device)
    #model = torch.load(enc_fn)
    vo = Conv1dMapper((h_dim,seq_len),(seq_len,12)).to(device)
    model.dec = vo
    for param in model.enc.parameters():
        param.requires_grad = False

    ##model = VanillaAutoencoder((1,)+new_dim).to(device)
    #model = VanillaAutoencoder((2,)+new_dim,h_dim).to(device)
    #model = MLPAutoencoder((2,)+new_dim,h_dim).to(device)
    #model = FastDirectOdometry((1,)+new_dim,(12,)).to(device)
    params = model.parameters()
    optimizer = optim.Adam(params,lr=1e-3)
    min_loss = 1e15
    epoch = 0
    writer = SummaryWriter('/home/ubuntu/log/exp_h{}_l{}_{}x{}'\
                           .format(h_dim,seq_len,new_dim[0],new_dim[1]))

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
    k = 0
    #epoch = num_epochs-1
    for i in range(epoch,num_epochs):
        model.train()
        losses = []
        for j,xy in enumerate(valid_loader):
            #t = time()
            x,y = xy[0].to(device), xy[1].to(device)
            optimizer.zero_grad()
            y_ = model(x)
            #print(y.size(),y_.size())
            loss = loss_fn(y,y_) #loss_fn(y_,y)
            loss.backward()
            optimizer.step()
            writer.add_scalar('train_cost',loss.item(),k)
            losses.append(loss.item())
            k += 1
            print('Batch {} loss: {:.4f}'.format(j,loss.item()))
            #print('inference',time()-t)
        x_ = x[0].view(-1,x.size(-1)).unsqueeze(0)
        #print(x_.size())
        writer.add_image('_img_seq_{}'.format(i),x_)
        #writer.add_image('_img_emb_{}'.format(i),\
        #                 z[:seq_len].unsqueeze(0))

        torch.save({'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'min_loss': min_loss,
                    'epoch': i}, model_fn)
        '''model.eval()
        v_losses = []
        for j,xy in enumerate(valid_loader):
            x,y = xy[0].to(device), xy[1].to(device)
            y_,*_ = model(x)
            loss = loss_fn(y_,y)
            v_losses.append(loss.item())
        mean_train, mean_valid = np.mean(losses),np.mean(v_losses)
        epoch_losses.append([i,mean_train,mean_valid])
        print('Epoch {}\tloss\t{:.3f}\tValid loss\t{:.3f}'\
              .format(i,mean_train,mean_valid))
        if mean_valid < min_loss:
            min_loss = mean_valid
            torch.save({'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'min_loss': min_loss,
                        'epoch': i+1}, model_fn)'''
    model.eval()
    print('Start of plot_eval')
    plot_eval(model,valid_loader,seq_len,device,logger=writer)
    writer.close()
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
