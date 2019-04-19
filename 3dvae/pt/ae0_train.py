import sys
import signal
import numpy as np
import glob, os
import pickle
import torch
import torch.optim as optim

from pt_ae import VanillaAutoencoder
from plotter import *

class UnTrainer():
    def __init__(self,model,output_fn,model_fn,batch_size,valid_ids,device):
        #self.frames = torch.tensor(frames).float()
        self.output_fn = output_fn
        self.model_fn = model_fn
        self.batch_size = batch_size
        self.valid_ids = valid_ids
        #self.test_ids = test_ids
        self.min_loss = 1e15
        self.epoch = 0
        self.device = device
        self.loss_fn = torch.nn.MSELoss()
        self.model = model
        params = self.model.parameters()
        self.optimizer = optim.Adam(params,lr=3e-4) # optim.SGD(params,lr=1e-3,momentum=0.1)
        if os.path.isfile(model_fn):
            print('Loading existing model')
            checkpoint = torch.load(model_fn)
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.min_loss = checkpoint['min_loss']
            self.epoch = checkpoint['epoch']
        else:
            print('Creating new model')

    def save_emb(self,data):
        self.model.eval()
        embs = []
        for s in data:
            semb = []
            s = torch.tensor(s).float()
            for i in range(0,len(s),self.batch_size):
                x = s[i:i+self.batch_size]
                x = x.transpose(1,3).transpose(2,3).to(self.device)
                if len(x) > 0:
                    z = self.model.forward_z(x)
                    semb += z.cpu().detach().numpy().tolist()
            embs.append(np.array(semb))
        with open(self.output_fn,'wb') as f:
            pickle.dump(embs,f)

    def plot_eval(self,data_x,n):
        self.model.eval()
        recs = []
        gt = []
        for i in range(0,n,self.batch_size):
            x = data_x[i:i+self.batch_size].to(self.device)
            y_ = self.model(x).cpu().detach().numpy()
            for r,g in zip(y_,x):
                recs.append(r[0])
                gt.append(g.cpu().detach().numpy()[0])
        print(recs[0].shape,gt[0].shape)
        if not os.path.isdir('/tmp'):
            os.mkdir('/tmp')
        for i,r in enumerate(recs):
            cv2.imwrite('/tmp/{}_rec.png'.format(i),r)
            cv2.imwrite('/tmp/{}_gt.png'.format(i),gt[i])

    def evaluate(self,data_x):
        self.model.eval()
        losses = []
        for i in range(0,len(data_x),self.batch_size):
            x = data_x[i:i+self.batch_size].to(self.device)
            y_ = self.model(x)
            loss = self.loss_fn(y_,x)
            losses.append(loss.item())
        mean_loss = np.mean(losses)
        print('Evaluation: {}'.format(mean_loss))
        return mean_loss

    def train(self,frames,num_epochs):
        frames_valid = frames[:self.valid_ids]
        frames = frames[self.valid_ids:]

        self.model.train()
        num_iter = len(frames)//self.batch_size
        all_ids = np.random.choice(len(frames),[num_epochs,num_iter,self.batch_size])
        for j in range(self.epoch,num_epochs):
            loss = self.evaluate(frames_valid)
            self.model.train()
            if loss < self.min_loss:
                self.min_loss = loss
                torch.save({'model_state': self.model.state_dict(),
                            'optimizer_state': self.optimizer.state_dict(),
                            'min_loss': self.min_loss,
                            'epoch': j}, self.model_fn)
            losses = []
            for i in range(num_iter):
                self.optimizer.zero_grad()
                ids = all_ids[j][i]
                x = frames[ids].to(self.device)
                y_ = self.model(x)
                loss = self.loss_fn(y_,x)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            print('Epoch {}: {}'.format(j,np.mean(losses)))

if __name__ == '__main__':
    if len(sys.argv) != 9:
        print('Usage: input_fn output_fn model_fn batch_size valid_ids test_ids device')
        exit()

    input_fn = sys.argv[1] #'/home/ronnypetson/Documents/deep_odometry/kitti/dataset_frames/sequences/flows_00-10_128x128.pck'
    output_fn = sys.argv[2] #'/home/ronnypetson/Documents/deep_odometry/kitti/dataset_frames/sequences/emb0_128x128/emb0_flows_128x128_00-10.pck'
    model_fn = sys.argv[3] #'/home/ronnypetson/models/pt/test_ae0_.pth'
    batch_size = int(sys.argv[4]) #8
    valid_ids = int(sys.argv[5]) #256
    test_ids = int(sys.argv[6]) #256
    epochs = int(sys.argv[7])
    device = sys.argv[8] #'cuda:0'

    # Load the data
    with open(input_fn,'rb') as f:
        frames = pickle.load(f) #[:1]
    #print(frames[0].shape)

    device = torch.device(device)
    model = VanillaAutoencoder((frames[0].shape[3],frames[0].shape[1],frames[0].shape[2])).to(device)
    t = UnTrainer(model=model,output_fn=output_fn,model_fn=model_fn,batch_size=batch_size\
                  ,valid_ids=valid_ids,device=device)

    if epochs == -1: # Save embeddings
        t.save_emb(frames)
    else:
        # Group the data
        frames = np.concatenate(frames,axis=0).transpose(0,3,1,2)
        print(frames.shape)
        frames = torch.tensor(frames).float()
        frames_test = frames[:test_ids]
        frames = frames[test_ids:]
        t.train(frames,epochs)
        t.evaluate(frames_test)
        t.plot_eval(frames[:valid_ids],10)
