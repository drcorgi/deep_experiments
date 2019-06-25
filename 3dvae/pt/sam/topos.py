import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class SEM(nn.Module):
    '''B x L x D -> B x L' x D', B x L x D
    '''
    def __init__(self,l,d,hdim):
        super().__init__()
        self.l = l
        self.d = d
        self.hdim = hdim
        self.fc1 = nn.Linear(d*l,d*l)
        self.fc2 = nn.Linear(d*l,hdim)

    def forward(self,x):
        shape = x.size()
        assert shape[1] % self.l == 0
        L_ = shape[1]//self.l
        assert L_ > 0
        x = x.view(shape[0]*L_,self.l,shape[2])
        x = x.transpose(2,1)
        x = x.contiguous().view(-1,self.l*self.d)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(shape[0],L_,self.hdim)
        return x

class SDM(nn.Module):
    '''B x L' x D' -> B x L x D
    '''
    def __init__(self,l,d,hdim):
        super().__init__()
        self.l = l
        self.d = d
        self.hdim = hdim
        self.fc1 = nn.Linear(hdim,d*l)
        self.fc2 = nn.Linear(d*l,d*l)

    def forward(self,x):
        shape = x.size()
        L = shape[1]*self.l
        assert L > 0
        x = x.view(shape[0]*shape[1],self.hdim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(shape[0],L,self.d)
        return x

class SAM(nn.Module):
    '''B x L x D -> B x L' x D', B x L x D
    '''
    def __init__(self,l,d,hdim):
        super().__init__()
        self.sem = SEM(l,d,hdim)
        self.sdm = SDM(l,d,hdim)

    def forward(self,x):
        z = self.sem(x)
        x = self.sdm(z)
        return x,z

class SAM2(nn.Module):
    '''B x L x D -> B x L' x D', B x L x D
    '''
    def __init__(self,l,d,hdim):
        super().__init__()
        self.sam1 = SAM(l,d,hdim)
        self.sam2 = SAM(l,hdim,hdim)

    def forward(self,x):
        x1,z1 = self.sam1(x)
        x2,z2 = self.sam2(z1)
        return x1,x2,z1,z2

def check_grads(model,label=''):
    for name,p in model.named_parameters():
        if p.grad is None:
            print(label,name)

if __name__=='__main__':
    #model = SEM(2,16,8)
    #model = SDM(2,16,8)
    #model = SAM(2,16,8)
    model = SAM2(2,16,8)

    loss_fn = torch.nn.MSELoss()
    params = model.parameters()
    optimizer = optim.Adam(params,lr=1e-3)
    data = torch.tensor(np.random.uniform(-10.0,10.0,[256,16,16])).float()
    data_y = torch.tensor(np.random.uniform(0.0,20.0,[256,4,8])).float()

    ids = np.random.choice(data.size(0),[200,64])
    for i in range(200):
        optimizer.zero_grad()
        x = data[ids[i]]
        y = data_y[ids[i]]

        x1,x2,z1,z2 = model(x)
        loss = loss_fn(x,x1)
        loss2 = loss_fn(y,z2)
        print(loss.item(),loss2.item())
        loss = loss + loss2
        loss.backward()
        optimizer.step()
        #check_grads(model)
