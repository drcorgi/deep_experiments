import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class VanillaAutoencoder(nn.Module):
    def __init__(self,in_shape):
        super().__init__()
        self.in_shape = in_shape # C,H,W
        self.filters = 32
        self.h_dim = 128
        self.conv1 = nn.Conv2d(in_shape[0],self.filters,(5,5),(2,2))
        #self.bn1 = nn.BatchNorm2d(self.filters)
        self.conv2 = nn.Conv2d(self.filters,self.filters,(3,3),(1,1))
        #self.bn2 = nn.BatchNorm2d(self.filters)
        self.conv3 = nn.Conv2d(self.filters,self.filters,(3,3),(1,1))
        #self.bn3 = nn.BatchNorm2d(self.filters)
        self.new_h = (((((in_shape[1]-4)//2-2)//1)-2)//1)
        self.new_w = (((((in_shape[2]-4)//2-2)//1)-2)//1)
        self.flat_dim = self.new_h*self.new_w*self.filters
        print(self.new_h,self.new_w)
        self.fc1 = nn.Linear(self.flat_dim,self.h_dim)
        #self.bn4 = nn.BatchNorm1d(self.h_dim)
        self.fc2 = nn.Linear(self.h_dim,self.flat_dim)
        #self.bn5 = nn.BatchNorm1d(self.flat_dim)
        self.deconv1 = nn.ConvTranspose2d(self.filters,self.filters,(3,3),(1,1),padding=0)
        #self.bn6 = nn.BatchNorm2d(self.filters)
        self.deconv2 = nn.ConvTranspose2d(self.filters,self.filters,(3,3),(1,1),padding=0) # ,output_padding=1
        #self.bn7 = nn.BatchNorm2d(self.filters)
        self.deconv3 = nn.ConvTranspose2d(self.filters,in_shape[0],(5,5),(2,2),padding=0,output_padding=1)
        self.conv_drops = [nn.Dropout(0.1) for _ in [0,1,2]]
        self.fc_drop = [nn.Dropout(0.5) for _ in [0,1]]
        self.deconv_drops = [nn.Dropout(0.1) for _ in [0,1]]

    def forward_z(self,x):
        x = F.relu(self.conv1(x))
        #x = self.bn1(x)
        x = self.conv_drops[0](x)
        x = F.relu(self.conv2(x))
        #x = self.bn2(x)
        x = self.conv_drops[1](x)
        x = F.relu(self.conv3(x))
        #x = self.bn3(x)
        x = self.conv_drops[2](x)
        x = x.view(-1,self.flat_dim)
        x = self.fc1(x)
        return x

    def forward(self,x):
        #print(x.size())
        x = self.forward_z(x)
        x = F.relu(x)
        #x = self.bn4(x)
        x = self.fc_drop[0](x)
        x = F.relu(self.fc2(x))
        #x = self.bn5(x)
        x = self.fc_drop[1](x)
        x = x.view(-1,self.filters,self.new_w,self.new_h)
        x = F.relu(self.deconv1(x))
        #x = self.bn6(x)
        x = self.deconv_drops[0](x)
        x = F.relu(self.deconv2(x))
        #x = self.bn7(x)
        x = self.deconv_drops[1](x)
        x = self.deconv3(x)
        #print(x.size())
        return x

class Vanilla1dAutoencoder(nn.Module):
    def __init__(self,in_shape):
        super().__init__()
        self.in_shape = in_shape # C,L
        self.filters = 32
        self.conv1 = nn.Conv1d(in_shape[0],self.filters,5,2)
        self.conv2 = nn.Conv1d(self.filters,self.filters,3,1)
        self.conv3 = nn.Conv1d(self.filters,self.filters,3,1)
        self.h_shape = ((((((in_shape[1]-4)//2-2)//1)-2)//1)-2)//1
        print(self.h_shape)
        self.fc1 = nn.Linear(self.h_shape*self.filters,self.in_shape[1]//4)
        self.fc2 = nn.Linear(self.in_shape[1]//4,self.h_shape*self.filters)
        self.deconv1 = nn.ConvTranspose1d(self.filters,self.filters,3,1,padding=0)
        self.deconv2 = nn.ConvTranspose1d(self.filters,self.filters,3,1,padding=0) # ,output_padding=1
        self.deconv3 = nn.ConvTranspose1d(self.filters,in_shape[0],5,2,padding=0,output_padding=1)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        conv2_size = list(x.size())
        x, inds = F.max_pool1d(x,3,1,return_indices=True) # kernel size 3 and strides 1
        x = F.relu(self.conv3(x))
        x = x.view(-1,self.h_shape*self.filters)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(-1,self.filters,self.h_shape)
        x = F.relu(self.deconv1(x))
        x = F.max_unpool1d(x,inds,3,1,output_size=conv2_size)
        x = F.relu(self.deconv2(x))
        x = self.deconv3(x)
        return x

'''class MLP_Mapper(nn.Module):
    def __init__(self,in_shape,out_shape):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.fc1 = nn.Linear(self.h_shape,self.in_shape[1])
        self.fc2 = nn.Linear(self.in_shape[1],self.in_shape[1])
        self.fc3 = nn.Linear(self.in_shape[1],np.prod(out_shape))
        #self.dropout1 = nn.Dropout(p=0.1)
        #self.dropout2 = nn.Dropout(p=0.5)
        self.dropout3 = nn.Dropout(p=0.5)'''

class NormalNoise(nn.Module):
    def __init__(self,stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self,x):
        if self.training:
            x = x + torch.randn(x.size()).cuda() * self.stddev
        return x

class Conv1dMapper(nn.Module):
    def __init__(self,in_shape,out_shape):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.filters = 64
        self.conv1 = nn.Conv1d(in_shape[0],self.filters,3,1,groups=1)
        self.bn1 = nn.BatchNorm1d(self.filters)
        self.conv2 = nn.Conv1d(self.filters,self.filters,3,1,groups=1)
        self.bn2 = nn.BatchNorm1d(self.filters)
        self.conv3 = nn.Conv1d(self.filters,self.filters,3,1,groups=1)
        self.bn3 = nn.BatchNorm1d(self.filters)
        self.h_shape = ((((in_shape[1]-2)//1-2)//1)-2)//1
        print(self.h_shape)
        self.fc1 = nn.Linear(self.h_shape*self.filters,100*self.in_shape[1])
        self.bn4 = nn.BatchNorm1d(100*self.in_shape[1])
        self.fc2 = nn.Linear(100*self.in_shape[1],100*self.in_shape[1])
        self.bn5 = nn.BatchNorm1d(100*self.in_shape[1])
        self.fc3 = nn.Linear(100*self.in_shape[1],np.prod(out_shape))
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.dropout3 = nn.Dropout(p=0.1)
        self.dropout4 = nn.Dropout(p=0.5)
        self.dropout5 = nn.Dropout(p=0.5)
        #self.noise = NormalNoise(0.01)

    def forward(self,x):
        #x = self.noise(x)
        x = self.dropout1(self.bn1(F.relu(self.conv1(x))))
        #x = F.relu(x)
        #x = F.relu(self.conv1(x))
        #x = self.dropout1(x)
        x = self.dropout2(self.bn2(F.relu(self.conv2(x))))
        #x = F.relu(x)
        #x = F.relu(self.conv2(x))
        #x = self.dropout2(x)
        #conv2_size = list(x.size())
        #x, inds = F.max_pool1d(x,3,1,return_indices=True) # kernel size 3 and strides 1
        x = self.dropout3(self.bn3(F.relu(self.conv3(x))))
        #x = F.relu(x)
        #x = self.dropout3(x)
        x = x.view(-1,self.h_shape*self.filters)
        x = self.dropout4(self.bn4(F.relu(self.fc1(x))))
        #x = F.relu(x)
        x = self.dropout5(self.bn5(F.relu(self.fc2(x))))
        #x = F.relu(x)
        x = self.fc3(x)
        x = x.view((-1,)+tuple(self.out_shape))
        x[:,[1,4,6,7,9],:] = torch.zeros(x.size(0),5,x.size(2)).cuda()
        x[:,5,:] = torch.tensor(1.0).cuda()
        #x[:,[0,2,8,10],:] = x[:,[0,2,8,10],:]/torch.norm(x[:,[0,2,8,10],:],p=2,dim=1).unsqueeze(1)
        return x

if __name__=='__main__':
    model = Conv1dMapper([64,128],5) #Vanilla1dAutoencoder([64,128]) #VanillaAutoencoder([1,64,64])
    loss_fn = torch.nn.MSELoss()
    params = model.parameters()
    optimizer = optim.Adam(params,lr=1e-3)
    data = torch.tensor(np.random.uniform(-10.0,10.0,[256,64,128])).float()
    data_y = torch.tensor(np.random.uniform(0.0,20.0,[256,5])).float()
    ids = np.random.choice(data.size(0),[50,64])
    for i in range(50):
        optimizer.zero_grad()
        x = data[ids[i]]
        y = data_y[ids[i]]
        y_ = model(x)
        loss = loss_fn(y,y_)
        print(loss.item())
        loss.backward()
        #for p in model.parameters(): print(p.grad)
        optimizer.step()
