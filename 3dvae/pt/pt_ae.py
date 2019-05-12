import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DepthWiseConv2d(nn.Module):
    def __init__(self,in_filters,out_filters,kernel_size,stride,padding):
        super().__init__()
        self.depthwise = nn.Conv2d(in_filters,in_filters,\
                                   kernel_size=kernel_size,stride=stride,\
                                   padding=padding,groups=in_filters)
        self.pointwise = nn.Conv2d(in_filters,out_filters,kernel_size=1)

    def forward(self,x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DepthWiseConvTranspose2d(nn.Module):
    def __init__(self,in_filters,out_filters,kernel_size,stride,padding,output_padding):
        super().__init__()
        self.depthwise = nn.ConvTranspose2d(in_filters,in_filters,\
                                   kernel_size=kernel_size,stride=stride,\
                                   padding=padding,output_padding=output_padding,\
                                   groups=in_filters)
        self.pointwise = nn.ConvTranspose2d(in_filters,out_filters,kernel_size=1)

    def forward(self,x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class VanillaAutoencoder(nn.Module):
    def __init__(self,in_shape,h_dim):
        super().__init__()
        self.in_shape = in_shape # C,H,W
        self.filters = 32
        self.h_dim = h_dim #256
        self.conv1 = nn.Conv2d(in_shape[0],self.filters,(5,5),(2,2))
        #self.conv1 = DepthWiseConv2d(in_shape[0],self.filters,(5,5),(2,2),padding=0)
        #self.bn1 = nn.BatchNorm2d(self.filters)
        self.conv2 = nn.Conv2d(self.filters,self.filters,(3,3),(1,1))
        #self.conv2 = DepthWiseConv2d(self.filters,self.filters,(3,3),(1,1),padding=0)
        #self.bn2 = nn.BatchNorm2d(self.filters)
        self.conv3 = nn.Conv2d(self.filters,self.filters,(3,3),(1,1))
        #self.conv3 = DepthWiseConv2d(self.filters,self.filters,(3,3),(1,1),padding=0)
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
        #self.deconv1 = DepthWiseConvTranspose2d(self.filters,self.filters,(3,3),(1,1),padding=0,output_padding=0)
        #self.bn6 = nn.BatchNorm2d(self.filters)
        self.deconv2 = nn.ConvTranspose2d(self.filters,self.filters,(3,3),(1,1),padding=0) # ,output_padding=1
        #self.deconv2 = DepthWiseConvTranspose2d(self.filters,self.filters,(3,3),(1,1),padding=0,output_padding=0)
        #self.bn7 = nn.BatchNorm2d(self.filters)
        self.deconv3 = nn.ConvTranspose2d(self.filters,in_shape[0],(5,5),(2,2),padding=0,output_padding=1)
        #self.deconv3 = DepthWiseConvTranspose2d(self.filters,in_shape[0],(5,5),(2,2),padding=0,output_padding=1)
        #self.conv_drops = [nn.Dropout(0.1) for _ in [0,1,2]]
        self.fc_drop = [nn.Dropout(0.5) for _ in [0,1]]
        #self.deconv_drops = [nn.Dropout(0.1) for _ in [0,1]]

    def forward_z(self,x):
        #print(x.size())
        x = F.relu(self.conv1(x))
        #print(x.size())
        x = F.relu(self.conv2(x))
        #print(x.size())
        x = F.relu(self.conv3(x))
        #print(x.size())
        #print()
        x = x.view(-1,self.flat_dim)
        x = F.relu(self.fc1(x))
        x = self.fc_drop[0](x)
        return x

    def forward(self,x):
        x = self.forward_z(x)
        #x = F.relu(x)
        #x = self.fc_drop[0](x)
        x = F.relu(self.fc2(x))
        x = self.fc_drop[1](x)
        x = x.view(-1,self.filters,self.new_h,self.new_w)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.deconv3(x)
        return x

class MLPAutoencoder(nn.Module):
    def __init__(self,in_shape,h_dim):
        super().__init__()
        self.in_shape = in_shape
        flat_shape = np.prod(in_shape)
        self.fc1 = nn.Linear(flat_shape,2*flat_shape)
        #self.fc2 = nn.Linear(2*flat_shape,2*flat_shape)
        self.fc3 = nn.Linear(2*flat_shape,h_dim)
        self.fc4 = nn.Linear(h_dim,2*flat_shape)
        #self.fc5 = nn.Linear(2*flat_shape,2*flat_shape)
        self.fc6 = nn.Linear(2*flat_shape,flat_shape)

    def forward_z(self,x):
        #print(x.size())
        x = x.view((-1,np.prod(self.in_shape)))
        #print(x.size())
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

    def forward(self,x):
        x = self.forward_z(x)
        x = F.relu(self.fc4(x))
        #x = F.relu(self.fc5(x))
        x = self.fc6(x)
        x = x.view((-1,)+self.in_shape)
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

class MLPMapper(nn.Module):
    def __init__(self,in_shape,out_shape):
        super().__init__()
        self.out_shape = out_shape
        self.in_flat = np.prod(in_shape)
        out_flat = np.prod(out_shape)
        self.fc1 = nn.Linear(self.in_flat,2*self.in_flat)
        self.fc2 = nn.Linear(2*self.in_flat,2*self.in_flat)
        self.fc3 = nn.Linear(2*self.in_flat,out_flat)

    def forward(self,x):
        x = x.view(-1,self.in_flat)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.view((-1,)+self.out_shape)
        x[:,[1,4,6,7,9],:] = torch.zeros(x.size(0),5,x.size(2)).cuda()
        x[:,5,:] = torch.tensor(1.0).cuda()
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
        self.fc3 = nn.Linear(100*self.in_shape[1],np.prod(out_shape)) # seq_len * x,y,theta
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.dropout3 = nn.Dropout(p=0.1)
        self.dropout4 = nn.Dropout(p=0.5)
        self.dropout5 = nn.Dropout(p=0.5)

        self.control_dist = in_shape[1]//3 # in_shape[1] % 3 == 1
        self.control_pts = [1,self.control_dist,2*self.control_dist,3*self.control_dist]
        self.regular_pts = [p for p in range(in_shape[1]) if p not in self.control_pts]

    def forward(self,x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.dropout3(x)
        x = x.view(-1,self.h_shape*self.filters)
        x = self.dropout4(self.bn4(F.relu(self.fc1(x))))
        x = self.dropout5(self.bn5(F.relu(self.fc2(x))))
        x = self.fc3(x)
        x = x.view((-1,)+tuple(self.out_shape))

        x[:,[1,4,6,7,9],:] = torch.zeros((x.size(0),5,self.out_shape[-1])).cuda()
        x[:,5,:] = torch.tensor(1.0).cuda()

        for i in self.regular_pts[1:]:
            j = i//self.control_dist
            l,r = self.control_pts[j], self.control_pts[j+1]
            alpha = torch.tensor((i-l)/self.control_dist).cuda()
            alpha_ = (torch.tensor(1.0) - alpha).cuda()
            for k in [0,2,8,10,3,11]:
                x[:,k,i] = alpha*x[:,k,l].clone() + alpha_*x[:,k,r].clone()

        return x

        '''x = x.view((-1,3,self.in_shape[1]))
        x_ = torch.zeros((x.size(0),)+tuple(self.out_shape)).cuda()
        x_[:,5,:] = torch.tensor(1.0).cuda()

        x_[:,0,self.control_pts] = torch.cos(x[:,0,self.control_pts])
        x_[:,2,self.control_pts] = -torch.sin(x[:,0,self.control_pts])
        x_[:,8,self.control_pts] = -x_[:,2,self.control_pts]
        x_[:,10,self.control_pts] = x_[:,0,self.control_pts]
        x_[:,3,self.control_pts] = x[:,1,self.control_pts]
        x_[:,11,self.control_pts] = x[:,2,self.control_pts]

        for i in self.regular_pts:
            j = i//self.control_dist
            l,r = j*self.control_dist,(j+1)*self.control_dist
            alpha = torch.tensor((i-l)/self.control_dist).cuda()
            alpha_ = (torch.tensor(1.0) - alpha).cuda()
            for k in [0,2,8,10,3,11]:
                x_[:,k,i] = alpha*x_[:,k,l].clone() + alpha_*x_[:,k,r].clone()

        return x_'''

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
