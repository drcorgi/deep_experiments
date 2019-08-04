import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def seq_pose_loss(p,p_):
    ''' B x L x P
    '''
    '''theta = torch.acos(torch.clamp(p[:,:,0],min=-1+1e-7,max=1-1e-7))
    theta_ = torch.acos(torch.clamp(p_[:,:,0],min=-1+1e-7,max=1-1e-7))
    dtheta1 = torch.abs(theta-theta_)
    dtheta2 = torch.abs(torch.tensor(6.2830)-dtheta1)
    dtheta = torch.min(dtheta1,dtheta2)
    #print('dtheta',torch.max(theta))
    ltheta = torch.exp(torch.tensor(0.0)*dtheta)
    ltheta = torch.mean(ltheta)
    lxy = torch.mean((p[:,:,[3,11]]-p_[:,:,[3,11]])**2.0)
    #print('losses',ltheta,lxy,torch.mean(dtheta))
    return ltheta + lxy'''
    diff = torch.sum((p-p_)**2.0,dim=-1)
    diff = torch.max(diff,dim=-1)[0]
    '''ltheta = torch.sum(diff[:,:,[0,1,2,4,5,6,7,8,9,10]]**4.0)
    lxy = torch.sum(diff[:,:,[3,11]]**2.0)
    b,l,p = diff.size()
    return (ltheta+lxy)/(b*l*p)'''
    return torch.mean(diff)

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

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class VanillaEncoder(nn.Module):
    def __init__(self,in_shape,h_dim):
        ''' (B x L) x C x H x W; B x C x H x W
        '''
        super().__init__()
        self.in_shape = in_shape # C,H,W
        self.filters = 32
        self.h_dim = h_dim #256
        self.conv1 = nn.Conv2d(in_shape[0],self.filters,(5,5),(2,2))
        self.bn1 = nn.BatchNorm2d(self.filters)
        self.conv2 = nn.Conv2d(self.filters,self.filters,(3,3),(2,2))
        self.bn2 = nn.BatchNorm2d(self.filters)
        self.conv3 = nn.Conv2d(self.filters,self.filters,(3,3),(1,1))
        self.bn3 = nn.BatchNorm2d(self.filters)
        #self.conv4 = nn.Conv2d(self.filters,self.filters,(3,3),(1,1))
        self.new_h = (((((in_shape[1]-4)//2-2)//2)-2)//1) #-2
        self.new_w = (((((in_shape[2]-4)//2-2)//2)-2)//1) #-2
        self.flat_dim = self.new_h*self.new_w*self.filters
        print(self.new_h,self.new_w)
        self.fc1 = nn.Linear(self.flat_dim,self.h_dim)
        #self.conv_drop = [nn.Dropout(0.1) for _ in range(3)]

    def forward(self,x):
        shape = x.size()
        if len(shape) == 5:
            x = x.view(shape[0]*shape[1],shape[2],shape[3],shape[4])
        #print(shape)
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        #x = F.relu(self.conv4(x))
        x = x.view(-1,self.flat_dim)
        x = self.fc1(x)
        #x = self.fc1_drop(F.relu(x))
        #x = x.view(shape[0],self.h_dim,shape[1])
        return x

class VanillaDecoder(nn.Module):
    def __init__(self,in_shape,h_dim):
        super().__init__()
        self.in_shape = in_shape # C,H,W
        self.filters = 32
        self.h_dim = h_dim
        self.new_h = (((((in_shape[1]-4)//2-2)//1)-2)//1)
        self.new_w = (((((in_shape[2]-4)//2-2)//1)-2)//1)
        self.flat_dim = self.new_h*self.new_w*2*self.filters
        print(self.new_h,self.new_w)
        self.fc2 = nn.Linear(self.h_dim,self.flat_dim)
        self.deconv1 = nn.ConvTranspose2d(2*self.filters,2*self.filters,(3,3),(1,1),padding=0)
        self.deconv2 = nn.ConvTranspose2d(2*self.filters,self.filters,(3,3),(1,1),padding=0) # ,output_padd$
        self.deconv3 = nn.ConvTranspose2d(self.filters,in_shape[0],(5,5),(2,2),padding=0,output_padding=1)
        self.fc1_drop = nn.Dropout(0.5)
        self.fc2_drop = nn.Dropout(0.5)

    def forward(self,x):
        #x = x.transpose(2,1)
        #shape = x.size()
        #x = x.contiguous().view(shape[0]*shape[1],shape[2])
        #print('d',x.size())
        #x = self.fc1_drop(F.relu(x))
        x = F.relu(self.fc2(x))
        #x = self.fc2_drop(x)
        x = x.view(-1,2*self.filters,self.new_h,self.new_w)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.deconv3(x)
        return x

class VanAE(nn.Module):
    def __init__(self,in_shape,h_dim):
        super().__init__()
        self.enc = VanillaEncoder(in_shape,h_dim)
        self.dec = VanillaDecoder(in_shape,h_dim)

    def forward(self,x):
        x = self.enc(x)
        x = self.dec(x)
        return x

class ImgFlowOdom(nn.Module):
    def __init__(self,flow,in_shape,h_dim,device='cuda:0'):
        super().__init__()
        self.flow = flow
        self.device = device

    def forward(self,x):
        ''' B x L x C x H x W -> (B x L) x C x 2 x H x W
        '''
        sz = x.size()
        assert len(sz) == 5
        b,l,c,h,w = sz
        x_ = torch.zeros(b,l,c,2,h,w).to(self.device)
        for i in range(l):
            x_[:,i] = x[:,max(0,i-1):i+1].transpose(1,2) # or min
        x_ = x_.contiguous().view(b*l,c,2,h,w)
        x_ = self.flow(x_)
        return x_

class DummyFlow(nn.Module):
    def __init__(self,flow,in_shape,h_dim,device='cuda:0'):
        super().__init__()
        self.training = False
        self.device = device

    def forward(self,x):
        ''' B x L x C x H x W -> (B x L) x C x 2 x H x W
        '''
        sz = x.size()
        assert len(sz) == 5
        b,l,c,h,w = sz
        x_ = torch.zeros(b,l,c,2,h,w).to(self.device)
        for i in range(l):
            x_[:,i] = x[:,max(0,i-1):i+1].transpose(1,2)
        x_ = x_.contiguous().view(b*l,c,2,h,w)
        x_ = x_[:,0,1,:,:] - x_[:,0,0,:,:]
        x_ = x_.unsqueeze(1).repeat(1,2,1,1)
        return x_

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

class OdomNorm2d(nn.Module):
    ''' x shape: B x L x O
    '''
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.fc1 = nn.Linear(in_channels,in_channels)
        self.fc2 = nn.Linear(in_channels,3)

    def forward(self,x):
        ''' (B x L) x C -> (B x L) x 3 (theta,x,y) -> B x L x O
        '''
        b,l,c = x.size()
        x = x.contiguous().view(-1,c)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(b,l,3)
        x_ = torch.zeros(b,l,self.out_channels).cuda()
        #print(x[0,-1,0])
        x_[:,:,0] = torch.cos(x[:,:,0])
        x_[:,:,2] = -torch.sin(x[:,:,0])
        x_[:,:,8] = torch.sin(x[:,:,0])
        x_[:,:,10] = torch.cos(x[:,:,0])
        x_[:,:,3] = x[:,:,1]
        x_[:,:,11] = x[:,:,2]
        x_[:,:,[1,4,6,7,9]] = torch.tensor(0.0).cuda()
        x_[:,:,5] = torch.tensor(1.0).cuda()
        return x_

class DirectOdometry(nn.Module):
    ''' Encoder + Odometry into same module
        Input: (B x L) x C x H x W, in_shape: C x H x W
        Output: B x L x O, out_shape: O
    '''
    def __init__(self,in_shape,out_shape,n_hidden):
        super().__init__()
        self.n_hidden = n_hidden
        self.filters = 32
        # Batch dim trick
        self.conv1 = nn.Conv2d(in_shape[0],self.filters,(5,5),(2,2))
        self.conv2 = nn.Conv2d(self.filters,2*self.filters,(3,3),(1,1))
        self.conv3 = nn.Conv2d(2*self.filters,4*self.filters,(3,3),(1,1))
        self.new_h = (((((in_shape[1]-4)//2-2)//1)-2)//1)
        self.new_w = (((((in_shape[2]-4)//2-2)//1)-2)//1)
        self.flat_dim = self.new_h*self.new_w*4*self.filters
        print('Flat dim',self.flat_dim)
        self.fc1 = nn.Linear(self.flat_dim,self.n_hidden)
        self.drop1 = nn.Dropout(0.5)
        self.conv4 = nn.Conv1d(self.n_hidden,2*self.n_hidden,3,1,padding=1)
        self.conv5 = nn.Conv1d(2*self.n_hidden,4*self.n_hidden,3,1,padding=1)
        self.conv6 = nn.Conv1d(4*self.n_hidden,4*self.n_hidden,3,1,padding=1)
        #self.odom_norm = OdomNorm2d(self.n_hidden,12)
        self.fc2 = nn.Linear(4*n_hidden,100*n_hidden)
        self.fc3 = nn.Linear(100*n_hidden,12)

    def forward(self,x):
        size = x.size()
        x = x.view(-1,size[2],size[3],size[4])
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1,self.flat_dim)
        x = F.relu(self.fc1(x))
        z = self.drop1(x)
        #print(x.size())
        x = z.view(-1,size[1],self.n_hidden).transpose(1,2)
        #print(x.size())
        x = F.relu(self.conv4(x))
        #print(x.size())
        x = F.relu(self.conv5(x)) #.transpose(1,2)
        x = F.relu(self.conv6(x)).transpose(1,2)
        #print(x.size())
        #x = self.odom_norm(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #print(x.size())
        return x #,z

class MeanEncoder(nn.Module):
    ''' Encoder + Odometry into same module
        Input: (B x L) x C x H x W, in_shape: C x H x W
        Output: B x L x O, out_shape: O
    '''
    def __init__(self,in_shape):
        super().__init__()
        self.n_hidden = in_shape[-1] + in_shape[-2]

    def forward(self,x):
        mh = torch.mean(x,dim=3)
        mw = torch.mean(x,dim=4)
        x = torch.cat((mh,mw),dim=-1)
        x = torch.mean(x,dim=2)
        x = x.view(-1,self.n_hidden)
        return x

class FastDirectOdometry(nn.Module):
    ''' Encoder + Odometry into same module
        Input: (B x L) x C x H x W, in_shape: C x H x W
        Output: B x L x O, out_shape: O
    '''
    def __init__(self,in_shape,out_shape):
        super().__init__()
        self.n_hidden = in_shape[-1] + in_shape[-2]
        self.fc1 = nn.Linear(self.n_hidden,2*self.n_hidden)
        self.drop1 = nn.Dropout(0.5)
        self.conv4 = nn.Conv1d(2*self.n_hidden,2*self.n_hidden,3,1,padding=1)
        self.conv5 = nn.Conv1d(2*self.n_hidden,2*self.n_hidden,3,1,padding=1)
        self.conv6 = nn.Conv1d(2*self.n_hidden,2*self.n_hidden,3,1,padding=1)
        self.odom_norm = OdomNorm2d(2*self.n_hidden,12)

    def forward(self,x):
        size = x.size()
        mh = torch.mean(x,dim=3)
        mw = torch.mean(x,dim=4)
        x = torch.cat((mh,mw),dim=-1)
        x = torch.mean(x,dim=2)
        x = x.view(-1,self.n_hidden)
        x = F.relu(self.fc1(x))
        z = self.drop1(x)
        x = z.view(-1,size[1],2*self.n_hidden).transpose(1,2)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x)).transpose(1,2)
        x = self.odom_norm(x)
        return x #,z

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
    ''' B x D x L -> B x L x O
    '''
    def __init__(self,in_shape,out_shape):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.filters = 32
        self.conv1 = nn.Conv1d(in_shape[0],self.filters,3,1,groups=1)
        self.bn1 = nn.BatchNorm1d(self.filters)
        self.conv2 = nn.Conv1d(self.filters,self.filters,3,1,groups=1)
        self.bn2 = nn.BatchNorm1d(self.filters)
        self.conv3 = nn.Conv1d(self.filters,self.filters,3,1,groups=1)
        self.bn3 = nn.BatchNorm1d(self.filters)
        self.h_shape = ((((in_shape[1]-2)//1-2)//1)-2)//1
        print(self.h_shape)
        self.fc1 = nn.Linear(self.h_shape*self.filters,10*self.in_shape[1])
        self.bn4 = nn.BatchNorm1d(10*self.in_shape[1])
        self.fc2 = nn.Linear(10*self.in_shape[1],10*self.in_shape[1])
        self.bn5 = nn.BatchNorm1d(10*self.in_shape[1])
        self.fc3 = nn.Linear(10*self.in_shape[1],np.prod(out_shape)) # seq_len * x,y,theta
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.dropout3 = nn.Dropout(p=0.1)
        self.dropout4 = nn.Dropout(p=0.5)
        self.dropout5 = nn.Dropout(p=0.5)

    def forward(self,x):
        ''' x may come as (B x L) x D
        '''
        shape = x.size()
        if len(shape) == 2:
            x = x.view(-1,self.in_shape[-1],shape[-1]).transpose(2,1)
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.dropout3(x)
        x = x.view(-1,self.h_shape*self.filters)
        x = self.bn4(F.relu(self.fc1(x)))
        x = self.dropout4(x)
        x = self.bn5(F.relu(self.fc2(x)))
        x = self.dropout5(x)
        x = self.fc3(x)
        x = x.view((-1,)+self.out_shape)

        '''x_ = x.clone()
        det2 = x_[:,:,0]*x_[:,:,10]-x_[:,:,2]*x_[:,:,8]
        print(det2[0,0])'''
        '''det3 = x_[:,:,0]*x_[:,:,5]*x_[:,:,10]+x_[:,:,1]*x_[:,:,6]*x_[:,:,8]+\
              x_[:,:,2]*x_[:,:,4]*x_[:,:,9]-x_[:,:,2]*x_[:,:,5]*x_[:,:,8]-\
              x_[:,:,1]*x_[:,:,4]*x_[:,:,10]-x_[:,:,0]*x_[:,:,6]*x_[:,:,9]
        print(det3[0,0])'''
        '''det = det + torch.tensor(1e-7)
        x[:,:,[0,1,2,4,5,6,8,9,10]] = x_[:,:,[0,1,2,4,5,6,8,9,10]]/det.unsqueeze(-1).repeat(1,1,9)'''

        x[:,:,[1,4,6,7,9]] = torch.zeros((x.size(0),x.size(1),5)).cuda()
        #x[:,:,[3,11]] = torch.tensor(0.0).cuda()
        x[:,:,5] = torch.tensor(1.0).cuda()

        return x

class Conv1dRecMapper(nn.Module):
    def __init__(self,in_shape,out_shape,bidirectional=False,device='cuda:0'):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.device = device
        self.num_cells = 1
        self.num_dir = 1 if not bidirectional else 2
        self.h = self.num_dir*in_shape[0]
        self.rec = nn.GRU(in_shape[0],in_shape[0],\
                          self.num_cells,bidirectional=bidirectional) # bidirectional=True
        self.fc1 = nn.Linear(self.h,self.h)
        self.fc2 = nn.Linear(self.h,out_shape[-1])
        #self.drop1 = nn.Dropout(p=0.5)
        #self.odom_norm = OdomNorm2d(2*in_shape[0],12)

    def forward(self,x):
        ''' (B x L) x D -> L x B x D
        '''
        shape = x.size()
        #print('x size',shape)
        if len(shape) == 2:
             x = x.view(-1,self.in_shape[-1],shape[-1]).permute(1,0,2)
        #print('x unpacked',x.size())
        h0 = torch.zeros(self.num_dir*self.num_cells,\
                         x.size(1),self.in_shape[0]).to(self.device)
        x, hn = self.rec(x,h0)
        x = x.transpose(1,0) ##
        #print('gru out',x.size())

        x = x.contiguous().view(-1,self.h)
        x = F.relu(self.fc1(x))
        #x = self.drop1(x)
        x = self.fc2(x)
        #print('fc2',x.size())
        x = x.view((-1,)+self.out_shape)
        #print('view',x.size())

        x[:,:,[1,4,6,7,9]] = torch.zeros((x.size(0),x.size(1),5)).to(self.device)
        #x[:,:,[3,11]] = torch.tensor(0.0).cuda()
        x[:,:,5] = torch.tensor(1.0).to(self.device)
        #x = self.odom_norm(x)

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
