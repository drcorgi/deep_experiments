import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch.autograd import Variable

class Embedder(nn.Module):
    def __init__(self,vocab_size,d_model):
        super().__init__()
        #self.embed = nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        #return self.embed(x)
        return x

class PositionalEncoder(nn.Module):
   # B, D, L
   def __init__(self,d_model,max_seq_len=16):
        super().__init__()
        self.d_model = torch.tensor(d_model).float()
        pe = torch.zeros(max_seq_len,d_model)
        for pos in range(max_seq_len):
            for i in range(0,d_model,2):
                pe[pos,i] = \
                np.sin(pos/(10000 ** ((2*i)/d_model)))
                pe[pos,i+1] = \
                np.cos(pos/(10000 ** ((2*(i+1))/d_model)))
        pe = pe.transpose(0,1)
        pe = pe.unsqueeze(0) # dimensão do batch adicionada
        self.pe = pe
        #self.register_buffer('pe',pe)

   def forward(self,x):
       # x.size() = [batch size, d_model, max_seq_len]
       x = x*torch.sqrt(self.d_model)
       seq_len = x.size(2)
       x = x + self.pe
       return x

class AttentionHead(nn.Module):
    # B, D, L
    def __init__(self,d_model):
        super().__init__()
        self.W_q = nn.Parameter(torch.zeros(d_model,d_model,requires_grad=True))
        self.W_k = nn.Parameter(torch.zeros(d_model,d_model,requires_grad=True))
        self.W_v = nn.Parameter(torch.zeros(d_model,d_model,requires_grad=True))
        self.sqrt_d = torch.tensor(np.sqrt(d_model))

    def forward(self,x,mask_from=None,Q=None,K=None,V=None):
        #print(x.size())
        if Q is None:
            Q = torch.bmm(self.W_q.unsqueeze(0).repeat(x.size(0),1,1),x)
        #print(Q.size())
        if K is None:
            K = torch.bmm(self.W_k.unsqueeze(0).repeat(x.size(0),1,1),x)
        #print(K.size())
        if V is None:
            V = torch.bmm(self.W_v.unsqueeze(0).repeat(x.size(0),1,1),x)
        #print(Q.size(),K.size())
        x = torch.bmm(Q.transpose(1,2),K)/self.sqrt_d
        #print(x.size())
        if mask_from is not None:
            x[:,mask_from:,mask_from:] = torch.tensor(-1e12)
        x = F.softmax(x)
        x = torch.bmm(V,x)
        #print(x.size())
        return x,Q,K,V

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,n_heads):
        super().__init__()
        self.heads = nn.ModuleList([])
        for i in range(n_heads):
            self.heads.append(AttentionHead(d_model))
        # D, n_heads*D x n_heads*D, D
        self.W = nn.Parameter(torch.zeros(n_heads*d_model,d_model),requires_grad=True)

    def forward(self,x,mask_from=None,Q=None,K=None,V=None):
        #print(x.size())
        if Q is None or K is None or V is None:
            x,Q,K,V = zip(*[h(x,mask_from) for h in self.heads])
            x,Q,K,V = [list(a) for a in [x,Q,K,V]]
        else:
            x,Q,_,_ = zip(*[self.heads[i](x,mask_from,Q[i],K[i],V[i])\
                          for i in range(len(self.heads))])
            x = list(x)
        #print(len(list(zip([h(x) for h in self.heads]))))

        x = torch.cat(x,dim=1)
        #print(x)
        x = torch.bmm(self.W.unsqueeze(0).repeat(x.size(0),1,1).transpose(1,2),x)
        #print(x.size())
        return x,Q,K,V

class TransformerEncoder(nn.Module):
    def __init__(self,d_model,n_heads):
        super().__init__()
        self.att = MultiHeadAttention(d_model,n_heads)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.fc1 = nn.Linear(d_model,4*d_model)
        self.fc2 = nn.Linear(4*d_model,d_model)

    def forward(self,x):
        #print(x.size())
        x_,Q,K,V = self.att(x)
        #print(x_.size())
        x = self.bn1(x+x_)
        size = x.size()
        #print(size)
        x = x.view(-1,x.size(1))
        x_ = F.relu(self.fc1(x))
        x_ = self.fc2(x_)
        x = x.view(size)
        x_ = x_.view(size)
        #print(x_.size())
        x = self.bn2(x+x_)
        #print(x.size())
        #print(x)
        return x,Q,K,V

class TransformerDecoder(nn.Module):
    def __init__(self,d_model,n_heads):
        super().__init__()
        self.att = MultiHeadAttention(d_model,n_heads)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.enc_dec = MultiHeadAttention(d_model,n_heads)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.fc1 = nn.Linear(d_model,4*d_model)
        self.fc2 = nn.Linear(4*d_model,d_model)
        self.bn3 = nn.BatchNorm1d(d_model)

    def forward(self,x,mask_from,K,V):
        x_,Q,_,_ = self.att(x,mask_from)
        #print(x_.size())
        x = self.bn1(x+x_)
        size = x.size()
        x_,_,_,_ = self.enc_dec(x,mask_from=None,Q=Q,K=K,V=V)
        x = self.bn2(x+x_)
        #print(size)
        x = x.view(-1,x.size(1))
        x_ = F.relu(self.fc1(x))
        x_ = self.fc2(x_)
        x = x.view(size)
        x_ = x_.view(size)
        #print(x_.size())
        x = self.bn3(x+x_)
        #print(x.size())
        #print(x)
        return x

class TransformerEncoderStack(nn.Module):
    def __init__(self,d_model,n_heads,n_encoders):
        super().__init__()
        self.pos_enc = PositionalEncoder(d_model)
        self.encs = nn.ModuleList([])
        for i in range(n_encoders):
            self.encs.append(TransformerEncoder(d_model,n_heads))

    def forward(self,x):
        x = self.pos_enc(x)
        #print(x.size())
        for enc in self.encs[:-1]:
            x = enc(x)[0]
            #print(x.size())
        x,_,K,V = self.encs[-1](x)
        #print(x)
        return x,K,V

class TransformerDecoderStack(nn.Module):
    def __init__(self,d_model,d_out,n_heads,n_decoders):
        super().__init__()
        self.d_model = d_model
        self.d_out = d_out
        self.fc1 = nn.Linear(d_out,d_model)
        self.pos_enc = PositionalEncoder(d_model)
        self.decs = nn.ModuleList([])
        for i in range(n_decoders):
            self.decs.append(TransformerDecoder(d_model,n_heads))
        self.fc2 = nn.Linear(d_model,d_out)

    def forward(self,x,mask_from,K,V):
        #print(x.size())
        size = (x.size(0),self.d_model,x.size(2))
        x = x.view(-1,x.size(1))
        x = F.relu(self.fc1(x))
        x = x.view(size)
        x = self.pos_enc(x)
        #print(x.size())
        for dec in self.decs:
            x = dec(x,mask_from,K,V)
            #print(x.size())
        size = (x.size(0),self.d_out,x.size(2))
        x = x.view(-1,x.size(1))
        x = F.relu(self.fc2(x))
        x = x.view(size)[:,:,-1:]
        #print(x)
        return x

class Transformer(nn.Module):
    def __init__(self,d_model,d_out,n_heads,n_encoders,n_decoders):
        super().__init__()
        self.d_out = d_out
        self.enc = TransformerEncoderStack(d_model,n_heads,n_encoders)
        self.dec = TransformerDecoderStack(d_model,d_out,n_heads,n_decoders)

    def forward(self,x):
        x,K,V = self.enc(x)
        #print(x.size())
        all_y = torch.zeros(x.size(0),self.d_out,x.size(2)+1)
        y = all_y[:,:,[0]]
        for i in range(1,x.size(2)+1):
            y = self.dec(y,i,K,V)
            print(y.size())
            all_y[:,:,[i]] = y
        return all_y[:,:,1:]

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = Transformer(32,32,4,2,2)
    #TransformerDecoderStack(64,64,8,4)
    #TransformerDecoder(64,8)
    #TransformerEncoderStack(64,8,4)
    #TransformerEncoder(64,8)
    #MultiHeadAttention(64,8)
    #AttentionHead(64)
    params = model.parameters()

    loss_fn = torch.nn.MSELoss()
    optimizer = optim.Adam(params,lr=1e-3)
    data = torch.tensor(np.random.uniform(-10.0,10.0,[256,32,16])).float()
    #data_y = torch.tensor(np.random.uniform(0.0,20.0,[256,5])).float()

    ids = np.random.choice(data.size(0),[50,64])
    for i in range(50):
        optimizer.zero_grad()
        x = data[ids[i]]
        #y = data_y[ids[i]]
        #K = [torch.zeros(64,64,128) for _ in range(8)]
        #V = [torch.ones(64,64,128) for _ in range(8)]
        y_ = model(x)
        loss = loss_fn(x,y_)
        print(loss.item())
        loss.backward()
        #for p in model.parameters(): print(p.grad)
        optimizer.step()
