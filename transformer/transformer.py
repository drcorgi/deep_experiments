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
   def __init__(self,d_model,max_seq_len=128):
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

    def forward(self,x):
        #print(x.size())
        Q = torch.bmm(self.W_q.unsqueeze(0).repeat(x.size(0),1,1),x)
        #print(Q.size())
        K = torch.bmm(self.W_k.unsqueeze(0).repeat(x.size(0),1,1),x)
        #print(K.size())
        V = torch.bmm(self.W_v.unsqueeze(0).repeat(x.size(0),1,1),x)
        #print(V.size())
        x = torch.bmm(Q.transpose(1,2),K)/self.sqrt_d
        #print(x.size())
        x = F.softmax(x)
        x = torch.bmm(V,x)
        #print(x.size())
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,n_heads):
        super().__init__()
        self.heads = nn.ModuleList([])
        for i in range(n_heads):
            self.heads.append(AttentionHead(d_model))
        # D, n_heads*D x n_heads*D, D
        self.W = nn.Parameter(torch.zeros(n_heads*d_model,d_model),requires_grad=True)

    def forward(self,x):
        #print(x.size())
        x = [h(x) for h in self.heads]
        x = torch.cat(x,dim=1)
        #print(x.size(),self.W.size())
        x = torch.bmm(self.W.unsqueeze(0).repeat(x.size(0),1,1).transpose(1,2),x)
        #print(x.size())
        return x

class TransformerEncoder(nn.Module):
    def __init__(self,d_model,n_heads):
        super().__init__()
        self.att = MultiHeadAttention(d_model,n_heads)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.fc = nn.Linear(d_model,d_model)

    def forward(self,x):
        #print(x.size())
        x_ = self.att(x)
        #print(x_.size())
        x = self.bn1(x+x_)
        size = x.size()
        #print(size)
        x = x.view(-1,x.size(1))
        x_ = F.relu(self.fc(x))
        x = x.view(size)
        x_ = x_.view(size)
        #print(x_.size())
        x = self.bn2(x+x_)
        #print(x.size())
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
        for enc in self.encs:
            x = enc(x)
            #print(x.size())
        return x

if __name__ == '__main__':
    model = TransformerEncoderStack(64,8,4)
    #TransformerEncoder(64,8)
    #MultiHeadAttention(64,8)
    #AttentionHead(64)
    params = model.parameters()

    loss_fn = torch.nn.MSELoss()
    optimizer = optim.Adam(params,lr=1e-3)
    data = torch.tensor(np.random.uniform(-10.0,10.0,[256,64,128])).float()
    #data_y = torch.tensor(np.random.uniform(0.0,20.0,[256,5])).float()

    ids = np.random.choice(data.size(0),[50,64])
    for i in range(50):
        optimizer.zero_grad()
        x = data[ids[i]]
        #y = data_y[ids[i]]
        y_ = model(x)
        loss = loss_fn(x,y_)
        print(loss.item())
        loss.backward()
        #for p in model.parameters(): print(p.grad)
        optimizer.step()
