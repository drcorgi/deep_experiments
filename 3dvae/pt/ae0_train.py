import sys
import signal
import numpy as np
import glob, os
import pickle
from pt_ae import VanillaAutoencoder

batch_size = 64
wlen = 128
stride = wlen
seq_len = 64
test_ids = 16000
num_classes = 5
__train = sys.argv[1]

input_fn = '/home/ronnypetson/Documents/deep_odometry/kitti/dataset_frames/sequences/flows_128x128/flows_128x128_26_30.npy'
input_fn_poses = '/home/ronnypetson/Documents/deep_odometry/kitti/dataset_frames/sequences/flows_128x128/poses_flat_26-30.npy'
model_fn = '/home/ronnypetson/models/pt/ae0_.pth'

if __name__ == '__main__':
    # Load the data
    frames = np.load(input_fn)
    poses_abs = np.load(input_fn_poses)

    # Group the data
    frames = np.array([frames[i:i+seq_len] for i in range(len(data))])
    # ... load relative poses windows

    device = torch.device('cuda:0')
    print(device)
    data = torch.tensor(data,requires_grad=False).float()
    emos = torch.tensor(emos,requires_grad=False).float()
    model = VanillaAutoencoder([data.size(1),data.size(2)],num_classes).to(device)
    params = model.parameters()
    optimizer = optim.Adam(params,lr=3e-4) # optim.SGD(params,lr=1e-3,momentum=0.1)

    if os.path.isfile(model_fn):
        checkpoint = torch.load(model_fn)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    if __train != '0': # Train mode
        data = data[test_ids:]
        emos = emos[test_ids:]
        model.train()
        loss_fn = torch.nn.MSELoss()
        num_iter = len(data)//batch_size
        for j in range(2):
            all_ids = np.random.choice(len(data),[num_iter,batch_size])
            for i in range(num_iter):
                optimizer.zero_grad()
                ids = all_ids[i]
                x = data[ids].to(device)
                y = emos[ids].to(device)
                y_ = model(x)
                loss = loss_fn(y_,y)
                loss.backward()
                optimizer.step()
                if i%50 == 0:
                    amax_y = torch.argmax(y,dim=1)
                    amax_y_ = torch.argmax(y_,dim=1)
                    acc = 100*(amax_y == amax_y_).sum()/batch_size
                    print('Epoch {}: {}\t{}'.format(j,loss.item(),acc))
        torch.save({'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict()},model_fn)
    else: # Evaluation mode
        data = data[:test_ids]
        emos = emos[:test_ids]
        model.eval()
        loss_fn = torch.nn.MSELoss()
        accs = []
        losses = []
        for i in range(0,len(data),batch_size):
            x = data[i:i+batch_size].to(device)
            y = emos[i:i+batch_size].to(device)
            y_ = model(x)
            loss = loss_fn(y_,y)
            amax_y = torch.argmax(y,dim=1)
            amax_y_ = torch.argmax(y_,dim=1)
            acc = 100*(amax_y == amax_y_).sum()/batch_size
            losses.append(loss.item())
            accs.append(acc.item())
        print('Test: {} {}'.format(np.mean(losses),np.mean(accs)))
