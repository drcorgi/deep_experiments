import cv2
import numpy as np
import os, glob
import pykitti
import re
from plotter import *

class OptFlowsSaver:
    def __init__(self,seq_dirs,\
                      flows_dir=\
                      '/home/ronnypetson/Documents/deep_odometry/kitti/dataset_frames/sequences/flows_128x128/',\
                      frame_shape=(128,128)):
        self.seq_dirs = seq_dirs
        self.flows_dir = flows_dir
        self.frame_shape = tuple(frame_shape)

    def load_frames(self,fdir):
        frames = []
        fnames = os.listdir(fdir)
        fnames = [f for f in fnames if os.path.isfile(fdir+'/'+f)]
        fnames = sorted(fnames,key=lambda x: int(x[:-4]))
        imgs = [cv2.imread(fdir+'/'+fname,0) for fname in fnames]
        for f in imgs:
            f = cv2.resize(f,self.frame_shape,interpolation=cv2.INTER_LINEAR)
            frames.append(f)
        return frames

    def load_seq_frames_(self):
        seq_limits = []
        print('Loading sequence from '+self.seq_dirs[0])
        frames = self.load_frames(self.seq_dirs[0])
        seq_limits.append(len(frames))
        for s in self.seq_dirs[1:]:
            print('Loading sequence from '+s)
            sframes = self.load_frames(s)
            frames = np.concatenate((frames,sframes),axis=0)
            seq_limits.append(len(frames))
        np.save(self.flows_dir+'/seq_limits.npy',np.array(seq_limits))
        return frames

    def load_seq_frames(self,base,date,drive):
        data = [pykitti.raw(b,d,drv) for b,d,drv in zip(base,date,drive)]
        frames = []
        for d in data:
            frames += [cv2.resize(np.array(im),self.frame_shape) for im in d.cam0]
        return frames

    def save_seq_poses(self,base,date,drive,fname):
        data = [pykitti.raw(b,d,drv) for b,d,drv in zip(base,date,drive)]
        poses = []
        for d_ in data:
            poses += [flat_homogen(d[1]) for d in d_.oxts]
        np.save(self.flows_dir+'/'+fname,np.array(poses[1:]))

    def save_opt_flows(self,base,date,drive,fname):
        frames = self.load_seq_frames(base,date,drive)
        flows = []
        for i in range(len(frames)-1):
            flow = cv2.calcOpticalFlowFarneback(frames[i],frames[i+1],None,0.5,3,15,3,5,1.2,0)
            flows.append(flow)
        flows = np.array(flows)
        np.save(self.flows_dir+'/'+fname,flows)

if __name__=='__main__':
    seq_dirs = glob.glob('/home/ronnypetson/Downloads/*_*_*/*_drive_*_sync/image_00/data/')
    seq_dirs = sorted(seq_dirs)
    base = [re.findall('.*[0-9]+_[0-9]+_[0-9]+',s)[0][:-21] for s in seq_dirs]
    date = [re.findall('[0-9]+_[0-9]+_[0-9]+',s)[0] for s in seq_dirs]
    drive = [re.findall('drive_[0-9]+_sync',s)[0][6:-5] for s in seq_dirs]
    saver = OptFlowsSaver(seq_dirs)
    saver.save_opt_flows(base,date,drive,'flows_128x128_26_30.npy')
    saver.save_seq_poses(base,date,drive,'poses_flat_26-30.npy')
    pass
