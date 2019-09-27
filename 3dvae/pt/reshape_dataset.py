import cv2
import os, sys
from glob import glob

if __name__ == '__main__':
    base_in = sys.argv[1] #'/home/ubuntu/kitti/raw/<seq_date>/seq_id/image_00/data/'
    base_out = sys.argv[2] #'/home/ubuntu/kitti/raw_32x128/<seq_date>/seq_id/image_00/data/'
    new_h = int(sys.argv[3])
    new_w = int(sys.argv[4])

    dirname = base_out #+'/{}x{}/'.format(new_h,new_w)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    indir = base_in+'/*.png'
    infns = glob(indir)
    for fn in infns:
        img = cv2.imread(fn,0) #cv2.imread(fn,0)
        img = cv2.resize(img,(new_w,new_h)) # yes
        bname = os.path.basename(fn)
        outfn = dirname + bname
        cv2.imwrite(outfn,img)
        print(img.shape,outfn)
