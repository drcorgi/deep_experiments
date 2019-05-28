''' Metadados da base de Odometria visual
    Tipo: lista de dicionários ('sub_base' 'sequence' 'sid_frame' 'frame_fn' 'odom_fn')
'''

import os
import pickle
from glob import glob

if __name__=='__main__':
    id_sub_base = 'KITTI - grayscale visual odometry'

    fn1 = '/home/ronnypetson/Documents/deep_odometry/kitti/dataset_frames/sequences/*/image_0/*'
    fn2 = '/home/ronnypetson/Documents/deep_odometry/kitti/dataset_frames/00/image_0/*'
    fn3 = '/home/ronnypetson/Documents/deep_odometry/kitti/dataset_frames/01/image_0/*'
    fn1 = sorted([fn for fn in glob(fn1) if os.path.isfile(fn)])
    fn2 = sorted([fn for fn in glob(fn2) if os.path.isfile(fn)])
    fn3 = sorted([fn for fn in glob(fn3) if os.path.isfile(fn)])
    fns = fn2+fn3+fn1

    odom_fns = glob('/home/ronnypetson/Documents/deep_odometry/kitti/dataset/poses/*.txt')
    odom_fns = sorted(odom_fns)

    ''' Supposing the frame (one per image) filenames are aligned with the odometry (one per sequence) files
    '''
    frame_fn_id = 0
    meta = []
    for fn in odom_fns:
        print(fn)
        with open(fn,'r') as f:
            content = f.readlines()
        fid = 0
        for l in content:
            entry = {}
            entry['sub_base'] = id_sub_base
            entry['sequence'] = fn[-6:-4]
            entry['sid_frame'] = fid
            entry['slength'] = len(content)
            entry['frame_fn'] = fns[frame_fn_id]
            entry['odom_fn'] = fn
            fid += 1
            frame_fn_id += 1
            meta.append(entry)
            print(entry)

    with open('visual_odometry_database.meta','wb') as f:
        pickle.dump(meta,f)
