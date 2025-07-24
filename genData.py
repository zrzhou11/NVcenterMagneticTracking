import sys
from multiprocessing import Pool
from data_utils import *
from system_config import *
import pickle
import time
import os
import json

def make_dirs(filename):
    # make dirs
    image_pathm = '{}_image_m'.format(filename)  # difference of two image sequences
    label_path = '{}_label'.format(filename)     # 2D distribution label
    labeln_path = '{}_labeln'.format(filename)   # coordinate label
    
    if not os.path.isdir(image_pathm):
        os.makedirs(image_pathm)
    if not os.path.isdir(label_path):
        os.makedirs(label_path)
    if not os.path.isdir(labeln_path):
        os.makedirs(labeln_path)

def genData(num, ex, ey, ez, file, NVpara, nvinfo, iminfo):
    # gen image and label
    # genImage with MNP at (ex, ey, ez)
    ImG = Image_generate(ex, ey, ez, nvinfo, iminfo, NVpara)     
    ImgSeq_w = ImG.genImage()
    label = ImG.genLabel()
    labeln = np.array([ex, ey, ez])
    # genImage without MNP, equal to (ex, ey, 1) while 1 >> 1e-9    
    ImGd = Image_generate(ex, ey, 1, nvinfo, iminfo, NVpara)
    ImgSeq_o = ImGd.genImage()
    
    # save path
    image_pathm = '{}_image_m'.format(file)
    label_path = '{}_label'.format(file)
    labeln_path = '{}_labeln'.format(file)

    # save file
    np.save(image_pathm + '/{:05d}'.format(num), ImgSeq_w - ImgSeq_o)
    np.save(label_path + '/{:05d}'.format(num), label)
    np.save(labeln_path + '/{:05d}'.format(num), labeln)     
        

if __name__ == '__main__':
    pos_std_xy = 30e-9
    pos_std_z = 6e-9
    spt_std = 9e3
    # train and test 8000
    N = 8000
    filename = "data/main"       

    make_dirs(filename)

    # generate NV lattice
    NVpara = [genNVdisp(nvinfo, pos_std_xy=pos_std_xy, pos_std_z=pos_std_z, spt_std=spt_std) for i in range(N)]
    # generate mnp location
    ex_array = [(np.random.rand() * 2 + 1) * nvinfo.lat for i in range(N)]
    ey_array = [(np.random.rand() * 2 + 1) * nvinfo.lat for i in range(N)]
    # save the NV's setting
    with open(filename + '.pkl', 'wb') as p:
        pickle.dump(NVpara, p)
            
    with open(filename + '-nvinfo.json', 'w') as p:
        json.dump(vars(nvinfo), p)
    with open(filename + '-iminfo.json', 'w') as p:
        json.dump(vars(iminfo), p)

    def mpg(num):
        ex = ex_array[num]
        ey = ey_array[num]
        ez = 100e-9
        genData(num, ex, ey, ez, filename, NVpara[num], nvinfo, iminfo)

        return 0
    
    '''
    Multiprocess generation, work on linux(may have some issue on windows)
    In windows, the subprocess can not view the variables like ex_array, eyarray, filename......
    One can use starmap to input all those variables into every subprocess, which may cost more computation resources.
    '''
    with Pool(20) as p:
        nums = [i for i in range(N)]
        p.map(mpg, nums)
