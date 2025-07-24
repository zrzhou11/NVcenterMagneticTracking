import numpy as np
import sys
import os
from system_config import *


# calculate the errors
def calError(result, labeln):
    result = result.cpu().detach().numpy()
    labeln = labeln.cpu().detach().numpy()
    
    nums = len(result)
    resultxy = np.zeros((nums, 2))
    labelxy = (labeln[:, 0:2] - nvinfo.lat)
    
    x_unit = nvinfo.lat * 2 / (iminfo.label_xpixel - 1)
    y_unit = nvinfo.lat * 2 / (iminfo.label_ypixel - 1)

    for i in range(nums):
        x_r, y_r = get_xy(result[i])
        resultxy[i][0] = x_r * x_unit
        resultxy[i][1] = y_r * y_unit

    return resultxy - labelxy 

# calculate xy position from the output distribution
def get_xy(result):
    xx = np.arange(0, 10)
    yy = np.arange(0, 10)
    xx, yy = np.meshgrid(xx, yy)
    return (result * xx).sum(), (result * yy).sum()

def get_files_in_folder_sorted_by_time(folder_path):

    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    sorted_files = sorted(files, key=lambda f: os.path.getmtime(os.path.join(folder_path, f)))

    return sorted_files

# rolling search 
def RollingSearch(size, step, FullImage):
    n = (250 - size) // step + 1  # 250 x 250 image
    for i in range(0, n):
        for j in range(0, n):
            if i == 0 and j == 0:
                maxstd = FullImage[:, step*i: step*i+size, step*j: step*j+size].std(axis=0).mean()
                maxij = (i, j)
            else:
                ijstd  = FullImage[:, step*i: step*i+size, step*j: step*j+size].std(axis=0).mean()
                if maxstd < ijstd:
                    maxstd = ijstd
                    maxij = (i, j)

    LocalImage = FullImage[:, step*maxij[0]: step*maxij[0]+size, step*maxij[1]: step*maxij[1]+size]                    

    return LocalImage, maxij