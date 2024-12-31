import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import numpy as np
from system_config import *



class NVLocate1(nn.Module):
    def __init__(self, nvinfo, iminfo):
        super(NVLocate1, self).__init__()
        NVnum = nvinfo.xnum * nvinfo.ynum
        output_size = iminfo.label_xpixel * iminfo.label_ypixel
        self.convMix1 = nn.Conv3d(1, NVnum, kernel_size=(1, iminfo.image_xpixel, iminfo.image_ypixel))
        self.convCurve = nn.Conv2d(1, 6, kernel_size=(1, iminfo.timesteps))
        self.fc1 = nn.Linear(NVnum * 6, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, output_size)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.convMix1(x)
        x = x.view(x.shape[0], 1, x.shape[1], -1)
        x = self.convCurve(x)
        x = F.relu(x)
        x = x.view(-1, reduce(lambda a, b: a * b, x.size()[1:]))
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        
        return x.view(-1, 10, 10)    

    
if __name__ == '__main__':
    Net = NVLocate1(nvinfo, iminfo)
    input_image = np.random.rand(2, 1, 20, 100, 100)
    ouput_image = Net(input_image)
    print(ouput_image.shape)