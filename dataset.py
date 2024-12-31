import torch
from torch.utils.data import Dataset
import os
import numpy as np
import random

class dataset(Dataset):
    def __init__(self, root, transforms=None, train=True, labeln_flag=False, train_size=None):
        super(dataset, self).__init__()

        image_root = root + '_image_m'
        self.images = sorted([os.path.join(image_root, img) for img in os.listdir(image_root)])
        label_root = root + '_label'
        self.labels = sorted([os.path.join(label_root, label) for label in os.listdir(label_root)])
        labeln_root = root + '_labeln'
        self.labeln = sorted([os.path.join(labeln_root, labeln) for labeln in os.listdir(labeln_root)])
        self.labeln_flag = labeln_flag
        
        max_train_size = int(len(self.labels) * 0.8)
        if train_size == None:
            train_size = max_train_size
        train_size = min(train_size, max_train_size)
        
        index = np.random.choice(max_train_size, train_size, replace=False)    
        if train == True:
            self.images = np.array(self.images[:max_train_size])[index]
            self.labels = np.array(self.labels[:max_train_size])[index]
            self.labeln = np.array(self.labeln[:max_train_size])[index]
        else:
            self.images = self.images[train_size:]
            self.labels = self.labels[train_size:]
            self.labeln = self.labeln[train_size:]


    def __getitem__(self, idx):
        input_image = np.load(self.images[idx])
        label_image = np.load(self.labels[idx])

        if self.labeln_flag == True:
            labeln = np.load(self.labeln[idx])
            return input_image, label_image, labeln
        else:
            return input_image, label_image
        
    def __len__(self):
        return len(self.labels)


