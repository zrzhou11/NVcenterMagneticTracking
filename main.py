import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import visdom
import matplotlib.pyplot as plt
import time
import os
from dataset import *
from model import *
from system_config import *
from model_test import *


class Training_configs:
    def __init__(self, net_num, data_root, exp_num):
        self.net_num = net_num
        self.exp_num = exp_num
        self.gpu_num = 0
        self.data_root = data_root
        self.max_epoch = 501
        self.ckp = None
        self.train_size = None
        self.visual = False
        self.learning_rate = 0.001
        
def test(epoch, test_dataloader, model, device, loss_fn):
    #在测试集上看效果
    signal_image, label_image, labeln = next(iter(test_dataloader))
    signal_image = signal_image.to(device)
    label_image = label_image.to(device)
    # ignore the dropout layer in test mode
    model.eval()
    result = model(signal_image)
    loss_test = loss_fn(result, label_image)
    Errors = calError(result, labeln)
    RMSE = np.sqrt((Errors**2).sum(axis=1).mean(axis=0))
    
    print("epoch: {}-----  RMSE: {}nm --- loss: {}".format(epoch, RMSE * 1e9, loss_test.cpu().detach()))
    model.train()
        
    return 

def train(configs):
    
    # set the train device
    if torch.cuda.is_available():    
        device = "cuda:{}".format(configs.gpu_num) 
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"    
    
    # set the model
    model = eval('NVLocate{}'.format(configs.net_num))(nvinfo, iminfo).to(device)
    # load the initial parameters
    if configs.ckp != None:
        res = torch.load(configs.ckp, map_location=device)
        model.load_state_dict(res['net'])
    
    model.train()
    
    # set the dataloader
    training_data = dataset(configs.data_root, train_size=configs.train_size)
    dataloader = DataLoader(training_data, batch_size=128, num_workers=12, shuffle=True)
    test_data = dataset(configs.data_root, labeln_flag=True, train=False)
    test_dataloader = DataLoader(test_data, batch_size=512, shuffle=True)  
    loss_fn = nn.MSELoss()
                              
    # set the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.learning_rate)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    #设置并记录checkpoint
    checkpoint = {
        'epoch':0,
        'net':model.state_dict(),
        'optimizer':optimizer.state_dict()}            
    ckp_path = 'checkpoint/{}'.format(configs.exp_num)
    if not os.path.isdir(ckp_path):
        os.makedirs(ckp_path)
    torch.save(checkpoint, '{}/{}.pth'.format(ckp_path, time.strftime("epoch_{:04d}_time_%Y-%m-%d_%H-%M-%S".format(0), time.localtime())))
    
    for epoch in range(configs.max_epoch):
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if epoch % 100 == 0:
            test(epoch, test_dataloader, model, device, loss_fn)
            #save checkpoint
            checkpoint = {
                'epoch':epoch,
                'net':model.state_dict(),
                'optimizer':optimizer.state_dict()}            
            torch.save(checkpoint, '{}/{}.pth'.format(ckp_path, time.strftime("epoch_{:04d}_time_%Y-%m-%d_%H-%M-%S".format(epoch + 1), time.localtime())))


if __name__ == '__main__':
                              
    net_num = 1
    data = "data/main"
    name = "main"
    tconfigs = Training_configs(net_num, data, name)
    tconfigs.train_size = 6400
    tconfigs.gpu_num = 0
    train(tconfigs)
                    
