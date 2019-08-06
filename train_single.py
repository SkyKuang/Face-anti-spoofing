import sys
import time
import argparse
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import pdb
import numpy as np

import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.optim import lr_scheduler

from loger import Logger
from dataloader import load_cisia_surf
from model_single import Model
from evalution import eval_fun
from utils import plot_figure

time_str = '_'.join(time.ctime().split())
time_str = time_str.replace(':','_')
use_cuda = True if torch.cuda.is_available() else False

parser = argparse.ArgumentParser(description='face anto-spoofing')
parser.add_argument('--batch-size', default='128', type=int, help='train batch size')
parser.add_argument('--test-size', default='64', type=int, help='test batch size')
parser.add_argument('--save-path', default='./logs/', type=str, help='log save path')
parser.add_argument('--checkpoint', default='model.pth', type=str, help='pretrained model checkpoint')
parser.add_argument('--message', default='message', type=str, help='pretrained model checkpoint')
parser.add_argument('--epochs', default=101, type=int, help='train epochs')
parser.add_argument('--train', default=True, type=bool, help='train')
args = parser.parse_args()

save_path = args.save_path + f'{args.message}_{time_str}'

if not os.path.exists(save_path):
    os.mkdir(save_path)
logger = Logger(f'{save_path}/log.log')
logger.Print(args.message)

train_data, val_data, test_data = load_cisia_surf(train_size=args.batch_size,test_size=args.test_size)
model = Model(pretrained=False,num_classes=2)
criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,weight_decay=5e-4)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

if use_cuda:
    model = model.cuda()
    criterion = criterion.cuda()

loss_history = []
eval_history = []

def train(epochs):
    for epoch in range(epochs):
        logger.Print(f"|~~~~~~~~~~~~~~~~~~~~~~~~Training epoch:{epoch}~~~~~~~~~~~~~~~~~~~~~~~~~~~|")     
        model.train()     
        scheduler.step()
        print(epoch,scheduler.get_lr()[0])
        for itr, data in enumerate(train_data,1):
            rgb_img = data[0]
            depth_img = data[1]
            ir_img = data[2]
            hsv_img = data[3]
            ycb_img = data[4]
            labels = data[5]

            if use_cuda:
                rgb_img = rgb_img.cuda()
                depth_img = depth_img.cuda()
                ir_img = ir_img.cuda()
                hsv_img = hsv_img.cuda()
                ycb_img = ycb_img.cuda()
                labels = labels.cuda()
            
            optimizer.zero_grad()
            outputs = model(ir_img)
            score, prob_outputs = torch.max(outputs.data, 1)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            if itr%200 == 0:
                message = f'|epoch:{epoch}-iter:{itr}|loss:{loss:.6f}|'
                logger.Print(message)
                y_prob_list = []
                y_pLabel_list = []
                y_label_list = []
                for i in range(len(labels)):
                    if prob_outputs[i] > 0.5: 
                        y_pLabel_list.append(1)
                    else:
                        y_pLabel_list.append(0)
                y_prob_list = prob_outputs.data.cpu().numpy()
                y_label_list = labels.data.cpu().numpy()
                eval_result = eval_fun(y_prob_list,y_pLabel_list,y_label_list)
                logger.Print(eval_result)
                loss_history.append(loss.item())

        if epoch%5 == 0:    
            logger.Print(f"|~~~~~~~~~~~~~~~~~~~~~~~~val epoch:{epoch}~~~~~~~~~~~~~~~~~~~~~~~~~~~|")     
            val(epoch, val_data)
            logger.Print(f"|~~~~~~~~~~~~~~~~~~~~~~~~test epoch:{epoch}~~~~~~~~~~~~~~~~~~~~~~~~~~~|")     
            val(epoch, test_data)
            pass    
 

    plot_figure(save_path,loss_history)
    loss_np = np.array(loss_history)
    np.save(save_path+'/loss.npy',loss_np)

def val(epoch=0, data_set=val_data):
    y_prob_list = []  
    y_pLabel_list = []  
    y_label_list = []
    model.eval()
    with open(save_path+f'/prob_{epoch}.txt', 'w') as fb:
        with torch.no_grad():
            for itr ,data in enumerate(data_set,1):
                rgb_img = data[0]
                depth_img = data[1]
                ir_img = data[2]
                hsv_img = data[3]
                ycb_img = data[4]
                labels = data[5]
                if use_cuda:
                    rgb_img = rgb_img.cuda()
                    depth_img = depth_img.cuda()
                    ir_img = ir_img.cuda()
                    hsv_img = hsv_img.cuda()
                    ycb_img = ycb_img.cuda()
                    labels = labels.cuda()

                outputs = model(ir_img)
                prob_outputs = F.softmax(outputs,1)[:,1]  # 预测为1的概率    
                for i in range(len(labels)):
                    if prob_outputs[i] > 0.5: 
                        y_pLabel_list.append(1)
                    else:
                        y_pLabel_list.append(0)
                    message = f'{prob_outputs[i]:0.6f},{labels[i]}'
                    fb.write(message)
                    fb.write('\n')
                    y_prob_list.append(prob_outputs[i].data.cpu().numpy())
                    y_label_list.append(labels[i].data.cpu().numpy())
        fb.close()

    # pdb.set_trace()
    with open(save_path+f'/val_{epoch}.txt', 'w') as f:
        for i in range(len(y_label_list)):
            message = f'{y_prob_list[i]:.6f} {y_pLabel_list[i]} {y_label_list[i]}'
            f.write(message)
            f.write('\n')
        f.close()

    eval_result = eval_fun(y_prob_list,y_pLabel_list,y_label_list,logger)
    eval_history.append(eval_result)
    logger.Print(eval_result)

if __name__ == '__main__':
    if args.train == True:
        train(args.epochs)
    else:
        val()
