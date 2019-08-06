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
from torch.optim import lr_scheduler

from loger import Logger
from dataloader import load_cisia_surf
from model_mmf import Model
from evalution import eval_fun
from utils import plot_figure
from centerloss import CenterLoss
from utils import plot_figure,plot_curve

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

train_data, val_data, test_data= load_cisia_surf(train_size=args.batch_size,test_size=args.test_size)

model = Model(pretrained=False,num_classes=2)
criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,weight_decay=5e-4)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

ct_loss = CenterLoss(num_classes=2, feat_dim=512)
optimzer4ct = optim.SGD(ct_loss.parameters(), lr =0.01, momentum=0.9,weight_decay=5e-4)
scheduler4ct = lr_scheduler.ExponentialLR(optimzer4ct, gamma=0.95)

if use_cuda:
    model = model.cuda()
    criterion = criterion.cuda()
    ct_loss = ct_loss.cuda()

eval_history = []
train_loss = []
eval_loss = []
train_score = []
eval_score = []
test_score = []

def train(epochs):
    for epoch in range(epochs):
        logger.Print(f"|~~~~~~~~~~~~~~~~~~~~~~~~Training epoch:{epoch}~~~~~~~~~~~~~~~~~~~~~~~~~~~|")     
        model.train()        
        scheduler.step()
        scheduler4ct.step()
        y_prob_list = []
        y_pLabel_list = []
        y_label_list = []
        total_loss = 0
        total_itr = 0

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
                labels = labels + torch.randn(labels.shape).long()/10
                labels = labels.cuda()
                
            optimizer.zero_grad()
            optimzer4ct.zero_grad()
            outputs, features = model(rgb_img,depth_img,ir_img,hsv_img,ycb_img)
            score, prob_outputs = torch.max(outputs.data, 1)

            loss_anti = criterion(outputs,labels)
            # loss_ct = ct_loss(features,labels)
            loss_ct = 0
            
            a = 1
            b = scheduler.get_lr()[0]
            loss = a*loss_anti+b*loss_ct
            
            loss.backward()
            optimizer.step()
            # optimzer4ct.step()  

            total_loss += loss.item()
            total_itr = itr

            if itr>150:
                for i in range(len(labels)):
                    if prob_outputs[i] > 0.5: 
                        y_pLabel_list.append(1)
                    else:
                        y_pLabel_list.append(0)
                y_prob_list.extend(prob_outputs.data.cpu().numpy())
                y_label_list.extend(labels.data.cpu().numpy())

        eval_result,score = eval_fun(y_prob_list,y_pLabel_list,y_label_list)
        train_score.append(score)        
        logger.Print(eval_result)
        avg_loss = total_loss/total_itr
        train_loss.append(avg_loss)
        # message = f'|epoch:{epoch}-iter:{itr}|loss:{loss:.6f}|'
        message = f'|epoch:{epoch}-iter:{itr}|loss:{loss.item():.6f}|loss_anti:{loss_anti.item():.6f}|loss_ct:{loss_ct:.6f}'
        logger.Print(message)

        logger.Print(f"|~~~~~~~~~~~~~~~~~~~~~~~~val epoch:{epoch}~~~~~~~~~~~~~~~~~~~~~~~~~~~|")     
        val(epoch, val_data, 0)

        if (epoch+1)%5 == 0:    
            logger.Print(f"|~~~~~~~~~~~~~~~~~~~~~~~~test epoch:{epoch}~~~~~~~~~~~~~~~~~~~~~~~~~~~|")     
            val(epoch, test_data, 1)
            pass

        if (epoch+1)%20 == 0:    
            plot_curve(save_path,train_score,eval_score,epoch*5)       
            plot_figure(save_path,train_loss,eval_loss,epoch*5)

    for i in range(len(eval_history)):
        logger.Print(eval_history[i])

    train_loss_np = np.array(train_loss)
    eval_loss_np = np.array(eval_loss)
    np.save(save_path+f'/train_loss_np.npy',train_loss_np)
    np.save(save_path+f'/eval_loss_np.npy',eval_loss_np)
    train_np = np.array(train_score)
    eval_np = np.array(eval_score)
    test_np = np.array(test_score)
    np.save(save_path+f'/train.npy',train_np)
    np.save(save_path+f'/eval.npy',eval_np)
    np.save(save_path+f'/test.npy',test_np)

def val(epoch=0, data_set=val_data,flag=0):
    y_prob_list = []  
    y_pLabel_list = []  
    y_label_list = []
    model.eval()
    total_loss = 0
    total_itr = 0
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

                outputs,_ = model(rgb_img,depth_img,ir_img,hsv_img,ycb_img)
                prob_outputs = F.softmax(outputs,1)[:,1]  # 预测为1的概率    
                loss = criterion(outputs,labels)

                total_loss += loss.item()
                total_itr = itr

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


    eval_result,score = eval_fun(y_prob_list,y_pLabel_list,y_label_list,logger)
    eval_history.append(eval_result)
    logger.Print(eval_result)

    if flag == 0 :
        eval_score.append(score)
        avg_loss = total_loss/total_itr
        eval_loss.append(avg_loss)
        message = f'|eval|loss:{avg_loss:.6f}|'
        logger.Print(message)
    else:
        test_score.append(score)

    with open(save_path+f'/val_{epoch}.txt', 'w') as f:
        for i in range(len(y_label_list)):
            message = f'{y_prob_list[i]:.6f} {y_pLabel_list[i]} {y_label_list[i]}'
            f.write(message)
            f.write('\n')
        f.close()

if __name__ == '__main__':
    torch.manual_seed(999)
    if args.train == True:
        train(args.epochs)
    else:
        val()
