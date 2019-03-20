import sys
import time
import argparse
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
plt.switch_backend('agg')
import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F

from evalution import eval_fun
from model_main import load_model
from loghepler import Logger
from dataloader import load_cisia_surf
from dataloader import load_sample_cisia_surf

import pdb  

time_str = '_'.join(time.ctime().split())
time_str = time_str.replace(':','_')
use_cuda = True if torch.cuda.is_available() else False

parser = argparse.ArgumentParser(description='face anto-spoofing')
parser.add_argument('--batch-size', default='128', type=int, help='train batch size')
parser.add_argument('--test-size', default='64', type=int, help='test batch size')
parser.add_argument('--save-path', default='./logs/', type=str, help='log save path')
parser.add_argument('--root', default='/home/kuanghuafeng/datasets/CASIA-SURF', type=str, help='log save path')
parser.add_argument('--epochs', default=52, type=int, help='train epochs')
parser.add_argument('--train', default=False, type=bool, help='train')
parser.add_argument('--message', default='message:', type=str, help='message')
parser.add_argument('--pretrained-path', default='./model.tar', type=str, help='pretrained path')
parser.add_argument('--pretrained', default=False, type=bool, help='pretrained')
parser.add_argument('--argument-data', default=True, type=bool, help='pretrained')

args = parser.parse_args()

save_path = args.save_path

if not os.path.exists(save_path):
    os.mkdir(save_path)
logger = Logger(f'{save_path}/log.log')
logger.Print(args.message)

train_data,val_data,test_data,train_data_trans = load_cisia_surf(root=args.root,train_size=args.batch_size,test_size=args.test_size)
model = load_model(pretrained=False,num_classes=2)
optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9,weight_decay=5e-4)
criterion = nn.CrossEntropyLoss() 
criterion_mse = nn.MSELoss() 

# optimizer = AdaBound(model.parameters(), 0.0005, betas=(0.9, 0.999),
#                         final_lr=0.001, gamma=1e-4,
#                         weight_decay=5e-4)

Live_centers = torch.zeros(512)
Spoof_centers = torch.zeros(512)

if args.pretrained == True:
    checkpoint = torch.load(args.pretrained_path)
    model.load_state_dict(checkpoint)

if use_cuda:
    model = model.cuda()
    criterion = criterion.cuda()
    Live_centers = Live_centers.cuda()
    Spoof_centers = Spoof_centers.cuda()

def train(epochs):
    loss_history = []
    data_loader = train_data
    for epoch in range(args.epochs):
        model.train(True)
        # logger.Print(f"|~~~~~~~~~~~~~~~~~~~~~~~~Training epoch:{epoch}~~~~~~~~~~~~~~~~~~~~~~~~~~~|")     
        # pdb.set_trace()
        for itr ,data in enumerate(data_loader,1):
            rgb_img = data[0]
            depth_img = data[1]
            ir_img = data[2]
            hsv_img = data[3]
            YCbCr_img = data[4]
            labels = data[5]
            if use_cuda:
                rgb_img = rgb_img.cuda()
                depth_img = depth_img.cuda()
                ir_img = ir_img.cuda()
                hsv_img = hsv_img.cuda()
                YCbCr_img = YCbCr_img.cuda()
                labels = labels.cuda()
            
            optimizer.zero_grad()
            img_crops = image_crop_f(rgb_img)
            features,outputs = model(rgb_img,depth_img,ir_img,hsv_img,YCbCr_img,img_crops)         
            prob_outputs = F.softmax(outputs,1)[:,1]  # 预测为1的概率    
            score, preds_outputs = torch.max(outputs.data, 1)

            mask_live = preds_outputs.view(-1,1).float()
            global Live_centers
            Live_centers = torch.mean(features*mask_live,0)
            mask_spoof = torch.abs(preds_outputs.view(-1,1)-1).float()
            global Spoof_centers
            Spoof_centers = torch.mean(features*(mask_spoof),0)

            loss_anti = criterion(outputs,labels)
            loss_feat = criterion_mse(Live_centers,Spoof_centers)
            loss_feat = torch.exp(-loss_feat*0.001)
            a = 1
            b = 0.1
            loss = loss_anti*a + loss_feat*b
            loss.backward()
            optimizer.step()
            loss_history.append(loss)
  
            if itr%100 == 0:
                # logger.Print(f"|~~~~~~~~~~~~~~~~~~~~~~~~Training~~~~~~~~~~~~~~~~~~~~~~~~~~~|")     
                message = f'|epoch:{epoch}-iter:{itr}|loss:{loss:.6f} |loss_feat:{loss_feat:.6f} |loss_anti:{loss_anti:0.6f}|'
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
                eval_fun(y_prob_list,y_pLabel_list,y_label_list,logger)
            # with open(save_path+f'/val_{epoch}.txt', 'w') as f:
            #     for i in range(len(y_prob_list)):
            #         message = f'{y_prob_list[i]:.6f} {y_pLabel_list[i]} {y_label_list[i]}'
            #         f.write(message)
            #         f.write('\n')
            #     f.close()
        
        if epoch%2 == 0:   
            val(epoch)
            plot(loss_history)
            # with open(save_path+f'/val_{epoch}.txt', 'w') as f:
        else:
            plot(loss_history)
        if (epoch+1)%49 ==0:
            test(epoch)
            torch.save(model.state_dict(),save_path+f'/model_new.tar')  

def val(epoch=0):
    logger.Print(f"|~~~~~~~~~~~~~~~~~~~~~~~~val epoch:{epoch}~~~~~~~~~~~~~~~~~~~~~~~~~~~|")     
    model.eval()   
    y_prob_list = []  
    y_pLabel_list = []  
    y_label_list = [] 
    for itr ,data in enumerate(val_data,1):
        rgb_img = data[0]
        depth_img = data[1]
        ir_img = data[2]
        hsv_img = data[3]
        YCbCr_img = data[4]
        labels = data[5]
        epoch_data = val_data
        if use_cuda:
            rgb_img = rgb_img.cuda()
            depth_img = depth_img.cuda()
            ir_img = ir_img.cuda()
            hsv_img = hsv_img.cuda()
            YCbCr_img = YCbCr_img.cuda()
            labels = labels.cuda()

        img_crops = image_crop_f(ir_img)
        features,outputs = model(rgb_img,depth_img,ir_img,hsv_img,YCbCr_img,img_crops)
        prob_outputs = F.softmax(outputs,1)[:,1]  # 预测为1的概率    

        for i in range(len(labels)):
            if prob_outputs[i] > 0.5:
                y_pLabel_list.append(1)
            else:
                y_pLabel_list.append(0)
            y_prob_list.append(prob_outputs[i].data.cpu().numpy())
            y_label_list.append(labels[i].data.cpu().numpy())
       
    eval_fun(y_prob_list,y_pLabel_list,y_label_list,logger)

def test(epoch=0):
    model.eval()
    logger.Print(f"|~~~~~~~~~~~~~~~~~~~~~~~~Testing epoch:{epoch}~~~~~~~~~~~~~~~~~~~~~~~~~~~|")     
    with open(save_path+f'/outputs.txt', 'w') as f:
        for itr ,data in enumerate(test_data,1):
            rgb_img = data[0]
            depth_img = data[1]
            ir_img = data[2]
            img_paths = data[3]
            hsv_img = data[4]
            YCbCr_img = data[5]

            if use_cuda:    
                rgb_img = rgb_img.cuda()
                depth_img = depth_img.cuda()
                ir_img = ir_img.cuda()
                hsv_img = hsv_img.cuda()
                YCbCr_img = YCbCr_img.cuda()

            img_crops = image_crop_f(ir_img)
            features,outputs = model(rgb_img,depth_img,ir_img,hsv_img,YCbCr_img,img_crops)
            outputs = F.softmax(outputs,1)
            prob_outputs = F.softmax(outputs,1)[:,1]  # 预测为1的概率    

            y_prob_list = []  
            depth_mean_live = torch.tensor(21700).cuda().float()
            depth_mean_spoof = torch.tensor(12500).cuda().float()
            depth_var_live = torch.tensor(11500000).cuda().float()
            depth_var_spoof = torch.tensor(16000000).cuda().float()
            for i in range(len(img_paths[0])):
                depth_mean = torch.mean(torch.sum(depth_img[i]))
                x = torch.sum(depth_img[i])
                prob_live = Gauss(x,depth_mean_live,depth_var_live)
                prob_spoof = Gauss(x,depth_mean_spoof,depth_var_spoof)
                prob_depth = (depth_mean-depth_mean_spoof)/(depth_mean_live-depth_mean_spoof)
                a ,b,c,d = 0.7,0.1,0.1,0.1
                if prob_depth < 0:
                    prob_depth = torch.tensor(0).cuda().float()
                elif prob_depth > 1:
                    prob_depth = torch.tensor(1).cuda().float()
                if prob_depth < 0.15:
                    a -= 0.15
                    b += 0.15
                if prob_live < 0.15:
                    a -= 0.15
                    c += 0.15
                if (1-prob_spoof) < 0.15:
                    a -= 0.15
                    d += 0.15
                
                final_prob = a*prob_outputs[i] + prob_depth*b + prob_live*c + (1-prob_spoof)*d
                y_prob_list.append(final_prob.data.cpu().numpy())

            for j in range(len(img_paths[0])):
                message = f'{img_paths[0][j]} {img_paths[1][j]} {img_paths[2][j]} {y_prob_list[j]:.8f}'
                f.write(message)
                f.write('\n')           
        f.close()


def image_crop_f(image):
    image_crops = []
    for i in range(3):
        for j in range(3):
            x_ = i*24
            y_ = j*24
            w_ = x_+64
            h_ = y_+64
            # img_crop = image[:,x_:w_,y_:h_]
            img_crop = image[:,:,x_:w_,y_:h_]
            image_crops.append(img_crop)
    return image_crops

def plot(data):
    plt.figure()
    plt.plot([i for i in range(len(data))],data)
    plt.savefig(save_path+f'/loss.png')

def Gauss(x,mean,var):
    PI = 3.141592654
    a = (x-mean)*(x-mean)/(2*var)
    b = torch.exp(-a)
    c = 10000/(torch.sqrt(var)*torch.sqrt(torch.tensor(2*PI).cuda().float()))
    y = b*c
    return y

if __name__ == '__main__':
    print(args.train)
    if args.train == True:
        train(args.epochs)
    else:
        model.load_state_dict(torch.load(args.pretrained_path))
        test()



            
            

