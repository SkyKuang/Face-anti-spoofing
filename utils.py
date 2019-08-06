import sys
import time
import argparse
import os
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import spline  
matplotlib.use('Agg')
plt.switch_backend('agg')
import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

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

def plot_figure(path,train_loss,eval_loss,length):
    plt.figure()
    plt.title('Loss--Epoch')
    # xnew = np.linspace(0,len(train_loss)-1,length)
    x_train = np.array([i for i in range(len(train_loss))])
    y_train = np.array(train_loss)
    # train_smooth = spline(x_train,y_train,xnew)
    x_eval = np.array([i for i in range(len(eval_loss))])
    y_eval = np.array(eval_loss)
    # eval_smooth = spline(x_eval,y_eval,xnew)
    plt.plot(x_train, y_train,lw=1, label='train')
    plt.plot(x_eval, y_eval, lw=1, label='eval')
    plt.legend(loc='upper left')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig(path+f'/loss.png')
    
def plot_curve(path,train,eval,length):
    plt.figure()
    plt.title('TPR@FPR=e4--Epoch')
    # xnew = np.linspace(0,len(train)-1,length)
    x_train = np.array([i for i in range(len(train))])
    y_train = np.array(train)
    # train_smooth = spline(x_train,y_train,xnew)
    x_eval = np.array([i for i in range(len(eval))])
    y_eval = np.array(eval)
    # eval_smooth = spline(x_eval,y_eval,xnew)
    plt.plot(x_train, y_train, lw=1,label='train')
    plt.plot(x_eval, y_eval, lw=1,label='eval')
    plt.legend(loc='upper left')
    plt.ylabel('TPR@FPR=e4')
    plt.xlabel('Epoch') 
    plt.savefig(path+f'/curve.png')

