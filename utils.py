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
import torch.nn.functional as F

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


