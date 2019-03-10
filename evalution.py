import pdb  
import sys
import time
import argparse
import os
import logging

import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')
class Logger():
    def __init__(self,logPath):
        logger = logging.getLogger(__name__)
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(logPath)
        handler.setLevel(logging.INFO)
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.addHandler(console)
        self.logger = logger

    def Print(self,message):
        self.logger.info(message)

def add_args():
    parser = argparse.ArgumentParser(description='face anto-spoofing')
    parser.add_argument('--file_path', default='./prob.txt', type=str, help='y prob txt path')
    parser.add_argument('--label_path', default='./val_label.txt', type=str, help='y label txt path')
    args = parser.parse_args()
    return args

def evalution(y_probs,y_labels):
    fpr, tpr, thresholds = metrics.roc_curve(y_labels, y_probs, pos_label=1)
    score_1 = tpr[np.where(fpr>=0.01)[0][0]]
    score_2 = tpr[np.where(fpr>=0.001)[0][0]]
    score_3 = tpr[np.where(fpr>=0.0001)[0][0]] 
    return score_1,score_2,score_3

def eval_fun(y_prob_list,y_pLabel_list,y_label_list,logger=None):
    TP,TN,FP,FN = 0,0,0,0

    for i in range(len(y_label_list)):
        if y_pLabel_list[i] == 1 and y_label_list[i] == 1:
            TP += 1
        elif y_pLabel_list[i] == 0 and y_label_list[i] == 0:
            TN += 1
        elif y_pLabel_list[i] == 1 and y_label_list[i] == 0:
            FP += 1
        elif y_pLabel_list[i] == 0 and y_label_list[i] == 1:
            FN += 1
        else:
            pass
        
    APCER = float(FP)/(TN+FP)
    NPCER = float(FN)/(FN+TP)
    ACER = (APCER+NPCER)/2
    FPR = float(FP)/(FP+TN)
    TPR = float(TP)/(TP+FN)
    
    y_prob_np = np.array(y_prob_list)
    y_label_np = np.array(y_label_list)
    score_1,score_2,score_3 = evalution(y_prob_np,y_label_np)
    message = f'|TP:{TP} |TN:{TN} |FP:{FP} |FN:{FN} |APCER:{APCER:.6F} |NPCER:{NPCER:.6F} '\
                f'|ACER:{ACER:.6F} |FPR:{FPR:.6F} |TPR:{TPR} |FPR=e2:{score_1:.6f} |FPR=e3:{score_2:.6f} |FPR=e4:{score_3:.6f}|'
    logger.Print(message)

if __name__ == '__main__':
    args = add_args()
    logger = Logger('./eval.log')
    y_prob_list = []
    y_pLabel_list = []
    y_label_list = []
    lines_in_yProb = open(args.file_path,'r')
    lines_in_yLabel = open(args.label_path,'r')

    for line in lines_in_yProb:
        line = line.rstrip()
        split_str = line.split()
        y_prob = float(split_str[3])
        y_prob_list.append(y_prob)
        if y_prob > 0.5:
            y_pLabel_list.append(1)
        else:
            y_pLabel_list.append(0)
    
    for line in lines_in_yLabel:
        line = line.rstrip()
        split_str = line.split()
        y_label = float(split_str[3])
        y_label_list.append(y_label)
    
    eval_fun(y_prob_list,y_pLabel_list,y_label_list,logger)
