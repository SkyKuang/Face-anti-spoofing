from PIL import Image
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms 
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler
import pdb

class CISIA_SURF(Dataset):
    def __init__(self, root = '/home/kuanghuafeng/datasets/CASIA-SURF', datatxt='train_list.txt', transform=None):
        self.root = root 
        self.transform = transform
        self.img_paths = []
        self.rgb_paths = []
        self.depth_paths = []
        self.ir_paths = []
        self.labels = []

        lines_in_txt = open(os.path.join(root,datatxt),'r')

        for line in lines_in_txt:
            line = line.rstrip() 
            split_str = line.split()
            rgb_path = os.path.join(root,split_str[0])
            depth_path = os.path.join(root,split_str[1])
            ir_path = os.path.join(root,split_str[2])
            label = split_str[3]
            self.rgb_paths.append(rgb_path)
            self.depth_paths.append(depth_path)
            self.ir_paths.append(ir_path)
            self.labels.append(label)

    def __getitem__(self,index):
        rgb_path = self.rgb_paths[index]
        depth_path = self.depth_paths[index]
        ir_path = self.ir_paths[index]
        rgb_img = Image.open(rgb_path).convert('RGB')
        depth_img = Image.open(depth_path).convert('RGB')
        ir_img = Image.open(ir_path).convert('RGB')

        hsv_img_cv = cv2.cvtColor(np.asarray(rgb_img), cv2.COLOR_RGB2HSV)
        hsv_img = Image.fromarray(hsv_img_cv)
        YCbCr_img = rgb_img.convert('YCbCr')

        if self.transform is not None:
            rgb_img = self.transform(rgb_img)
            depth_img = self.transform(depth_img)
            ir_img = self.transform(ir_img)

            hsv_img = self.transform(hsv_img)
            YCbCr_img = self.transform(YCbCr_img)
        label = torch.as_tensor(int(self.labels[index]))
        
        return rgb_img,depth_img,ir_img,hsv_img,YCbCr_img,label

    def __len__(self):
        return len(self.rgb_paths)


def load_cisia_surf(train_size=128, test_size=64):
    
    train_transforms = transforms.Compose([
        transforms.Resize((124,124)),
        transforms.CenterCrop((112,112)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((124,124)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(112, padding=5),
        transforms.ToTensor(),
        # transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    train_data=CISIA_SURF(datatxt='train_list.txt', transform=train_transforms)
    val_data=CISIA_SURF(datatxt='val_private_list.txt', transform=train_transforms)
    test_data=CISIA_SURF(datatxt='test_private_list.txt', transform=train_transforms)

    train_loader = DataLoader(dataset=train_data, batch_size=train_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=val_data, batch_size=test_size, shuffle = False, num_workers=4)
    test_loader = DataLoader(dataset=test_data, batch_size=test_size, shuffle = False, num_workers=4)
        
    return train_loader, val_loader, test_loader


