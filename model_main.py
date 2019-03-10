import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
from torchvision.models import ResNet
import torchvision.models as models
import pdb

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class QAN(nn.Module):
    def __init__(self):
        super(QAN,self).__init__()
        self.qan_block_1 = nn.Sequential(
            nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2),
        )
        self.qan_block_2 = nn.Sequential(
            nn.Conv2d(64,32,kernel_size=1,stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(7,stride=1),
        )
        self.qan_block_fc = nn.Linear(32,1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,feat):
        qan_feat = feat
        qan_feat = self.qan_block_1(qan_feat)
        qan_feat = self.qan_block_2(qan_feat)
        qan_fc_feat = qan_feat.view(qan_feat.size(0),-1)
        u = self.qan_block_fc(qan_fc_feat)
        u = self.sigmoid(u)
        u = u.view(u.size(0),u.size(1),1,1)
        return u
    
class Model(nn.Module):
    def __init__(self, pretrained=False, num_classes=10):
        super(Model, self).__init__()
        # original Resnet
        self.rgb_resnet = models.resnet34(pretrained)
        self.depth_resnet = models.resnet34(pretrained)
        self.ir_resnet = models.resnet34(pretrained)
        self.hsv_resnet = models.resnet34(pretrained)
        self.YCbCr_resnet = models.resnet34(pretrained)
        self.resnet = models.resnet34(pretrained)

        self.rgb_layer0 = nn.Sequential(
            self.rgb_resnet.conv1,
            self.rgb_resnet.bn1,
            self.rgb_resnet.relu,
            self.rgb_resnet.maxpool
        )

        self.rgb_layer1 = self.rgb_resnet.layer1
        self.rgb_layer2 = self.rgb_resnet.layer2
        self.rgb_selayer = SELayer(128)
        self.rgb_qan = QAN()

        self.depth_layer0 = nn.Sequential(
            self.depth_resnet.conv1,
            self.depth_resnet.bn1,
            self.depth_resnet.relu,
            self.depth_resnet.maxpool
        )
        self.depth_layer1 = self.depth_resnet.layer1
        self.depth_layer2 = self.depth_resnet.layer2
        self.depth_selayer = SELayer(128)
        self.depth_qan = QAN()

        self.ir_layer0 = nn.Sequential(
            self.ir_resnet.conv1,
            self.ir_resnet.bn1,
            self.ir_resnet.relu,
            self.ir_resnet.maxpool
        )
        self.ir_layer1 = self.ir_resnet.layer1
        self.ir_layer2 = self.ir_resnet.layer2
        self.ir_selayer = SELayer(128)
        self.ir_qan = QAN()
        
        self.hsv_layer0 = nn.Sequential(
            self.hsv_resnet.conv1,
            self.hsv_resnet.bn1,
            self.hsv_resnet.relu,
            self.hsv_resnet.maxpool
        )
        self.hsv_layer1 = self.hsv_resnet.layer1
        self.hsv_layer2 = self.hsv_resnet.layer2
        self.hsv_selayer = SELayer(128)
        self.hsv_qan = QAN()

        self.ycb_layer0 = nn.Sequential(
            self.YCbCr_resnet.conv1,
            self.YCbCr_resnet.bn1,
            self.YCbCr_resnet.relu,
            self.YCbCr_resnet.maxpool
        )
        self.ycb_layer1 = self.YCbCr_resnet.layer1
        self.ycb_layer2 = self.YCbCr_resnet.layer2
        self.ycb_selayer = SELayer(128)
        self.ycb_qan = QAN()
        
        self.qan = QAN()
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512,2)

        self.catConv = nn.Sequential(
            nn.Conv2d(640, 128, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, rgb_img,depth_img,ir_img,hsv_img,YCbCr_img,img_crops):
        rgb_feat = self.rgb_layer0(rgb_img)
        rgb_feat = self.rgb_layer1(rgb_feat)
        rgb_feat = self.rgb_layer2(rgb_feat)
        rgb_feat = self.rgb_selayer(rgb_feat)
        rgb_u = self.rgb_qan(rgb_feat)

        depth_feat = self.depth_layer0(depth_img)       
        depth_feat = self.depth_layer1(depth_feat)
        depth_feat = self.depth_layer2(depth_feat)
        depth_feat = self.rgb_selayer(depth_feat)
        depth_u = self.depth_qan(depth_feat)

        ir_feat = self.ir_layer0(ir_img)
        ir_feat = self.ir_layer1(ir_feat)
        ir_feat = self.ir_layer2(ir_feat)
        ir_feat = self.rgb_selayer(ir_feat)        
        ir_u = self.ir_qan(ir_feat)
        
        hsv_feat = self.hsv_layer0(hsv_img)
        hsv_feat = self.hsv_layer1(hsv_feat)
        hsv_feat = self.hsv_layer2(hsv_feat)
        hsv_feat = self.hsv_selayer(hsv_feat)        
        hsv_u = self.hsv_qan(hsv_feat)
        
        ycb_feat = self.ycb_layer0(YCbCr_img)
        ycb_feat = self.ycb_layer1(ycb_feat)
        ycb_feat = self.ycb_layer2(ycb_feat)
        ycb_feat = self.ycb_selayer(ycb_feat)        
        ycb_u = self.ycb_qan(ycb_feat)
        
        a,b,c,d,e = 1,3,2,1,2
        cat_feat = torch.cat((rgb_feat*a,depth_feat*b),1)
        cat_feat = torch.cat((cat_feat,ir_feat*c),1)
        cat_feat = torch.cat((cat_feat,hsv_feat*d),1)
        cat_feat = torch.cat((cat_feat,ycb_feat*e),1)

        cat_feat = self.catConv(cat_feat)  # 128 14 14
        cat_feat = self.layer3(cat_feat)   # 256 7 7
        cat_feat = self.layer4(cat_feat)   # 512 4 4
        gap_feat = self.avgpool(cat_feat)
        fc_feat = gap_feat.view(gap_feat.size(0), -1)
        out = self.fc(fc_feat)
        return fc_feat,out

def load_model(pretrained=False,num_classes=2):
    model = Model(pretrained,num_classes)
    return model
