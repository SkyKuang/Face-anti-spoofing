import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
from torchvision.models import ResNet
import torchvision.models as torch_models
import pdb
import torch.nn as nn
import torch.nn.init as init
import math
# from utee import misc
from collections import OrderedDict

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    # "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        m = OrderedDict()
        m['conv1'] = conv3x3(inplanes, planes, stride)
        m['bn1'] = nn.BatchNorm2d(planes)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = conv3x3(planes, planes)
        m['bn2'] = nn.BatchNorm2d(planes)
        self.group1 = nn.Sequential(m)

        self.relu= nn.Sequential(nn.ReLU(inplace=True))
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.group1(x) + residual

        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64
        super(ResNet, self).__init__()

        m = OrderedDict()
        m['conv1'] = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m['bn1'] = nn.BatchNorm2d(64)
        m['relu1'] = nn.ReLU(inplace=True)
        m['maxpool'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.group1= nn.Sequential(m)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.Sequential(nn.AvgPool2d(7))

        self.group2 = nn.Sequential(
            OrderedDict([
                ('fc', nn.Linear(512 * block.expansion, num_classes))
            ])
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def _make_layer(self, block, planes, blocks, stride=1):
        if planes == 512:
            self.inplanes = 256*5
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.group1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.group2(x)

        return x


def resnet34(pretrained=False, model_root=None, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        misc.load_state_dict(model, model_urls['resnet34'], model_root)
    return model


class SElayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SElayer,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel,channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b,c,_,_ = x.size()
        y = self.avg_pool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        return x*y.expand_as(x)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)

class Model(nn.Module):
    def __init__(self,pretrained=False, num_classes=2):
        super(Model,self).__init__()

        self.rgb_resnet = torch_models.resnet34(pretrained)
        self.depth_resnet = torch_models.resnet34(pretrained)
        self.ir_resnet = torch_models.resnet34(pretrained)
        self.hsv_resnet = torch_models.resnet34(pretrained)
        self.ycb_resnet = torch_models.resnet34(pretrained)
        self.main_resnet = resnet34(num_classes=2)
    
        self.rgb_layer0 = nn.Sequential(
            self.rgb_resnet.conv1,
            self.rgb_resnet.bn1,
            self.rgb_resnet.relu,
            self.rgb_resnet.maxpool
        )

        self.rgb_layer1 = self.rgb_resnet.layer1
        self.rgb_layer2 = self.rgb_resnet.layer2
        self.rgb_layer3 = self.rgb_resnet.layer3
        self.rgb_selayer = SElayer(256)

        self.depth_layer0 = nn.Sequential(
            self.depth_resnet.conv1,
            self.depth_resnet.bn1,
            self.depth_resnet.relu,
            self.depth_resnet.maxpool
        )
        self.depth_layer1 = self.depth_resnet.layer1
        self.depth_layer2 = self.depth_resnet.layer2
        self.depth_layer3 = self.depth_resnet.layer3
        self.depth_selayer = SElayer(256)


        self.ir_layer0 = nn.Sequential(
            self.ir_resnet.conv1,
            self.ir_resnet.bn1,
            self.ir_resnet.relu,
            self.ir_resnet.maxpool
        )
        self.ir_layer1 = self.ir_resnet.layer1
        self.ir_layer2 = self.ir_resnet.layer2
        self.ir_layer3 = self.ir_resnet.layer3
        self.ir_selayer = SElayer(256)

        self.hsv_layer0 = nn.Sequential(
            self.hsv_resnet.conv1,
            self.hsv_resnet.bn1,
            self.hsv_resnet.relu,
            self.hsv_resnet.maxpool
        )
        self.hsv_layer1 = self.hsv_resnet.layer1
        self.hsv_layer2 = self.hsv_resnet.layer2
        self.hsv_layer3 = self.hsv_resnet.layer3
        self.hsv_selayer = SElayer(256)

        self.ycb_layer0 = nn.Sequential(
            self.ycb_resnet.conv1,
            self.ycb_resnet.bn1,
            self.ycb_resnet.relu,
            self.ycb_resnet.maxpool
        )
        self.ycb_layer1 = self.ycb_resnet.layer1
        self.ycb_layer2 = self.ycb_resnet.layer2
        self.ycb_layer3 = self.ycb_resnet.layer3
        self.ycb_selayer = SElayer(256)

        self.main_layer4 = self.main_resnet.layer4

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512,2)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # if m.bias is not None:
                    # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                # m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                # m.bias.data.zero_()

    def forward(self, rgb_img, depth_img, ir_img, hsv_img, ycb_img):

        rgb_feat0 = self.rgb_layer0(rgb_img)
        rgb_feat1 = self.rgb_layer1(rgb_feat0)
        rgb_feat2 = self.rgb_layer2(rgb_feat1)
        rgb_feat3 = self.rgb_layer3(rgb_feat2)
        rgb_feat4 = self.rgb_selayer(rgb_feat3)

        depth_feat0 = self.depth_layer0(depth_img)
        depth_feat1 = self.depth_layer1(depth_feat0)
        depth_feat2 = self.depth_layer2(depth_feat1)
        depth_feat3 = self.depth_layer3(depth_feat2)
        depth_feat4 = self.depth_selayer(depth_feat3)

        ir_feat0 = self.ir_layer0(ir_img)
        ir_feat1 = self.ir_layer1(ir_feat0)
        ir_feat2 = self.ir_layer2(ir_feat1)
        ir_feat3 = self.ir_layer3(ir_feat2)
        ir_feat4 = self.ir_selayer(ir_feat3)

        hsv_feat0 = self.hsv_layer0(hsv_img)
        hsv_feat1 = self.hsv_layer1(hsv_feat0)
        hsv_feat2 = self.hsv_layer2(hsv_feat1)
        hsv_feat3 = self.hsv_layer3(hsv_feat2)
        hsv_feat4 = self.hsv_selayer(hsv_feat3)

        ycb_feat0 = self.ycb_layer0(ycb_img)
        ycb_feat1 = self.ycb_layer1(ycb_feat0)
        ycb_feat2 = self.ycb_layer2(ycb_feat1)
        ycb_feat3 = self.ycb_layer3(ycb_feat2)
        ycb_feat4 = self.ycb_selayer(ycb_feat3)

        main_feat = torch.cat((rgb_feat4,depth_feat4),1)
        main_feat = torch.cat((main_feat,ir_feat4),1)
        main_feat = torch.cat((main_feat,hsv_feat4),1)
        main_feat = torch.cat((main_feat,ycb_feat4),1)

        main_feat = self.main_layer4(main_feat)
        main_feat = self.avg_pool(main_feat)
        
        main_fc = main_feat.view(main_feat.size(0),-1)
        out = self.fc(main_fc)

        return out

