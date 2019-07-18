import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init
from torch.autograd import Variable
from layer_types import MaskedLinear_br, MaskedConv2d_br, MaskedBatchNorm_br

import torch.utils.model_zoo as model_zoo


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
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = MaskedConv2d_br(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = MaskedBatchNorm_br(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = MaskedConv2d_br(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = MaskedBatchNorm_br(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = MaskedConv2d_br(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = MaskedBatchNorm_br(planes)
        self.conv2 = MaskedConv2d_br(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = MaskedBatchNorm_br(planes)
        self.conv3 = MaskedConv2d_br(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = MaskedBatchNorm_br(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = MaskedConv2d_br(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = MaskedBatchNorm_br(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = MaskedLinear_br(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                MaskedConv2d_br(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                MaskedBatchNorm_br(planes * block.expansion),
            )


        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        print("self.inplanes",self.inplanes)
        for _ in range(1, blocks):
            print("self.inplanes",self.inplanes)
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def set_masks(self,masks):
        # Laziest way every to do it :D
        
        self.conv1.set_mask(masks[0])
        self.bn1.set_mask(masks[1])
        
        self.layer1[0].conv1.set_mask(masks[2])
        self.layer1[0].bn1.set_mask(masks[3])
        self.layer1[0].conv2.set_mask(masks[4])
        self.layer1[0].bn2.set_mask(masks[5])
        self.layer1[0].conv3.set_mask(masks[6])
        self.layer1[0].bn3.set_mask(masks[7])
        self.layer1[0].downsample[0].set_mask(masks[8])
        self.layer1[0].downsample[1].set_mask(masks[9])
        
        self.layer1[1].conv1.set_mask(masks[10])
        self.layer1[1].bn1.set_mask(masks[11])
        self.layer1[1].conv2.set_mask(masks[12])
        self.layer1[1].bn2.set_mask(masks[13])
        self.layer1[1].conv3.set_mask(masks[14])
        self.layer1[1].bn3.set_mask(masks[15])
        
        self.layer1[2].conv1.set_mask(masks[16])
        self.layer1[2].bn1.set_mask(masks[17])
        self.layer1[2].conv2.set_mask(masks[18])
        self.layer1[2].bn2.set_mask(masks[19])        
        self.layer1[2].conv3.set_mask(masks[20])
        self.layer1[2].bn3.set_mask(masks[21]) 
        
        
        self.layer2[0].conv1.set_mask(masks[22])
        self.layer2[0].bn1.set_mask(masks[23])
        self.layer2[0].conv2.set_mask(masks[24])
        self.layer2[0].bn2.set_mask(masks[25])
        self.layer2[0].conv3.set_mask(masks[26])
        self.layer2[0].bn3.set_mask(masks[27])
        self.layer2[0].downsample[0].set_mask(masks[28])
        self.layer2[0].downsample[1].set_mask(masks[29])
        
        self.layer2[1].conv1.set_mask(masks[30])
        self.layer2[1].bn1.set_mask(masks[31])
        self.layer2[1].conv2.set_mask(masks[32])
        self.layer2[1].bn2.set_mask(masks[33])
        self.layer2[1].conv3.set_mask(masks[34])
        self.layer2[1].bn3.set_mask(masks[35])
        
        self.layer2[2].conv1.set_mask(masks[36])
        self.layer2[2].bn1.set_mask(masks[37])
        self.layer2[2].conv2.set_mask(masks[38])
        self.layer2[2].bn2.set_mask(masks[39])
        self.layer2[2].conv3.set_mask(masks[40])
        self.layer2[2].bn3.set_mask(masks[41])
        
        self.layer2[3].conv1.set_mask(masks[42])
        self.layer2[3].bn1.set_mask(masks[43])
        self.layer2[3].conv2.set_mask(masks[44])
        self.layer2[3].bn2.set_mask(masks[45])
        self.layer2[3].conv3.set_mask(masks[46])
        self.layer2[3].bn3.set_mask(masks[47])
        
        
        self.layer3[0].conv1.set_mask(masks[48])
        self.layer3[0].bn1.set_mask(masks[49])
        self.layer3[0].conv2.set_mask(masks[50])
        self.layer3[0].bn2.set_mask(masks[51])
        self.layer3[0].conv3.set_mask(masks[52])
        self.layer3[0].bn3.set_mask(masks[53])
        self.layer3[0].downsample[0].set_mask(masks[54])
        self.layer3[0].downsample[1].set_mask(masks[55])
        
        self.layer3[1].conv1.set_mask(masks[56])
        self.layer3[1].bn1.set_mask(masks[57])
        self.layer3[1].conv2.set_mask(masks[58])
        self.layer3[1].bn2.set_mask(masks[59])
        self.layer3[1].conv3.set_mask(masks[60])
        self.layer3[1].bn3.set_mask(masks[61])
        
        self.layer3[2].conv1.set_mask(masks[62])
        self.layer3[2].bn1.set_mask(masks[63])
        self.layer3[2].conv2.set_mask(masks[64])
        self.layer3[2].bn2.set_mask(masks[65])
        self.layer3[2].conv3.set_mask(masks[66])
        self.layer3[2].bn3.set_mask(masks[67])
        
        self.layer3[3].conv1.set_mask(masks[68])
        self.layer3[3].bn1.set_mask(masks[69])
        self.layer3[3].conv2.set_mask(masks[70])
        self.layer3[3].bn2.set_mask(masks[71])  
        self.layer3[3].conv3.set_mask(masks[72])
        self.layer3[3].bn3.set_mask(masks[73]) 
        
        self.layer3[4].conv1.set_mask(masks[74])
        self.layer3[4].bn1.set_mask(masks[75])
        self.layer3[4].conv2.set_mask(masks[76])
        self.layer3[4].bn2.set_mask(masks[77])
        self.layer3[4].conv3.set_mask(masks[78])
        self.layer3[4].bn3.set_mask(masks[79])
        
        self.layer3[5].conv1.set_mask(masks[80])
        self.layer3[5].bn1.set_mask(masks[81])
        self.layer3[5].conv2.set_mask(masks[82])
        self.layer3[5].bn2.set_mask(masks[83])
        self.layer3[5].conv3.set_mask(masks[84])
        self.layer3[5].bn3.set_mask(masks[85])
    
        
        self.layer4[0].conv1.set_mask(masks[86])
        self.layer4[0].bn1.set_mask(masks[87])
        self.layer4[0].conv2.set_mask(masks[88])
        self.layer4[0].bn2.set_mask(masks[89])
        self.layer4[0].conv3.set_mask(masks[90])
        self.layer4[0].bn3.set_mask(masks[91])
        self.layer4[0].downsample[0].set_mask(masks[92])
        self.layer4[0].downsample[1].set_mask(masks[93])
        
        self.layer4[1].conv1.set_mask(masks[94])
        self.layer4[1].bn1.set_mask(masks[95])
        self.layer4[1].conv2.set_mask(masks[96])
        self.layer4[1].bn2.set_mask(masks[97])
        self.layer4[1].conv3.set_mask(masks[98])
        self.layer4[1].bn3.set_mask(masks[99])
        
        self.layer4[2].conv1.set_mask(masks[100])
        self.layer4[2].bn1.set_mask(masks[101])
        self.layer4[2].conv2.set_mask(masks[102])
        self.layer4[2].bn2.set_mask(masks[103])
        self.layer4[2].conv3.set_mask(masks[104])
        self.layer4[2].bn3.set_mask(masks[105])
        
        self.fc.set_mask(masks[106])    

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model