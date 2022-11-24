"""
:Author :GodofTheFallen
:Time:  :2022/11/13
:File   :ccnet.py
:content:CCNet

:Reference: git@github.com:speedinghzl/CCNet
"""

import jittor as jt
from jittor import nn

from cc_attention import CrissCrossAttention

affine_par = True

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = nn.BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm(planes * 4)
        self.relu = nn.ReLU() # ReLU in jittor has no key word inplace
        self.relu_inplace = nn.ReLU()
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def execute(self, x):
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

        out = out + residual      
        out = self.relu_inplace(out)

        return out


class RCCAModule(jt.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(RCCAModule, self).__init__()
        inter_channels = in_channels // 4
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm(inter_channels))
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm(inter_channels))
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm(out_channels),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
    
    def execute(self, x, recurrence=1):
        output = self.conva(x)
        for i in range(recurrence):
            output = self.cca(output)
        output = self.convb(output)
        
        output = self.bottleneck(jt.concat([x, output], 1))
        return output


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, criterion, recurrence):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm(64)
        self.relu1 = nn.ReLU()
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm(64)
        self.relu2 = nn.ReLU()
        self.conv3 = conv3x3(64, 128)
        self.bn3 = nn.BatchNorm(128)
        self.relu3 = nn.ReLU()
        self.maxpool = nn.Pool(kernel_size=3, stride=2, padding=1, op='maximum')

        self.relu = nn.ReLU()
        self.maxpool = nn.Pool(kernel_size=3, stride=2, padding=1, ceil_mode=True, op='maximum')
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=(1, 1, 1))
        self.head = RCCAModule(2048, 512, num_classes)

        self.dsn = nn.Sequential(
            nn.Conv(1024, 512, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm(512),
            nn.Dropout(p=0.1), 
            nn.Conv(512, num_classes, 1, stride=1, padding=0, bias=True)
            )
        self.criterion = criterion
        self.recurrence = recurrence

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if (stride != 1 or self.inplanes != planes * block.expansion):
            downsample = nn.Sequential(
                nn.Conv(self.inplanes, planes * block.expansion, 
                        kernel_size=1, stride=stride, bias=False), 
                nn.BatchNorm(planes * block.expansion, affine=affine_par))
        
        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))
        
        return nn.Sequential(*layers)

    def execute(self, x, labels=None):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_dsn = self.dsn(x)
        x = self.layer4(x)
        x = self.head(x, self.recurrence)
        outs = [x, x_dsn]
        if self.criterion is not None and labels is not None:
            return self.criterion(outs, labels)
        else:
            return outs


def load_pretrained_model(model, params_path):
    """Reference: https://github.com/Jittor/deeplab-jittor/blob/master/backbone.py"""
    pretrained_dict = jt.load(params_path)
    model_dict = {}
    param_name = model.parameters()
    name_list = [item.name() for item in param_name]
    for k, v in pretrained_dict.items():
        if k in name_list:
            model_dict[k] = v

    # check shape
    # for k, v in model_dict.items():
    #     if v.shape != model.state_dict()[k].shape:
    #         print('Shape mismatch: ', k, v.shape, model.state_dict()[k].shape)
    #         model_dict[k] = model.state_dict()[k]
    #     else:
    #         print('Shape match: ', k, v.shape, model.state_dict()[k].shape)
    model.load_parameters(model_dict)


def Seg_Model(num_classes, criterion=None, pretrained_model=None, recurrence=0, **kwargs):
    model = ResNet(Bottleneck,[3, 4, 23, 3], num_classes, criterion, recurrence)

    if pretrained_model is not None:
        # model.load_parameters(jt.load(pretrained_model))
        model.load(pretrained_model) # more concise
        # load_pretrained_model(model, pretrained_model)

    return model
