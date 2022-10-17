from __future__ import division
import torch.nn as nn
import torch.nn.functional as F
import torch
from numpy.linalg import svd
from numpy.random import normal
from math import sqrt


class UNet(nn.Module):
    def __init__(self,inputdim = 160,sgmodel = 4,feature_dim=20):
        super(UNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 128, 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, sgmodel, 1)
        self.bn5 = nn.BatchNorm2d(sgmodel)

        self.fc = nn.Linear(inputdim * inputdim, feature_dim)
        self.last_bn = nn.BatchNorm1d(feature_dim)
        
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self._initialize_weights()

    def forward(self, xs):
        x1 = F.relu(self.bn1(self.conv1(xs)))
        x2 = F.relu(self.bn2(self.conv2(self.maxpool(x1))))
        x3 = F.relu(self.bn3(self.conv3(self.maxpool(x2))))
        x4 = F.relu(self.bn4(self.conv4(self.upsample(x3))))
        x5 = F.relu(self.conv5(self.upsample(x4)))  # x7in
        
        e1 = torch.softmax(x5,dim=1)           # (bt,n_models,h,w)
        e2 = torch.mean(e1,dim=1)
        e3 = e2.view(e2.size(0), -1)
        #print('e.shape = ',e1.shape,e2.shape,e3.shape)
        e4 = self.fc(e3)
        #print(e4.shape)
        #e5 = self.last_bn(e4)
        return self.bn5(x5),e4        

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class conv_bn_relu(nn.Module):
    def __init__(self, kernel, numIn, numOut, stride = 1):
        super(conv_bn_relu, self).__init__()
        pad = (kernel - 1)//2
        self.conv = nn.Conv2d(in_channels = numIn, out_channels = numOut, kernel_size = kernel, stride = stride, padding = pad, bias = False)
        self.bn = nn.BatchNorm2d(num_features = numOut)
        self.relu = nn.ReLU(inplace = True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# Residual Network 
class Residual(nn.Module):
    def __init__(self, numIn, numOut, stride = 1):
        super(Residual, self).__init__()
        self.convBlock = nn.Sequential(
            nn.Conv2d(in_channels = numIn, out_channels = numOut, kernel_size = 3, stride = stride, padding = 1, bias = False),
            nn.BatchNorm2d(num_features = numOut),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = numOut, out_channels = numOut, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(num_features = numOut),
            nn.ReLU(inplace = True)
        )
        self.skip = nn.Sequential()
        if numIn != numOut or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels= numIn, out_channels = numOut, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(num_features = numOut)
            )
    def forward(self, x):
        residual = x
        x = self.convBlock(x)
        residual = self.skip(residual)
        x = x + residual
        return x


class HourGlass(nn.Module):
    # num: iter times; dims: channel of feature map; modules: number of residual blocks
    def __init__(self, num, dims, modules):
        super(HourGlass, self).__init__()
        cur_mod = modules[0]
        next_mod = modules[1]
        cur_dim = dims[0]
        next_dim = dims[1]

        self.up1 = self.make_layer(cur_dim, cur_dim, cur_mod)
        self.max1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.low1 = self.make_layer(cur_dim, next_dim, cur_mod)
        if num > 1:
            self.low2 = HourGlass(num-1, dims[1:], modules[1:])
        else:
            self.low2 = self.make_layer(next_dim, next_dim, next_mod)
        self.low3 = self.make_layer_revr(next_dim, cur_dim, cur_mod)
        self.up2 = nn.UpsamplingNearest2d(scale_factor = 2)    
    def forward(self, x):
        up1 = self.up1(x)
        max1 = self.max1(x)
        low1 = self.low1(max1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return up1 + up2
    # Make Residual blocks needed in HourGlass
    def make_layer(self, in_dim, out_dim, modules):
        layers = [Residual(in_dim, out_dim)]
        for _ in range(1, modules):
            layers.append(Residual(out_dim, out_dim))
        return nn.Sequential(*layers)
    def make_layer_revr(self, in_dim, out_dim, modules):
        layers = []
        for _ in range(modules-1):
            layers.append(Residual(in_dim, in_dim))
        layers.append(Residual(in_dim, out_dim))
        return nn.Sequential(*layers)


class BackBone(nn.Module):
    def __init__(self, inputdim = 160, sgmodel = 4,feature_dim=20):
        super(BackBone, self).__init__()
        self.num = 4
        self.dims = [64, 64, 128, 128, 256]
        self.modules = [2, 2, 2, 2, 4]
        self.pre_layer = nn.Sequential(
            conv_bn_relu(7,  3, 64, stride=1)
        )   
        self.hg1 = HourGlass(self.num, self.dims, self.modules)
        self.location_head = nn.Sequential(
            conv_bn_relu(3, 64, 64, stride=1),
            conv_bn_relu(3, 64, 1, stride=1)
        )
        self.hg2 = HourGlass(self.num, self.dims, self.modules)   
        self.patch_head = nn.Sequential(
            conv_bn_relu(3, 64, sgmodel, stride=1)
        )
        self.fc = nn.Linear(inputdim * inputdim, feature_dim)
        

    def forward(self, x):
        pre = self.pre_layer(x)
        hg1 = self.hg1(pre)
        loc = self.location_head(hg1)
        hg2 = self.hg2(hg1)
        patch = self.patch_head(hg2)

        e1 = torch.softmax(patch,dim=1)           # (bt,n_models,h,w)
        e2 = torch.mean(e1,dim=1)
        e3 = e2.view(e2.size(0), -1)
        e4 = self.fc(e3)

        return loc, patch, e4


    



if __name__=="__main__":
    unet = UNet(inputdim = 160,sgmodel = 3,feature_dim=20)
    x = torch.Tensor(1,3,160,160)
    r,e = unet(x)
    print(r.shape,e.shape)
    print(e)
    eps_probs = torch.softmax(e,dim=1)      # (bt,eps_dim)
    print(eps_probs)
    print(eps_probs[0].shape)
