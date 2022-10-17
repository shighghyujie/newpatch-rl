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
        # self.conv5_1 = nn.Conv2d(128, 64, 1)
        self.conv5_1 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(64)
        self.conv6_1 = nn.Conv2d(64, 1, 1)
        self.bn6_1 = nn.BatchNorm2d(1)
        self.conv5_2 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(256)
        # self.conv6_2 = nn.Conv2d(256, 128, 1)
        # self.bn6_2 = nn.BatchNorm2d(128)
        self.conv7_2 = nn.Conv2d(256, sgmodel, 1)
        self.bn7_2 = nn.BatchNorm2d(sgmodel)

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
        x5_1 = F.relu(self.bn5_1(self.conv5_1(self.upsample(x4))))
        x6_1 = F.relu(self.conv6_1(x5_1))
        # x6_1 = F.relu((self.conv6_1(x5_1)))
        x5_2 = F.relu(self.bn5_2(self.conv5_2(self.upsample(x4))))  # x7in
        # x6_2 = F.relu(self.bn6_2(self.conv6_2(x5_2)))
        x7_2 = F.relu(self.conv7_2(x5_2))

        
        e1 = torch.softmax(x5_2,dim=1)           # (bt,n_models,h,w)
        e2 = torch.mean(e1,dim=1)
        e3 = e2.view(e2.size(0), -1)
        #print('e.shape = ',e1.shape,e2.shape,e3.shape)
        e4 = self.fc(e3)
        #print(e4.shape)
        #e5 = self.last_bn(e4)
        return self.bn6_1(x6_1), self.bn7_2(x7_2),e4


    def forward2(self, xs):
        x1 = F.relu(self.bn1(self.conv1(xs)))
        x2 = F.relu(self.bn2(self.conv2(self.maxpool(x1))))
        x3 = F.relu(self.bn3(self.conv3(self.maxpool(x2))))
        x4 = F.relu(self.bn4(self.conv4(self.upsample(x3))))

        # 
        x5a = F.relu(self.bn5a(self.conv5a(self.upsample(x4))))
        x5b = F.relu(self.bn5b(self.conv5b(self.upsample(x4))))
        # 
        x5 = F.relu(self.conv5(self.upsample(x4)))  # x7in
        
        e1 = torch.softmax(x5,dim=1)           # (bt,n_models,h,w)
        e2 = torch.mean(e1,dim=1)
        e3 = e2.view(e2.size(0), -1)
        #print('e.shape = ',e1.shape,e2.shape,e3.shape)
        e4 = self.fc(e3)
        #print(e4.shape)
        #e5 = self.last_bn(e4)
        return self.bn5(x5),e4

        # x1 = F.relu(self.conv1(xs))
        # x2 = F.relu(self.conv2(self.maxpool(x1)))
        # x3 = F.relu(self.conv3(self.maxpool(x2)))
        # x4 = F.relu(self.conv4(self.upsample(x3)))
        # x5 = F.relu(self.conv5(self.upsample(x4))) # x7in
        # return x5

        # print('x1 size: %d'%(x1.size(2)))
        # print('x2 size: %d'%(x2.size(2)))
        # print('x3 size: %d'%(x3.size(2)))
        # print('x4 size: %d'%(x4.size(2)))
        #return F.softsign(self.bn5(x5))
        

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

if __name__=="__main__":
    unet = UNet(inputdim = 160,sgmodel = 3,feature_dim=20)
    x = torch.Tensor(1,3,160,160)
    r,e = unet(x)
    print(r.shape,e.shape)
    print(e)
    eps_probs = torch.softmax(e,dim=1)      # (bt,eps_dim)
    print(eps_probs)
    print(eps_probs[0].shape)
