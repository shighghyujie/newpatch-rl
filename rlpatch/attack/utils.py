import os
import sys
import csv
import math
import torch
import numpy as np
import torch.nn as nn
import scipy.stats as st
import torch.nn.functional as F



# 高斯核
def get_gaussian_kernel(kernel_size=15, sigma=2, channels=3, use_cuda=False):
    x_coord = torch.arange(kernel_size)   # x坐标系
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()  # kernel_size*kernel_size*2
    mean = (kernel_size - 1)/2.   # 均值
    variance = sigma**2.          # 方差
    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) / (2*variance))  # 二维高斯函数
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)                           # 复制三次
    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, padding=(kernel_size-1)//2, bias=False)
    if use_cuda:
        gaussian_filter.weight.data = gaussian_kernel.cuda()   # 使用cuda
    else:
        gaussian_filter.weight.data = gaussian_kernel          # 利用高斯函数值更新卷积核的权重
    gaussian_filter.weight.requires_grad = False               # 不更新高斯核的参数
    return gaussian_filter


# 加载每张图片的真实标签信息
def load_ground_truth(csv_filename):
    image_id_list = []
    label_ori_list = []
    with open(csv_filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            image_id_list.append(row['ImageId'] )          # 图片id
            label_ori_list.append(int(row['TrueLabel']))   # 该图片对应的label
    return image_id_list, label_ori_list


# 图片标准化处理
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)  # 均值
        self.std = torch.Tensor(std)    # 标准差
    def forward(self, x):
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]


# TI核 定义
def gkern(kernlen=15, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


# DI ： input diversity
def DI(X_in, input_size):
    max_size = int(input_size * 1.1)   # 图片原尺度放大的最大尺度
    rnd=np.random.randint(input_size, max_size,size=1)[0]  # 随机一个尺度大小
    h_rem = max_size - rnd             # 长
    w_rem = max_size - rnd             # 宽
    pad_top = np.random.randint(0, h_rem,size=1)[0]   # 上下padding的起止位置
    pad_bottom = h_rem - pad_top
    pad_left = np.random.randint(0, w_rem,size=1)[0]  # 左右padding的起止位置
    pad_right = w_rem - pad_left
    c=np.random.rand(1)                # 以一定概率执行input diversity
    if c<=0.7:
        X_out=F.pad(F.interpolate(X_in, size=(rnd,rnd)),(pad_left,pad_top,pad_right,pad_bottom),mode='constant', value=0)  # 填0
        return X_out
    else:
        return X_in


# clip操作
def clip_by_tensor(t, t_min, t_max):
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result