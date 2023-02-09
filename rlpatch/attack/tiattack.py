import os
import cv2
import torch
from torch import tensor
from torch.autograd.grad_mode import no_grad
import torch.nn as nn
import numpy as np
import time
import joblib
from config import Config
from PIL import Image
from matplotlib import pyplot as plt
from torch.nn import DataParallel
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from facenet_pytorch import MTCNN, InceptionResnetV1
import argparse
from attack.utils import load_ground_truth, Normalize, gkern, DI, get_gaussian_kernel
from models import *
from attack import stick
from mtcnn_pytorch_master.test import crop_face
from tqdm import tqdm

import sys
sys.path.append("../")
# from Loss_Modifier import NPSCalculator,TotalVariation
import random

trans = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
            ])

            
#localtime = time.asctime( time.localtime(time.time()) )
inputsize = {'arcface50':[112,112],'cosface50':[112,112],'arcface34':[112,112],'cosface34':[112,112],
             'facenet':[160,160],'insightface':[112,112],
             'sphere20a':[112,96],'re_sphere20a':[112,96],'mobilefacenet':[112,112]}

def DI(x, resize_rate=1.15, diversity_prob=0.7):
    assert resize_rate >= 1.0                                   # 随机放大的尺度上限
    assert diversity_prob >= 0.0 and diversity_prob <= 1.0      # 执行DI的概率
    img_size = x.shape[-1]                        # 获取输入图片的尺度
    img_resize = int(img_size * resize_rate)      # DI最大缩放尺度
    rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)    # 随机尺度
    rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)  # 双线性插值
    h_rem = img_resize - rnd                      # 需要填充的边界
    w_rem = img_resize - rnd                      # 需要填充的边界
    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)     # 顶部填充
    pad_bottom = h_rem - pad_top                                                        # 底部填充
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)    # 左边填充
    pad_right = w_rem - pad_left                                                        # 右边填充
    padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)  # 填充
    ret = padded if torch.rand(1) < diversity_prob else x
    return ret

def reward_slope(adv_face_ts, params_slove, sticker,device):
    advface_ts = adv_face_ts.to(device)
    x, y = params_slove[0]
    w, h = sticker.size
    advstk_ts = advface_ts[:,:,y:y+h,x:x+w]
    advstk_ts.data = advstk_ts.data.clamp(1/255.,224/255.)
    w = torch.arctanh(2*advstk_ts-1)
    x_wv = 1/2 - (torch.tanh(w)**2)/2
    mean_slope = torch.mean(x_wv)
    #print(w,x_wv)
    return mean_slope

def load_model(model_name, device):
    if(model_name == 'facenet'):
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        return resnet
    elif (model_name == 'insightface'):
        insightface_path = 'stmodels/insightface/insightface.pth'
        model = Backbone(50,0.6,'ir_se')
        model.load_state_dict(torch.load(eval("{}_path".format(model_name)),map_location=device))
        model.eval()
        model = model.to(device)
        return model
    elif (model_name == 'sphere20a'):
        sphere20a_path = 'stmodels/sphere20a/sphere20a.pth'
        model = sphere20a(feature=True)
        model.load_state_dict(torch.load(eval("{}_path".format(model_name)),map_location=device))
        model.eval()
        model = model.to(device)
        return model
    elif (model_name == 're_sphere20a'):
        sphere20a_path = 'stmodels/re_sphere20a/re_sphere20a.pth'
        model = sphere20a(feature=True)
        #model.load_state_dict(torch.load(eval("{}_path".format(model_name)),map_location=device))
        model.eval()
        model = model.to(device)
        return model
    elif (model_name == 'mobilefacenet'):
        mobilefacenet_path = 'stmodels/mobilefacenet/mobilefacenet_scripted.pt'
        #model = MobileFaceNet()
        #model.load_state_dict(torch.load(eval("{}_path".format(model_name)),map_location=device))
        model = torch.jit.load(eval("{}_path".format(model_name)),map_location=device)
        model.eval()
        model = model.to(device)
        return model
    elif (model_name == 'arcface34'):
        arcface34_path = 'stmodels/arcface34/arcface_34.pth'
        model = iresnet34(False, dropout=0, fp16=True)
        model.load_state_dict(torch.load(eval("{}_path".format(model_name)),map_location=device))
        model.eval()
        model = model.to(device)
        return model
    elif (model_name == 'cosface34'):
        cosface34_path = 'stmodels/cosface34/cosface_34.pth'
        model = iresnet34(False, dropout=0, fp16=True)
        model.load_state_dict(torch.load(eval("{}_path".format(model_name)),map_location=device))
        model.eval()
        model = model.to(device)
        return model
    elif (model_name == 'tencent'):
        return None
    else:
        arcface50_path = 'stmodels/arcface50/ms1mv3_arcface_r50_fp16.pth'
        cosface50_path = 'stmodels/cosface50/glint360k_cosface_r50_fp16_0.1.pth'
        model = iresnet50(False, dropout=0, fp16=True)
        model.load_state_dict(torch.load(eval("{}_path".format(model_name)),map_location=device))
        model.eval()
        model = model.to(device)
        return model

def load_anchors(model_name, device, target):
    anchor_embeddings =  joblib.load('stmodels/{}/embeddings_{}.pkl'.format(model_name,model_name))
    anchor = anchor_embeddings[target:target+1]
    anchor = anchor.to(device)
    return anchor

def make_stmask(face,sticker,x,y):
    w,h = face.size
    mask = stick.make_masktensor(w,h,sticker,x,y)
    return mask

def crop_imgs(imgs,w,h):
    crops_result = []
    crops_tensor = []
    if len(imgs)>1:
        for i in tqdm(range(len(imgs))):
            crop = crop_face(imgs[i],w,h)
            crop_ts = trans(crop)
            crops_result.append(crop)
            crops_tensor.append(crop_ts)
    else:
        for i in range(len(imgs)):
            crop = crop_face(imgs[i],w,h)
            crop_ts = trans(crop)
            crops_result.append(crop)
            crops_tensor.append(crop_ts)
    return crops_result, crops_tensor

def cosin_metric(prd,src,device):
    nlen = len(prd)
    mlt = torch.zeros((nlen,1)).to(device)
    src_t = torch.t(src)
    #print(prd.shape,src_t.shape)
    for i in range(nlen):
        #print(prd[i].shape,src_t[:,i].shape)
        mlt[i] = torch.mm(torch.unsqueeze(prd[i],0),torch.unsqueeze(src_t[:,i],1))
    norm_x1 = torch.norm(prd,dim=1)
    norm_x1 = torch.unsqueeze(norm_x1,1)
    norm_x2 = torch.norm(src_t,dim=0)
    norm_x2 = torch.unsqueeze(norm_x2,1)
    #print('norm_x1,norm_x2 ',norm_x1.shape,norm_x2.shape)
    denominator = torch.mul(norm_x1, norm_x2)
    metrics = torch.mul(mlt,1/denominator)
    return metrics

def tiattack_face(x, y, epsilon, weights, model_names,
                  img, label, target, device, sticker,
                  width, height, emp_iterations, di, adv_img_folder, targeted = True):
    flag = -1 if targeted else 1
    liner_interval = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    # TI 参数设置
    channels=3                                             # 3通道
    kernel_size=5                                          # kernel大小
    kernel = gkern(kernel_size, 1).astype(np.float32)      # 3表述kernel内元素值得上下限
    gaussian_kernel = np.stack([kernel, kernel, kernel])   # 5*5*3
    gaussian_kernel = np.expand_dims(gaussian_kernel, 1)   # 1*5*5*3
    gaussian_kernel = torch.from_numpy(gaussian_kernel).to(device)  # tensor and cuda
    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=kernel_size, groups=channels, bias=False, padding=7)
    gaussian_filter.weight.data = gaussian_kernel          # 高斯滤波，高斯核的赋值
    
    crops_result, crops_tensor = crop_imgs([img], width, height)
    X_ori = torch.stack(crops_tensor).to(device)
    #print(X_ori.shape)
    delta = torch.zeros_like(X_ori,requires_grad=True).to(device)
    #label = torch.tensor(label).to(device)

    fr_models, anchors = [], []
    for name in model_names:
        model = load_model(name, device)
        anchor = load_anchors(name, device, target)
        fr_models.append(model)
        anchors.append(anchor)
    #print(anchors.shape)
    mask = make_stmask(crops_result[0],sticker,x,y)

    for itr in range(emp_iterations):
        g_temp = []
        for t in range(len(liner_interval)):
            c = liner_interval[t]
            X_adv = X_ori + c * delta
            accm = 0
            for (i, name) in enumerate(model_names):
                X_op = nn.functional.interpolate(X_adv, (inputsize[name][0], inputsize[name][1]), mode='bilinear', align_corners=False)
                # if di:
                #     X_adv = X_ori + delta
                #     X_adv = DI(X_adv, 500)   # diverse input operation
                #     X_op = nn.functional.interpolate(X_adv, (inputsize[name][0], inputsize[name][1]), mode='bilinear', align_corners=False)
                feature = fr_models[i](X_op)
                l_sim = cosin_metric(feature,anchors[i],device)
                accm += l_sim * weights[i]
            #print('---iter {} interval {}--- loss = {}'.format(itr,t,loss))
            loss = flag * accm
            #print('---iter {} interval {}--- loss = {}'.format(itr,t,loss))
            loss.backward()
            
            # TI operation
            grad_c = delta.grad.clone()                        
            grad_c = F.conv2d(grad_c, gaussian_kernel, bias=None, stride=1, padding=(2,2), groups=3)
            #grad_a = grad_c / torch.mean(torch.abs(grad_c), (1, 2, 3), keepdim=True)+0.5*grad_momentum   # 1
            grad_a = grad_c
            # grad_momentum = grad_a
            g_temp.append(grad_a)
        g_syn = 0.0
        for j in range(9):
            g_syn += g_temp[j]
        g_syn = g_syn / 9.0
        delta.grad.zero_()
        # L-inf attack
        delta.data=delta.data-epsilon * torch.sign(g_syn)
        delta.data = delta.data * mask.to(device)
        #delta.data=delta.data.clamp(-args.linf_epsilon/255.,args.linf_epsilon/255.)
        delta.data=((X_ori+delta.data).clamp(0,1))-X_ori              # 噪声截取操作
        
        with torch.no_grad():
            X_adv = X_ori + delta
            accm = 0
            for (i, name) in enumerate(model_names):
                X_op = nn.functional.interpolate(X_adv, (inputsize[name][0], inputsize[name][1]), mode='bilinear', align_corners=False)
                feature = fr_models[i](X_op)
                l_sim = cosin_metric(feature,anchors[i],device).item()
                accm += l_sim * weights[i]
            print('---iter {} --- loss = {}'.format(itr,flag * accm))
    
    adv_final = (X_ori+delta)[0].cpu().detach().numpy()
    adv_final = (adv_final*255).astype(np.uint8)
    localtime1 = time.asctime( time.localtime(time.time()) )
    file_path = os.path.join(adv_img_folder, '{}.jpg'.format(localtime1))
    adv_x_255 = np.transpose(adv_final, (1, 2, 0))
    im = Image.fromarray(adv_x_255)
    im.save(file_path,quality=99)
    torch.cuda.empty_cache()

def getHW():
    row_color = []
    for i in range(30):
        row_color.append(random.randint(0,255)/255)
    row_color = torch.tensor(row_color)
    row_color = row_color.repeat(25,1)
    return row_color

def initRandPatch():
    color = torch.stack((getHW(),getHW(),getHW()),dim = 0)
    batch = torch.unsqueeze(color,dim = 0)
    return batch

def tensor2PIL(tensor): # 将tensor-> PIL
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image

def miattack_face(params_slove, model_names, fr_models,
                  img, label, target, device, sticker,
                  width, height, emp_iterations, di, adv_img_folder, targeted = True):
    mw = sticker.size[0]
    mh = sticker.size[1]
    x, y = params_slove[0]
    weights = params_slove[1]
    epsilon = params_slove[2]
    # nsig = params_slove[3]
    flag = 1 if targeted else -1
    w,h = img.size
    if(w!=width or h!=height):
        crops_result, crops_tensor = crop_imgs([img], width, height)
    else:
        crops_result = [img]
        crops_tensor = [trans(img)]
    X_ori = torch.stack(crops_tensor).to(device)
    

    #print(X_ori.shape)
    delta = torch.zeros_like(X_ori,requires_grad=True).to(device)

    with torch.no_grad():
        X_ori[0,:,y:y+mh,x:x+mw] = torch.zeros([3,mh,mw])

    anchors = []
    for name in model_names:
        # if name == "tencent":
        #     continue
        anchor = load_anchors(name, device, target)
        anchors.append(anchor)
        
    mask = make_stmask(crops_result[0],sticker,x,y)
    grad_momentum = 0
    for itr in range(emp_iterations):   # iterations in the generation of adversarial examples
        X_adv = X_ori + delta
        X_adv.retain_grad()
        accm = 0
        X_op = DI(X_adv)
        # print('---iter {}---'.format(itr),end=' ')
        for (i, name) in enumerate(model_names):
            # if name == "tencent":
            #     continue
            X_op = nn.functional.interpolate(X_op, (inputsize[name][0], inputsize[name][1]), mode='bilinear', align_corners=False)
            feature = fr_models[i](X_op)
            l_sim = cosin_metric(feature,anchors[i],device)
            # print(name,':','{:.4f}'.format(l_sim.item()),end=' ')
            accm += l_sim * weights[i]
            
        # print('---iter {} interval {}--- loss = {}'.format(itr,t,loss))
        #slope = reward_slope(X_adv,params_slove,sticker,device)
        # total_variation = TotalVariation().cuda()
        # tv = total_variation(delta[0])
        loss = flag * accm# + 0.3*slope# + 2.5*tv
        loss.backward()
        
        # MI operation
        grad_c = X_adv.grad.clone()  
        grad_a = grad_c / torch.mean(torch.abs(grad_c), (1, 2, 3), keepdim=True)+1.0*grad_momentum   # 1
        grad_momentum = grad_a
            
        X_adv.grad.zero_()
        X_adv.data=X_adv.data+epsilon * torch.sign(grad_momentum)* mask.to(device)
        #X_adv.data = X_adv.data 
        #delta.data=delta.data.clamp(-args.linf_epsilon/255.,args.linf_epsilon/255.)
        X_adv.data=X_adv.data.clamp(0,1)
        delta.data=X_adv-X_ori
        #将delta噪声进行平滑

        # 1.avg pooling
        # delta2 = torch.zeros(3,25,30)
        # delta2 = delta.data[0,:,y:y+25,x:x+30]
        # pool = nn.AvgPool2d(3,stride=1,padding=1)
        # delta2 = pool(delta2)
        # delta.data[0,:,y:y+25,x:x+30] = delta2
        # 2.maxPooling
        # delta2 = torch.zeros(3,25,30)
        # delta2 = delta.data[0,:,y:y+25,x:x+30]
        # pool = nn.MaxPool2d(3,stride=1,padding=1)
        # delta2 = pool(delta2)
        # delta.data[0,:,y:y+25,x:x+30] = delta2
        # 3.resize
        delta2 = torch.zeros(3,mh,mw)
        delta2 = delta.data[0,:,y:y+mh,x:x+mw]
        patch1 = torch.tensor(delta2)
        patch2 = patch1.cpu().numpy()
        patch2 = np.transpose(patch2,(2,1,0))
        patch3 = cv2.resize(patch2,(int(mh/2),int(mw/2)))
        patch3 = cv2.resize(patch3,(mh,mw))
        patch3 = np.transpose(patch3,(2,1,0))
        patch4 = torch.from_numpy(patch3)
        delta.data[0,:,y:y+mh,x:x+mw] = patch4
        #

    adv_face_ts = (X_ori+delta).cpu().detach()
    adv_final = (X_ori+delta)[0].cpu().detach().numpy()
    adv_final = (adv_final*255).astype(np.uint8)
    localtime2 = time.asctime( time.localtime(time.time()) )
    file_path = os.path.join(adv_img_folder, '{}.jpg'.format(localtime2))
    adv_x_255 = np.transpose(adv_final, (1, 2, 0))
    im = Image.fromarray(adv_x_255)
    #im.save(file_path,quality=99)
    #torch.cuda.empty_cache()
    return adv_face_ts,im,mask

def get_kernel(kernlen=15, nsig=3):
    import scipy.stats as st
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

def TI_kernel(nsig):
    kernel_size = 3                                   # kernel size
    kernel = get_kernel(kernel_size, nsig).astype(np.float32)
    gaussian_kernel = np.stack([kernel, kernel, kernel])   # 5*5*3
    gaussian_kernel = np.expand_dims(gaussian_kernel, 1)   # 1*5*5*3
    gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()  # tensor and cuda
    return gaussian_kernel

# def check():
#     # faceimg = Image.open('/home/lenovo/shighgh/newpatch_rl/code_rl/rlpatch/example/guoying/1103_gy.jpg')
#     # crops_result, crops_tensor = crop_imgs([faceimg],args.width, args.height)
#     crops_result = [Image.open('/home/lenovo/shighgh/newpatch_rl/code_rl/rlpatch/adv_imgs/0.jpg')]
#     anchor_embeddings =  joblib.load('/home/lenovo/shighgh/newpatch_rl/code_rl/rlpatch/stmodels/{}/embeddings_{}_5752.pkl'.format(args.source_model,args.source_model))
#     anchors = anchor_embeddings[5749:5750]
#     anchors = anchors.to(device)

#     intput = torch.unsqueeze(trans(crops_result[0]),0).to(device)
#     print(intput.shape)
#     intput = nn.functional.interpolate(intput, (inputsize[0], inputsize[1]), mode='bilinear', align_corners=False)            # 插值到224
#     feature = fr_model(intput)
#     m = cosin_metric(feature,anchors,device)
#     print(m)

if __name__=="__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parse_arguments()
    dataset = datasets.ImageFolder(args.input_dir)
    dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
    
    def collate_fn(x):
        return x
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    fr_model = load_model(args.source_model,device).eval().to(device)
    if(args.source_model == 'facenet'):
        inputsize = [160,160]
    else:
        inputsize = [112,112]
