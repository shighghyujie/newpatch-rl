import os
import cv2
import torch
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
from utils import load_ground_truth, Normalize, gkern, DI, get_gaussian_kernel
from models import *
import stick
from mtcnn_pytorch_master.test import crop_face

model_names = ['arcface','cosface','facenet']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
adv_img_folder = '/home/guoying/rlpatch/adv_imgs'
trans = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
            ])
sticker = Image.new('RGBA',(30,25),(255,255,255,255))
#sticker = Image.new('RGBA',(50,25),(255,255,255,255))

def load_model(model_name):
    if(model_name == 'facenet'):
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        return resnet
    else:
        arcface_path = '/home/guoying/rlpatch/stmodels/arcface/ms1mv3_arcface_r50_fp16.pth'
        cosface_path = '/home/guoying/rlpatch/stmodels/cosface/glint360k_cosface_r50_fp16_0.1.pth'
        model = iresnet50(False, dropout=0, fp16=True)
        model.load_state_dict(torch.load(eval("{}_path".format(model_name)),map_location=device))
        return model

def make_mask(face,sticker,x,y):
    w,h = face.size
    mask = stick.make_masktensor(w,h,sticker,x,y)
    return mask

def crop_imgs(imgs,w,h):
    crops_result = []
    crops_tensor = []
    for i in range(len(imgs)):
        crop = crop_face(imgs[i],w,h)
        crop_ts = trans(crop)
        crops_result.append(crop)
        crops_tensor.append(crop_ts)
    return crops_result, crops_tensor

def cosin_metric(prd,src):
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

def tiattack_face(data_all,fr_model):
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

    cnt = 0
    anchor_embeddings =  joblib.load('/home/guoying/rlpatch/stmodels/{}/embeddings_{}_5752.pkl'.format(args.source_model,args.source_model))
    for X in data_all:
        X = list(map(list, zip(*X)))
        imgs,label = X
        len_batch = len(imgs)
        crops_result, crops_tensor = crop_imgs(imgs, args.width, args.height)

        X_ori = torch.stack(crops_tensor).to(device)
        #print(X_ori.shape)
        delta = torch.zeros_like(X_ori,requires_grad=True).to(device)
        #label = torch.tensor(label).to(device)
        #anchors = anchor_embeddings[cnt*args.batch_size:cnt*args.batch_size+len_batch]
        anchors = anchor_embeddings[5749:5750]
        anchors = anchors.to(device)
        #print(anchors.shape)
        
        #x,y = 65,45
        x,y = 60,30
        mask = make_mask(crops_result[0],sticker,x,y)
        for itr in range(args.max_iterations):
            g_temp = []
            for t in range(len(liner_interval)):
                c = liner_interval[t]
                X_adv = X_ori + c * delta
                X_adv = nn.functional.interpolate(X_adv, (inputsize[0], inputsize[1]), mode='bilinear', align_corners=False)
                # if args.di:
                #     X_adv = X_ori + delta
                #     X_adv = DI(X_adv, 500)   # diverse input operation
                #     X_adv = nn.functional.interpolate(X_adv, (inputsize[0], inputsize[1]), mode='bilinear', align_corners=False)
                
                feature = fr_model(X_adv)
                l_sim = cosin_metric(feature,anchors)
                loss = l_sim
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
            delta.data=delta.data-args.lr * torch.sign(g_syn)
            delta.data = delta.data * mask.to(device)
            #delta.data=delta.data.clamp(-args.linf_epsilon/255.,args.linf_epsilon/255.)
            delta.data=((X_ori+delta.data).clamp(0,1))-X_ori              # 噪声截取操作

            with torch.no_grad():
                X_adv = X_ori + delta
                X_adv = nn.functional.interpolate(X_adv, (inputsize[0], inputsize[1]), mode='bilinear', align_corners=False)
                feature = fr_model(X_adv)
                sim = cosin_metric(feature,anchors).item()
                print('---iter {} --- loss = {}'.format(itr,sim))
        
        for i in range(len_batch):
            adv_final = (X_ori+delta)[i].cpu().detach().numpy()
            adv_final = (adv_final*255).astype(np.uint8)
            file_path = os.path.join(adv_img_folder, '{}.jpg'.format(cnt*args.batch_size+i))
            adv_x_255 = np.transpose(adv_final, (1, 2, 0))
            im = Image.fromarray(adv_x_255)
            im.save(file_path,quality=99)
        cnt+=1
    torch.cuda.empty_cache()

def miattack_face(data_all,fr_model):
    cnt = 0
    anchor_embeddings =  joblib.load('/home/guoying/rlpatch/stmodels/{}/embeddings_{}_5752.pkl'.format(args.source_model,args.source_model))
    for X in data_all:
        X = list(map(list, zip(*X)))
        imgs,label = X
        len_batch = len(imgs)
        crops_result, crops_tensor = crop_imgs(imgs, args.width, args.height)

        X_ori = torch.stack(crops_tensor).to(device)
        #print(X_ori.shape)
        delta = torch.zeros_like(X_ori,requires_grad=True).to(device)
        #label = torch.tensor(label).to(device)
        #anchors = anchor_embeddings[cnt*args.batch_size:cnt*args.batch_size+len_batch]
        anchors = anchor_embeddings[5749:5750]
        anchors = anchors.to(device)
        #print(anchors.shape)
        
        x,y = 65,45
        #x,y = 60,30
        mask = make_mask(crops_result[0],sticker,x,y)
        grad_momentum = 0
        for itr in range(args.max_iterations):
            X_adv = X_ori + delta
            X_adv = nn.functional.interpolate(X_adv, (inputsize[0], inputsize[1]), mode='bilinear', align_corners=False)
            
            feature = fr_model(X_adv)
            l_sim = cosin_metric(feature,anchors)
            loss = l_sim
            #print('---iter {} interval {}--- loss = {}'.format(itr,t,loss))
            loss.backward()
            
            # MI operation
            grad_c = delta.grad.clone()                        
            grad_a = grad_c / torch.mean(torch.abs(grad_c), (1, 2, 3), keepdim=True)+1.0*grad_momentum   # 1
            grad_momentum = grad_a
                
            delta.grad.zero_()
            # L-inf attack
            delta.data=delta.data-args.lr * torch.sign(grad_momentum)
            delta.data = delta.data * mask.to(device)
            #delta.data=delta.data.clamp(-args.linf_epsilon/255.,args.linf_epsilon/255.)
            delta.data=((X_ori+delta.data).clamp(0,1))-X_ori

            with torch.no_grad():
                X_adv = X_ori + delta
                X_adv = nn.functional.interpolate(X_adv, (inputsize[0], inputsize[1]), mode='bilinear', align_corners=False)
                feature = fr_model(X_adv)
                sim = cosin_metric(feature,anchors).item()
                print('---iter {} --- loss = {}'.format(itr,sim))
        
        for i in range(len_batch):
            adv_final = (X_ori+delta)[i].cpu().detach().numpy()
            adv_final = (adv_final*255).astype(np.uint8)
            file_path = os.path.join(adv_img_folder, '{}.jpg'.format(cnt*args.batch_size+i))
            adv_x_255 = np.transpose(adv_final, (1, 2, 0))
            im = Image.fromarray(adv_x_255)
            im.save(file_path,quality=99)
        cnt+=1
    torch.cuda.empty_cache()

def parse_arguments():
    parser = argparse.ArgumentParser('Running script', add_help=False)
    parser.add_argument('--input_dir', default='/home/guoying/rlpatch/example', type=str)
    #parser.add_argument('--output_dir', default='../output_dir', type=str)
    parser.add_argument('--batch_size', type=int, default=10)

    parser.add_argument('--source_model', default='arcface', type=str)
    parser.add_argument('--width', type=int, default=160)
    parser.add_argument('--height', type=int, default=160)
    parser.add_argument('--max_iterations', type=int, default=260)
    parser.add_argument('--lr', type=eval, default=0.009)
    parser.add_argument('--linf_epsilon', type=float, default=255)
    parser.add_argument('--di', type=eval, default="False")
    args = parser.parse_args()
    return args

# def check():
#     # faceimg = Image.open('/home/guoying/rlpatch/example/guoying/1103_gy.jpg')
#     # crops_result, crops_tensor = crop_imgs([faceimg],args.width, args.height)
#     crops_result = [Image.open('/home/guoying/rlpatch/adv_imgs/0.jpg')]
#     anchor_embeddings =  joblib.load('/home/guoying/rlpatch/stmodels/{}/embeddings_{}_5752.pkl'.format(args.source_model,args.source_model))
#     anchors = anchor_embeddings[5749:5750]
#     anchors = anchors.to(device)

#     intput = torch.unsqueeze(trans(crops_result[0]),0).to(device)
#     print(intput.shape)
#     intput = nn.functional.interpolate(intput, (inputsize[0], inputsize[1]), mode='bilinear', align_corners=False)            # 插值到224
#     feature = fr_model(intput)
#     m = cosin_metric(feature,anchors,device)
#     print(m)

if __name__=="__main__":
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

    fr_model = load_model(args.source_model).eval().to(device)
    if(args.source_model == 'facenet'):
        inputsize = [160,160]
    else:
        inputsize = [112,112]
    #check()
    #miattack_face(loader,fr_model)
    
    