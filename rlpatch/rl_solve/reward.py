from os import system
import numpy as np
import dlib
from cv2 import cv2
from PIL import Image
from torchvision import datasets
import copy
import joblib
import torch
import torch.nn as nn
import sys
sys.path.append("/home/lenovo/yujie/code_rl/newpatch_rl/rlpatch+api/")
from attack.tiattack import load_model,cosin_metric,crop_imgs
from torchvision import transforms
from torch.utils.data import DataLoader
from config import Config
import matplotlib.pyplot as plt 

from rl_solve.tct import check_tct, reward_tct

inputsize = {'arcface34':[112,112],'arcface50':[112,112],'cosface34':[112,112],'cosface50':[112,112],
             'facenet':[160,160],'insightface':[112,112],'sphere20a':[112,96],'re_sphere20a':[112,96],
             'mobilefacenet':[112,112],'tencent':[112,112]}
           
trans = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
            ])
def cosin_all(feature,model_name,device):
    embedding_sets = joblib.load('/home/lenovo/yujie/code_rl/newpatch_rl/rlpatch/stmodels/{}/embeddings_{}_5752.pkl'.format(model_name,model_name))
    sets = torch.t(embedding_sets).to(device)
    #print(embedding.shape,sets.shape)
    numerator = torch.mm(feature,sets)
    norm_x1 = torch.norm(feature,dim=1)
    norm_x1 = torch.unsqueeze(norm_x1,1)
    norm_x2 = torch.norm(sets,dim=0) #,keepdims=True
    norm_x2 = torch.unsqueeze(norm_x2,0)
    #print('norm_x1,norm_x2 ',norm_x1.shape,norm_x2.shape)
    denominator = torch.mm(norm_x1, norm_x2)
    metrics = torch.mul(numerator,1/denominator)
    return metrics.cpu().detach()
               
def load_anchors(model_name, device, target):
    anchor_embeddings =  joblib.load('/home/lenovo/yujie/code_rl/newpatch_rl/rlpatch/stmodels/{}/embeddings_{}_5752.pkl'.format(model_name,model_name))
    anchor = anchor_embeddings[target:target+1]
    anchor = anchor.to(device)
    return anchor

def reward_output(adv_face_ts, threat_model, threat_name, target,truelabel, device):
#def reward_output(adv_face_ts, threat_model, threat_name, target, device):
    # adv_tensor = [trans(adv_face)]
    # advface_ts = torch.stack(adv_tensor).to(device)

    threat = threat_model.to(device)
    advface_ts = adv_face_ts.to(device)
    X_op = nn.functional.interpolate(advface_ts, (inputsize[threat_name][0], inputsize[threat_name][1]), mode='bilinear', align_corners=False)
    feature = threat(X_op)

    anchor = load_anchors(threat_name, device, target)
    if target == truelabel:
        l_sim = cosin_metric(feature,anchor,device).cpu().detach().item()
        return l_sim
    else:
        anchor2 = load_anchors(threat_name, device, truelabel)
        l_sim = cosin_metric(feature,anchor,device).cpu().detach().item()
        l_sim2 = cosin_metric(feature,anchor2,device).cpu().detach().item()
        return l_sim2-l_sim
    
    #anchor2 = load_anchors(threat_name, device, truelabel)
    #l_sim2 = cosin_metric(feature,anchor2,device).cpu().detach().item()
    #return l_sim2-l_sim
    return l_sim

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
    

def check_all(adv_face_ts, threat_model, threat_name, device):
    # adv_tensor = [trans(adv_face)]
    # advface_ts = torch.stack(adv_tensor).to(device)
    percent = []
    typess = []
    
    # #print(adv_face_ts)
    # adv_face_arr = np.uint8(adv_face_ts.numpy()*255)
    # #print(adv_face_arr)
    # adv_face_ts = torch.from_numpy(adv_face_arr/255).half()
    # #print(adv_face_ts)

    threat = threat_model.to(device)
    threat.eval()
    def collate_fn(x):
        return x
    loader = DataLoader(
        adv_face_ts,
        batch_size=55,
        shuffle=False,
        collate_fn=collate_fn
    )

    for X in loader:
        #print(X[0].shape)
        advface_ts = torch.stack(X).to(device)
        X_op = nn.functional.interpolate(advface_ts, (inputsize[threat_name][0], inputsize[threat_name][1]), mode='bilinear', align_corners=False)
        feature = threat(X_op)
        for i in range(len(feature)):
            sim_all = cosin_all(torch.unsqueeze(feature[i],0),threat_name,device)
            _, indices = torch.sort(sim_all, dim=1, descending=True)
            cla = [indices[0][0].item(),indices[0][1].item(),indices[0][2].item(),\
                indices[0][3].item(),indices[0][4].item(),indices[0][5].item(),indices[0][6].item()]
            typess.append(cla)
            tage = sim_all[0].numpy()
            percent.append(tage)
    return typess,np.array(percent)

# img = cv2.imread(r"E:\newpatch_rl\code_rl\rlpatch\tencent_check\4\0_13269_.jpg")
# img = cv2.imread(r"E:\newpatch_rl\code_rl\rlpatch\pics_check\18\tar_0_13271_1800.png")
# b,g,r = cv2.split(img)
# outarray = cv2.merge([r, g, b])
# outImage = Image.fromarray(np.uint8(outarray))
# plt.imshow(outImage)
# plt.show()

# outImage = Image.open(r"E:\newpatch_rl\code_rl\lfw_images\Zzz_Yu_Jie2\Yu_Jie_0001.jpg")
# crops_result, crops_tensor = crop_imgs([outImage], 160, 160)
# plt.imshow(crops_result[0])
# plt.show()
# plt.imshow(outImage)
# plt.show()


# outImage = C
# threat_model = load_model(Config.threat_name, torch.device('cpu'))

# tempts = torch.unsqueeze(trans(outImage),0)
# typess,per = check_all(tempts, threat_model, Config.threat_name, Config.device)
# print(typess)
# print(per[0][typess[0][0]])
# print(per[0][typess[0]])

# img1 = Image.open(r"C:\Users\admin\Desktop\pic\新建文件夹\1715507899,3558235981.jpg")
# img2 = Image.open(r"C:\Users\admin\Desktop\pic\新建文件夹\gaoyunzhen.jpg")
# img3 = Image.open(r"C:\Users\admin\Desktop\pic\新建文件夹\huge.jpg")
# img4 = Image.open(r"E:\newpatch_rl\code_rl\lfw_images\Zz_Guo_Ying\Ying_Guo_0001.jpg")
# # img5 = Image.open(r"C:\Users\admin\Desktop\pic\新建文件夹\u=707230193,759635209&fm=26&fmt=auto&gp=0.webp")
# # img6 = Image.open(r"C:\Users\admin\Desktop\pic\新建文件夹\u=998837796,1390066138&fm=26&fmt=auto&gp=0.webp")
# # img7 = Image.open(r"C:\Users\admin\Desktop\pic\新建文件夹\u=4147916807,3473337324&fm=26&fmt=auto&gp=0.webp")
# # img = [img1,img2,img3,img4]
# crops_result, crops_tensor = crop_imgs([img4], 112, 112)                       # convert face image to tensor                                                                     # RL framework iterations
# init_face = crops_result[0]
# sim_labels, sim_probs = check_tct([init_face])
# fw = open(r"E:/1.txt","w")
# print(len(sim_probs[0]))
# for prob in sim_probs[0]:
#     fw.write(str(prob)+"\n")
# fw.close()



# sys.path.append("../")
# from attack import stick

# def make_stmask(face,sticker,x,y):
#     w,h = face.size
#     mask = stick.make_masktensor(w,h,sticker,x,y)
#     return mask
# def miattack_face2(x,y,weights,epsilon, model_names,
#                   img, target, device,
#                   width, height, emp_iterations, sticker, targeted = True):
#     flag = 1 if targeted else -1
#     w,h = img.size
#     if(w!=width or h!=height):
#         crops_result, crops_tensor = crop_imgs([img], width, height)
#     else:
#         crops_result = [img]
#         crops_tensor = [trans(img)]
#     X_ori = torch.stack(crops_tensor).to(device)
#     #print(X_ori.shape)
#     delta = torch.zeros_like(X_ori,requires_grad=True).to(device)
#     #label = torch.tensor(label).to(device)
    
#     fr_models, anchors = [], []
#     for name in model_names:
#         model = load_model(name, device)
#         anchor = load_anchors(name, device, target)
#         fr_models.append(model)
#         anchors.append(anchor)
        
#     mask = make_stmask(crops_result[0],sticker,x,y)
#     grad_momentum = 0
#     for itr in range(emp_iterations):   # iterations in the generation of adversarial examples
#         X_adv = X_ori + delta
#         X_adv.retain_grad()
#         accm = 0
#         for (i, name) in enumerate(model_names):
#             X_op = nn.functional.interpolate(X_adv, (inputsize[name][0], inputsize[name][1]), mode='bilinear', align_corners=False)
#             feature = fr_models[i](X_op)
#             l_sim = cosin_metric(feature,anchors[i],device)
#             accm += l_sim * weights[i]
#         loss = flag * accm
#         loss.backward()
        
#         # MI operation
#         grad_c = X_adv.grad.clone()                        
#         grad_a = grad_c / torch.mean(torch.abs(grad_c), (1, 2, 3), keepdim=True)+1.0*grad_momentum   # 1
#         grad_momentum = grad_a
            
#         X_adv.grad.zero_()
#         X_adv.data=X_adv.data+epsilon * torch.sign(grad_momentum)* mask.to(device)
#         X_adv.data=X_adv.data.clamp(0,1)
#         delta.data=X_adv-X_ori

#     adv_face_ts = (X_ori+delta).cpu().detach()
#     adv_final = (X_ori+delta)[0].cpu().detach().numpy()
#     adv_final = (adv_final*255).astype(np.uint8)
#     adv_x_255 = np.transpose(adv_final, (1, 2, 0))
#     im = Image.fromarray(adv_x_255)
#     return adv_face_ts,im

# def miattack_face(x,y,weights,epsilon, model_names,
#                   img, target, device,
#                   width, height, emp_iterations, targeted = True):
#     flag = 1 if targeted else -1
#     w,h = img.size
#     if(w!=width or h!=height):
#         crops_result, crops_tensor = crop_imgs([img], width, height)
#     else:
#         crops_result = [img]
#         crops_tensor = [trans(img)]
#     X_ori = torch.stack(crops_tensor).to(device)
    
#     delta = torch.zeros([1,3,25,30],requires_grad=True).to(device)
#     with torch.no_grad():
#         delta[:,:,:,:] = X_ori[:,:,y:y+25,x:x+30]
#         X_ori[0,:,y:y+25,x:x+30] = torch.zeros([3,25,30])

#     fr_models, anchors = [], []
#     for name in model_names:
#         model = load_model(name, device)
#         anchor = load_anchors(name, device, target)
#         fr_models.append(model)
#         anchors.append(anchor)
        
#     # mask = make_stmask(crops_result[0],sticker,x,y)
#     grad_momentum = 0
#     for itr in range(emp_iterations):   # iterations in the generation of adversarial examples
#         delta.retain_grad()
#         X_adv = X_ori
#         X_adv[0,:,y:y+25,x:x+30] = delta[0,:,:,:]

#         accm = 0
#         # print('---iter {}---'.format(itr),end=' ')
#         for (i, name) in enumerate(model_names):
#             X_op = nn.functional.interpolate(X_adv, (inputsize[name][0], inputsize[name][1]), mode='bilinear', align_corners=False)
#             feature = fr_models[i](X_op)
#             l_sim = cosin_metric(feature,anchors[i],device)
#             # print(name,':','{:.4f}'.format(l_sim.item()),end=' ')
#             accm += l_sim * weights[i]
#         loss = flag * accm
#         # print('L_sim = {:.4f},L_slope = {:.4f}'.format(flag * accm.item(),0.3*slope.item()),end='\n')
#         loss.backward(retain_graph=True)
        
#         # MI operation
#         grad_c = delta.grad.clone()                        
#         grad_a = grad_c / torch.mean(torch.abs(grad_c), (1, 2, 3), keepdim=True)+1.0*grad_momentum   # 1
#         grad_momentum = grad_a
            
#         delta.grad.zero_()
#         delta.data=delta.data+epsilon * torch.sign(grad_momentum).to(device)
#         delta.data=delta.data.clamp(0,1)
#     X_ori[:,:,y:y+25,x:x+30] = delta
#     adv_face_ts = X_ori.cpu().detach()
#     adv_final = X_ori[0].cpu().detach().numpy()
#     adv_final = (adv_final*255).astype(np.uint8)
#     adv_x_255 = np.transpose(adv_final, (1, 2, 0))
#     im = Image.fromarray(adv_x_255)
#     return adv_face_ts,im

# x,y,weights,epsilon = 39, 37, [0.2760891020298004, 0.269814133644104, 0.2587984800338745, 0.19529828429222107], 0.08
# model_names = ['facenet','arcface34','arcface50','cosface34']
# width,height = 160,160
# dataset = datasets.ImageFolder('/home/lenovo/yujie/code_rl/newpatch_rl/lfw_images')
# img = dataset[len(dataset)-1][0]

# crops_result, crops_tensor = crop_imgs([img], width, height)                       # convert face image to tensor                                                                     # RL framework iterations
# init_face = crops_result[0]
# target = 5708
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# emp_iterations = 130
# threat_name = "arcface34"
# threat_model = load_model(threat_name, torch.device('cpu'))
# all = 0
# emp_iterationslist = [1,10,20,30,40,50,60,70,80,90,100,110,120,130]
# sticker = Image.new('RGBA',(30,25),(255,255,255,255))
# fw = open(r"C:\Users\admin\Desktop\huizong2.txt","a+")
# adv_face_ts, adv_face = miattack_face2(x,y,weights,epsilon, model_names,
#                                             init_face,target, device, 
#                                             width, height, emp_iterations,sticker)

# if(threat_name!='tencent'):
#     sim_labels, sim_probs = check_all(adv_face_ts, threat_model, threat_name, device)
#     succ_label = sim_labels[0][:2]
#     succ_gap = sim_probs[0][succ_label]    
#     print('now_gap:{},{}'.format(succ_label,succ_gap))

# for emp_iterations in emp_iterationslist:
#     all = 0.0
#     for i in range(10):
#         adv_face_ts, adv_face = miattack_face2(x,y,weights,epsilon, model_names,
#                                             init_face,target, device, 
#                                             width, height, emp_iterations,sticker)
#         reward = reward_output(adv_face_ts,threat_model,threat_name,target,device)
#         all += reward
#     print(all/10)
#     fw.write(str(all))
#     fw.write("\n")
