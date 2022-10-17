# _*_ coding:utf-8 _*_
import numpy as np
import dlib
from cv2 import cv2
from PIL import Image
from torchvision import datasets
import copy
import torch
import sys
sys.path.append("..")
from attack.stick import make_basemap
from torch.distributions import Normal, Categorical
import torch.nn as nn

def face_landmarks(initial_pic):
    dotsets = np.zeros((1,81,2))
    detector = dlib.get_frontal_face_detector()
    #predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    predictor = dlib.shape_predictor('/home/lenovo/yujie/code_rl/newpatch_rl/shape_predictor_81_face_landmarks.dat')
    
    pic_array = np.array(initial_pic)
    r,g,b = cv2.split(pic_array)
    img = cv2.merge([b, g, r])
    #img = cv2.imread(pic_dir)                          # cv2读取图像

    imgsize = img.shape
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)   # 取灰度
    rects = detector(img_gray, 1)                      # 人脸数rects
    #print('num of rects=',len(rects),rects[1])
    #print(len(rects))
    landmarks = np.matrix([[p.x, p.y] for p in predictor(img,rects[0]).parts()])
    #print(landmarks)
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])           # 81点的坐标
        #print(idx,pos)
        if(idx >= 0 and idx <= 67):
            dotsets[0][idx] = pos
        elif(idx == 78):
            dotsets[0][68] = pos
        elif(idx == 74):
            dotsets[0][69] = pos
        elif(idx == 79):
            dotsets[0][70] = pos
        elif(idx == 73):
            dotsets[0][71] = pos
        elif(idx == 72):
            dotsets[0][72] = pos
        elif(idx == 80):
            dotsets[0][73] = pos
        elif(idx == 71):
            dotsets[0][74] = pos
        elif(idx == 70):
            dotsets[0][75] = pos
        elif(idx == 69):
            dotsets[0][76] = pos
        elif(idx == 68):
            dotsets[0][77] = pos
        elif(idx == 76):
            dotsets[0][78] = pos
        elif(idx == 75):
            dotsets[0][79] = pos
        elif(idx == 77):
            dotsets[0][80] = pos

    #         cv2.circle(img, pos, 1, color=(0, 255, 0)) # 利用cv2.circle给每个特征点画一个圈，共81个
    #         font = cv2.FONT_HERSHEY_SIMPLEX            # 利用cv2.putText输出1-68
    #         cv2.putText(img, str(idx+1), pos, font, 0.2, (0, 0, 255), 1,cv2.LINE_AA)
    # cv2.namedWindow("img", 2)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    #print('dotsets',dotsets.shape)
    return dotsets,imgsize

def circle_mark(facemask,dot,brw):
    height,width = facemask.shape
    dot = dot.astype(np.int16)
    dotlen = len(dot)
    for i in range(dotlen):
        x1,y1 = dot[i]
        facemask[x1,y1] = brw
        if(i == dotlen-1):
            j = 0
        else:
            j = i+1
        x2,y2 = dot[j]
        if(y2 - y1 != 0):
            k = (x2 - x1) / (y2 - y1)
            symbol = 1 if(y2 - y1 > 0) else -1
            for t in range(symbol*(y2 - y1)-1):
                y3 = y1 + symbol * (t + 1)
                x3 = int(round(k * (y3 - y1) + x1))
                # print('x1,y1,x2,y2',x1,y1,x2,y2)
                # print('x3,y3 = ',x3,y3)
                facemask[x3,y3] = brw

    dot = np.array(dot)
    lower = np.min(dot,axis = 0)[1]
    upper = np.max(dot,axis = 0)[1]
    # lower = clip(lower,0,width-1)
    # lower = clip(upper,0,width-2)
    for h in range(lower,upper+1):
        h = clip(h,0,width-1)
        left = 0
        right = 0
        cruitl = np.min(dot,axis = 0)[0]
        cruitr = np.max(dot,axis = 0)[0]
        # cruitl = clip(cruitl,1,height-1)
        # cruitr = clip(cruitr,0,height-3)
        for i in range(cruitl-1,cruitr+2):
            i = clip(i,0,height-1)
            if(facemask[i][h] == brw):
                left = i
                break
        for j in reversed(list(range(cruitl-1,cruitr+2))):
            j = clip(j,0,height-1)
            if(facemask[j][h] == brw):
                right = j
                break
        left_cursor = left
        right_cursor = right
        # print('h = ',h)
        # print('left_cursor,right_cursor = ',left_cursor,right_cursor)
        if(left_cursor != right_cursor):        
            while True:
                facemask[left_cursor][h] = brw
                left_cursor = left_cursor + 1
                if(facemask[left_cursor][h] == brw):
                    break
            while True:
                facemask[right_cursor][h] = brw
                right_cursor = right_cursor - 1
                if(facemask[right_cursor][h] == brw):
                    break
    return facemask

def clip(x,lower,upper):
    x = lower if(x<lower) else x
    x = upper if(x>upper) else x
    return x

def make_facemask(initial_pic):
    w,h = initial_pic.size
    dotsets,imgsize = face_landmarks(initial_pic)
    for i in range(len(dotsets[0])):
        dotsets[0][i][0] = clip(dotsets[0][i][0],0,w-1)
        dotsets[0][i][1] = clip(dotsets[0][i][1],0,h-1)
    facemask = np.zeros((imgsize[1],imgsize[0]))
    #----------face--------------
    face = dotsets[0][:17]
    face2 = dotsets[0][68:]
    face = np.vstack((face,face2))
    #print(face)
    facemask = circle_mark(facemask,face,brw=1)

    #---------eyebrow-----------
    # browl = dotsets[0][17:22]
    # browr = dotsets[0][22:27]
    # facemask = circle_mark(facemask,browl,brw=0)
    # facemask = circle_mark(facemask,browr,brw=0)

    #----------eye--------------
    eyel = dotsets[0][36:42]
    eyer = dotsets[0][42:48]
    facemask = circle_mark(facemask,eyel,brw=0)
    facemask = circle_mark(facemask,eyer,brw=0)

    #---------mouth-------------
    mouth = dotsets[0][48:61]
    facemask = circle_mark(facemask,mouth,brw=0)

    #---------nose--------------
    #nose = np.vstack((dotsets[0][31:36],dotsets[0][42],dotsets[0][27],dotsets[0][39]))
    nose = np.vstack((dotsets[0][31:36],dotsets[0][29]))
    # right = [dotsets[0][27][0]+1,dotsets[0][27][1]]
    # left = [dotsets[0][27][0]-1,dotsets[0][27][1]]
    # nose = np.vstack((dotsets[0][31:36],right,left))
    facemask = circle_mark(facemask,nose,brw=0)

    facemask = facemask.transpose()

    #facemask[5][15]=1
    # cv2.imshow("outImg",facemask)
    # #cv2.imshow("outImg",facemask)
    # cv2.waitKey(0)
    # num_space = np.sum(facemask).astype(int)
    # print(num_space)
    
    return facemask

def count_face(initial_pic):
    dotsets = np.zeros((1,81,2))
    detector = dlib.get_frontal_face_detector()
    #predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    predictor = dlib.shape_predictor('/home/lenovo/yujie/code_rl/newpatch_rl/shape_predictor_81_face_landmarks.dat')
    
    pic_array = np.array(initial_pic)
    r,g,b = cv2.split(pic_array)
    img = cv2.merge([b, g, r])
    #img = cv2.imread(pic_dir)                          # cv2读取图像

    imgsize = img.shape
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)   # 取灰度
    rects = detector(img_gray, 1)                      # 人脸数rects
    num = len(rects)
    #print('num of rects=',len(rects),rects[1])
    #print(len(rects))
    return num

def loc_space(img,sticker,threshold=100):
    facemask = make_facemask(img)
    h,w = facemask.shape
    space = facemask.copy()
    _, st_mask = make_basemap(w,h,sticker,1,1)
    st_space = np.sum(st_mask).astype(int)
    for i in range(h):
        for j in range(w):
            if(facemask[i][j]==1):
                _,st_mask = make_basemap(w,h,sticker,j,i)
                if(st_space-np.sum(st_mask*facemask)>threshold):
                    space[i][j] = 0
    return space

def vector_processor(featuremap,eps_logits,space,device):
    """
    Args: 
        featuremap: (1 * (n_models) * height * width) tensor//(on cuda)
        eps_logits: (1*eps_dim)                       tensor//(on cuda)
        space: (height * width) | 0/1 matrix | valid mask
    return: parameters:
            x,y,weights,epsilon
    """
    fm_op = featuremap[0]                         # ((n_models) * height * width)
    n,h,w = fm_op.shape
    n_models = n
    pre_actions = []

    op = fm_op[:n_models].reshape(n_models,1,-1)  # (n_models,1,h*w)
    '''-------------------weights action-------------------'''
    weg_resp = torch.mean(op,dim=2).t()             # (1,n_models)
    #print('weg_resp = ',weg_resp,weg_resp.shape)
    weg_probs = torch.softmax(weg_resp,dim=1)      # (1,n_models)
    #print('weg_probs = ',weg_probs)
    
    '''-------------------location action-------------------'''
    
    #print('op = ',op,op.shape)
    loct_resp = torch.softmax(op,dim=2).squeeze(1)            # (n_models,1,h*w)
    #print('loct_resp = ',loct_resp,loct_resp.shape)
    loct_probs = torch.mm(weg_probs,loct_resp,out=None)[0]    # (h*w,)
    #print('loct_probs = ',loct_probs)
    loct_pbspace = space.reshape(-1) * loct_probs
    #print('loct_pbspace',loct_pbspace)
    loct_preaction = Categorical(loct_pbspace)
    pre_actions.append(loct_preaction)
    # loct_action = loct_preaction.sample()
    # y,x = (loct_action // w).cpu().detach().numpy(), loct_action % w
    # print(y,x)
    
    '''-------------------weights action-------------------'''
    for i in range(n_models):
        dist_weg = Normal(weg_probs[0][i], torch.tensor(0.02).to(device))
        #print('dist_weg sample = ',dist_weg.sample())
        pre_actions.append(dist_weg)

    # '''-------------------epsilon action-------------------''' # value range (0.009,0.189)
    # epsilon_matrix = fm_op[n-1]         # (h*w)
    # #print('epsilon_matrix = ',epsilon_matrix,epsilon_matrix.shape)
    # eps_resp = torch.mean(epsilon_matrix)
    # #print('eps_resp = ',eps_resp)
    # eps_probs = 0.18*torch.sigmoid(eps_resp)+0.009
    # #print('eps_probs = ',eps_probs)
    # dist_eps = Normal(eps_probs, torch.tensor(0.01).to(device))
    # # eps_samp = dist_eps.sample()
    # # logp = dist_eps.log_prob(eps_samp)
    # # print('dist_eps.sample() = ',eps_samp,logp)
    # pre_actions.append(dist_eps)
    '''-----------------new epsilon action-------------------''' # value range (0.01,0.2)
    eps_probs = torch.softmax(eps_logits,dim=1)      # (bt,eps_dim)
    #print('eps_probs = ',eps_probs)
    dist_eps = Categorical(eps_probs[0])
    pre_actions.append(dist_eps)
    #print('eps_probs:',eps_probs)
    
    return pre_actions

def vector_processor2(featuremap,eps_logits,space,device):
    """
    Args: 
        featuremap: (1 * (n_models) * height * width) tensor//(on cuda)
        eps_logits: (1*eps_dim)                       tensor//(on cuda)
        space: (height * width) | 0/1 matrix | valid mask
    return: parameters:
            x,y,weights,epsilon
    """
    fm_op = featuremap[0]                         # ((n_models+1) * height * width)
    n,h,w = fm_op.shape
    n_models = n
    pre_actions = []
    '''-------------------location action-------------------'''
    op = fm_op[:n_models].reshape(n_models,1,-1)  # (n_models,1,h*w)
    #print('op = ',op,op.shape)
    loct_resp = torch.softmax(op,dim=2)           # (n_models,1,h*w)
    #print('loct_resp = ',loct_resp,loct_resp.shape)
    loct_probs = torch.mean(loct_resp,dim=0)[0]    # (h*w,)
    #print('loct_probs = ',loct_probs)
    loct_pbspace = space.reshape(-1) * loct_probs
    #print('loct_pbspace',loct_pbspace)
    loct_preaction = Categorical(loct_pbspace)
    pre_actions.append(loct_preaction)
    # loct_action = loct_preaction.sample()
    # y,x = (loct_action // w).cpu().detach().numpy(), loct_action % w
    # print(y,x)

    '''-------------------weights action-------------------'''
    weg_resp = torch.mean(op,dim=2).t()             # (1,n_models)
    #print('weg_resp = ',weg_resp,weg_resp.shape)
    weg_probs = torch.softmax(weg_resp,dim=1)      # (1,n_models)
    #print('weg_probs = ',weg_probs)
    for i in range(n_models):
        dist_weg = Normal(weg_probs[0][i], torch.tensor(0.02).to(device))
        #print('dist_weg sample = ',dist_weg.sample())
        pre_actions.append(dist_weg)

    # '''-------------------epsilon action-------------------''' # value range (0.009,0.189)
    # epsilon_matrix = fm_op[n-1]         # (h*w)
    # #print('epsilon_matrix = ',epsilon_matrix,epsilon_matrix.shape)
    # eps_resp = torch.mean(epsilon_matrix)
    # #print('eps_resp = ',eps_resp)
    # eps_probs = 0.18*torch.sigmoid(eps_resp)+0.009
    # #print('eps_probs = ',eps_probs)
    # dist_eps = Normal(eps_probs, torch.tensor(0.01).to(device))
    # # eps_samp = dist_eps.sample()
    # # logp = dist_eps.log_prob(eps_samp)
    # # print('dist_eps.sample() = ',eps_samp,logp)
    # pre_actions.append(dist_eps)
    '''-----------------new epsilon action-------------------''' # value range (0.01,0.2)
    eps_probs = torch.softmax(eps_logits,dim=1)      # (bt,eps_dim)
    #print('eps_probs = ',eps_probs)
    dist_eps = Categorical(eps_probs[0])
    pre_actions.append(dist_eps)
    #print('eps_probs:',eps_probs)
    
    return pre_actions

def clip(x,lower,upper):
    if(lower>upper):
        return 0
    if(x<lower):
        return lower
    if(x>upper):
        return upper
    return x

def generate_actions(pre_actions):
    actions = []
    for i in range(len(pre_actions)):
        ac = pre_actions[i].sample()
        actions.append(ac)
    return actions

def actions2params(actions,width):
    params_slove = []                                                                  # [[x,y],[w1,w2,...,wn],eps]
    for i in range(len(actions)):
        if(i==0):
            ind = actions[i].cpu().detach().item()#.numpy()#
            #print('ind = ',ind)
            y,x = (ind // width), (ind % width)#.cpu().detach().numpy()
            params_slove.append([x,y])
            temp =[]
            temp2 = []
            accmw = 1
        elif(i==len(actions)-1):
            params_slove.append(temp)
            eps = actions[i].cpu().detach().item()
            #params_slove.append(clip(eps,0.009,0.189)) #.copy()
            # eps_sets = np.arange(1/255,21/255,1/255)
            eps_sets = np.arange(0.01,0.21,0.01)
            params_slove.append(eps_sets[eps]) #.copy()
        # elif(i==len(actions)-1):
        #     eps = actions[i].cpu().detach().item()
            
        #     eps_sets = np.arange(0.05,10.05,0.05)
        #     params_slove.append(eps_sets[eps]) #.copy()
        else:
            w = actions[i].cpu().detach().item()       #.numpy()[0]
            #print(w,accmw,clip(w,0,accmw))
            clip_w = clip(w,0,accmw)
            #print('clip_w = ',clip_w)
            temp.append(clip_w) #.copy()
            accmw -=clip_w
        #print(i,' temp = ',temp)
    return params_slove

def agent_output(agent,clean_ts,space_ts,device):
    height,width = clean_ts.shape[2],clean_ts.shape[3]
    #print(height,width)
    featuremap = agent(clean_ts)
    pre_actions = vector_processor(featuremap,space_ts,device)
    actions = generate_actions(pre_actions)
    #print('actions = ',actions)
    params_slove = actions2params(actions,width)
    return params_slove
    
if __name__ == "__main__":
    img = Image.open('/home/lenovo/shighgh/newpatch_rl/code_rl/rlpatch/lfw_crop/228.jpg')
    facemask = make_facemask(img)
    cv2.imwrite("mask.jpg",facemask*255)
    # make_mask(img,sticker,x,y)
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # space = torch.from_numpy(np.random.randint(0,2,size=(2,3))).to(device)
    # print('space = ',space)
    # featuremap = torch.tensor([[ [[0.1,0.2,-0.3],[0.6,0.01,0.9]],
    #                              [[0.1,0.2,0.3],[-0.6,0.01,0.9]],
    #                              [[-0.23,0.2,-0.7],[-0.6,0.01,0.9]] ] ]).to(device)
    # print('featuremap.shape = ',featuremap.shape)
    # #print(featuremap[0][:2])
    # # op = featuremap.reshape(2,1,-1)
    # # print(op,op.shape)
    # ac = vector_processor(featuremap,space)
