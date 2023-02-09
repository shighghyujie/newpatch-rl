import os
import cv2
from skimage import transform
import torch
import numpy as np
import time
import joblib
from config import Config
from PIL import Image
from matplotlib import pyplot as plt
from torch.optim import lr_scheduler
from torch.nn import DataParallel
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from facenet_pytorch import MTCNN, InceptionResnetV1
import argparse
from attack.tiattack import miattack_face,crop_imgs,load_model

from rl_solve.attack_utils import loc_space,agent_output,vector_processor,generate_actions,actions2params
from rl_solve.agent import UNet, BackBone
from rl_solve.reward import reward_output, reward_slope, check_all
# from visualizer import Visualizer
# from Loss_Modifier import NPSCalculator,TotalVariation
#from adapselection.utils import finelist,agent_output,sparse_perturbation

loader = transforms.Compose([
    transforms.ToTensor()
])

def randomaction(params_slove):
    random_location = []
    epson = np.random.random(1)
    if epson < 0.1:
        random_location.append(int(np.random.randint(0,160-30-1,1)))
        random_location.append(int(np.random.randint(0,160-25-1,1)))
    else:
        random_location = params_slove[0]

    params_slove[0] = random_location

    return params_slove

def attack_process(img,sticker, threat_model, threat_name, model_names, fr_models, label, target_hs, device,
                   width ,height, emp_iterations, di, adv_img_folder, targeted = True,
                   sapce_thd=50,pg_m=5,max_iter=1000):
    num_iter = 0  
    crops_result, crops_tensor = crop_imgs([img], width, height)                       # convert face image to tensor                                                                     # RL framework iterations
    init_face = crops_result[0]
    before_face = init_face
    clean_ts = torch.stack(crops_tensor).to(device)
    space = loc_space(init_face,sticker,threshold=sapce_thd)
    space_ts = torch.from_numpy(space).to(device)
    n_models = len(model_names)

    sim_labels, sim_probs = check_all(crops_tensor, threat_model, threat_name, device)     
    
    start_label = sim_labels[0][:2]
    start_gap = sim_probs[0][start_label]
    #
    target = sim_labels[0][1] if targeted else sim_labels[0][0]
    truelabel = sim_labels[0][0]
    print('start_label: {} start_gap: {}'.format(start_label,start_gap)) 
    minila.write(str(start_label)+'|'+str(start_gap)+'|')                                          
    
    '''------------------------Agent initialization--------------------------'''
    print('Initializing the agent......')
    agent = UNet(inputdim = init_face.size[0],sgmodel = n_models,feature_dim=20).to(device)   
    # agent = BackBone(inputdim = init_face.size[0],sgmodel = n_models,feature_dim=20).to(device)                                       # agent(unet)
    optimizer = torch.optim.Adam(agent.parameters(),lr=1e-03,weight_decay=5e-04)       # optimizer
    scheduler = lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.1)                  # learning rate decay
    baseline = 0.0
    '''-------------Initialization with random parameters--------------------'''
    last_score = []                                                                    # predicted similarity
    all_final_params = []
    all_best_reward = -2.0
    all_best_face = init_face
    all_best_adv_face_ts = torch.stack(crops_tensor)
    while num_iter < max_iter:
        '''--------------------Agent output feature maps-------------------'''  
        featuremap1, featuremap2, eps_logits = agent(clean_ts)
        pre_actions = vector_processor(featuremap1,featuremap2,eps_logits,space_ts,device)
        cost = 0
        
        '''----------------Policy gradient and Get reward----------------'''
        pg_rewards = []
        phas_final_params = []
        phas_best_reward = -2.0
        phas_best_face = init_face
        phas_best_adv_face_ts = torch.stack(crops_tensor)
        for _ in range(pg_m):
            log_pis, log_sets = 0, []
            actions = generate_actions(pre_actions)                                    # sampling
            for t in range(len(actions)):
                log_prob = pre_actions[t].log_prob(actions[t])
                #print(log_prob)
                log_pis += log_prob
                log_sets.append(log_prob)
            params_slove = actions2params(actions,width)
            # params_slove = randomaction(params_slove)
            adv_face_ts, adv_face, mask = miattack_face(params_slove, model_names, fr_models,
                                        init_face, label, target, device, sticker,
                                        width, height, emp_iterations, di, adv_img_folder, targeted = targeted)
            x, y = params_slove[0]
            reward_m = reward_output(adv_face_ts, threat_model, threat_name, target, truelabel, device)
            
            reward_g = 0
            if(not targeted): reward_m = -1*reward_m
            reward_f = reward_m + 0.1*reward_g
            expected_reward = log_pis * (reward_f - baseline)
            
            cost -= expected_reward
            pg_rewards.append(reward_m)
            if reward_f > phas_best_reward:
                phas_final_params = params_slove
                phas_best_reward = reward_f
                phas_best_face = adv_face
                phas_best_adv_face_ts = adv_face_ts
        
        observed_value = np.mean(pg_rewards)
        # print('len pg_rewards = ',len(pg_rewards))
        print('{}-th: Reward is'.format(num_iter),end=' ')
        for p in range(len(pg_rewards)):
           print('{:.5f}'.format(pg_rewards[p]),end=' ')
        print('avg:{:.5f}\nparams = '.format(observed_value),phas_final_params)
        # log.write('{}-th: Reward is {}'.format(num_iter,pg_rewards)+str(observed_value)+'\n')
        # log.write('{}-th: params is {}'.format(num_iter,phas_final_params)+'\n')
        # if opt.display:
        #     visualizer.display_current_results(num_iter, observed_value, name='cosin similarity')
        #     visualizer.display_current_results(num_iter, cost.item(), name='loss')

        '''-------------------------Update Agent---------------------------'''
        optimizer.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(),5.0)
        optimizer.step()
        #observed_value = np.mean(pg_rewards)
        #baseline = 0.9*baseline + 0.1*observed_value
        
        scheduler.step()
        '''-------------------------Check Result---------------------------'''
        localtime2 = time.asctime( time.localtime(time.time()) )
        # file_path = os.path.join(adv_img_folder, '{}_{}.jpg'.format(num_iter,localtime2))
        # phas_best_face.save(file_path,quality=99)
        if phas_best_reward > all_best_reward:
            all_final_params = phas_final_params
            all_best_reward = phas_best_reward
            all_best_face = phas_best_face
            all_best_adv_face_ts = phas_best_adv_face_ts

        sim_labels, sim_probs = check_all(all_best_adv_face_ts, threat_model, threat_name, device)
        succ_label = sim_labels[0][:2]
        succ_gap = sim_probs[0][succ_label] 
        
        print('now_gap:{},{}'.format(succ_label,succ_gap))
        if ((targeted and sim_labels[0][0] == target) or                              # early stop
        # if ((targeted and sim_labels[0][0] == target and succ_gap[0] - succ_gap[1] > 10) or                              # early stop
            (not targeted and sim_labels[0][0] != target)):
            print('early stop at iterartion {},succ_label={},succ_gap={}'.format(num_iter,succ_label,succ_gap))
            minila.write(str(succ_label)+'|'+str(succ_gap)+'|')  
            return True, num_iter, [all_best_face,all_best_reward,all_final_params,all_best_adv_face_ts,before_face]

        last_score.append(observed_value)    
        last_score = last_score[-200:]   
        if last_score[-1] <= last_score[0] and len(last_score) == 200:
            print('FAIL: No Descent, Stop iteration')
            minila.write(str(succ_label)+'|'+str(succ_gap)+'|')  
            return False, num_iter, [all_best_face,all_best_reward,all_final_params,all_best_adv_face_ts,before_face]
        
        num_iter += 1
        #joblib.dump([all_best_adv_face_ts,all_final_params], folder_path+'/{}_{}_face&params.pkl'.format(i,idx))
        #joblib.dump(all_final_params, folder_path+'/all_final_params.pkl')
    minila.write(str(succ_label)+'|'+str(succ_gap)+'|')
    return False,num_iter, [all_best_face,all_best_reward,all_final_params,all_best_adv_face_ts,before_face]


if __name__=="__main__":
    opt = Config()
    localtime1 = time.asctime( time.localtime(time.time()) )
    localtime1 = localtime1.replace(":","_").replace(" ","_")
    # folder_path = opt.adv_img_folder
    folder_path = os.path.join(opt.adv_img_folder, localtime1)
    if not os.path.exists(folder_path): 
        os.makedirs(folder_path)
    
    dataset = datasets.ImageFolder(opt.dataset_folder)
    dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
    print(len(dataset))
    minila = open(folder_path+'/total_log.txt','a')
    minila.write(str(opt.model_names)+'\n')
    fr_models = []
    for name in opt.model_names:
        model = load_model(name, opt.device)
        fr_models.append(model)
    threat_model = load_model(opt.threat_name, opt.device)
    for i in range(len(dataset)):
        try:
            idx = i
            img = dataset[idx][0]
            label = dataset[idx][1]
            minila.write(str(i)+'|'+str(idx)+'|')
    
            flag, iters,vector = attack_process(img, opt.sticker, threat_model, opt.threat_name, opt.model_names, fr_models, 
                                label, opt.target, opt.device, opt.width, opt.height, opt.emp_iterations, 
                                opt.di, folder_path, opt.targeted, opt.sapce_thd, pg_m=5, max_iter=10
                                )
            final_img = vector[0]
            final_params = vector[2]
            final_facets = vector[3]
            before_img = vector[4]
            x,y = final_params[0]
            file_path = os.path.join(folder_path, '{}_{}.jpg'.format(i,idx))
            final_img.save(file_path,quality=99)
            minila.write(str(flag)+'|'+str(iters)+'\n')
            minila.flush()
        #break
        except:
            continue
    minila.close()



    # crops_result, crops_tensor = crop_imgs([opt.img], opt.width, opt.height)
    # clean_ts = torch.stack(crops_tensor).to(opt.device)
    # agent = UNet(sgmodel=5).to(opt.device)
    # space = loc_space(crops_result[0],opt.sticker,threshold=120)
    # space_ts = torch.from_numpy(space).to(opt.device)
    '''--------------test agent output--------------'''
    # params_slove = agent_output(agent,clean_ts,space_ts,opt.device)
    # print(params_slove)

    '''--------------test log_probs--------------'''
    # featuremap = agent(clean_ts)
    # pre_actions = vector_processor(featuremap,space_ts,opt.device)
    # log_pis, log_sets = 0, []
    # actions = generate_actions(pre_actions)                                    # sampling
    # print('actions = ',actions)
    # for t in range(len(actions)):
    #     log_prob = pre_actions[t].log_prob(actions[t])
    #     print(log_prob)
    #     log_pis += log_prob
    #     log_sets.append(log_prob)
    # print(log_sets)
    # print(log_pis)

    '''--------------test reward output--------------'''
    # threat_model = load_model(opt.threat_name, torch.device('cpu'))
    # l_sim = reward_output(torch.stack(crops_tensor),threat_model, opt.threat_name, opt.target,opt.device)
    # print(l_sim)

    '''----------------test check all----------------'''
    # threat_model = load_model(opt.threat_name, opt.device)
    # advface_ts = torch.stack(crops_tensor)
    # sim_labels, sim_probs = check_all(advface_ts,threat_model, opt.threat_name,opt.device)
    # print(sim_labels)
    # print(sim_probs[0][sim_labels])
    
    '''----------------test mi attack----------------'''
    #cv2.imwrite("mask.jpg",space*255)
    # params_slove = [[opt.x, opt.y], opt.weights, opt.epsilon]
    # adv_face_ts, adv_face = miattack_face(params_slove, opt.model_names,
    #                         opt.img, opt.label, opt.target, opt.device, opt.sticker,
    #                         opt.width, opt.height, opt.emp_iterations, opt.di, 
    #                         opt.adv_img_folder, opt.targeted)
                