import torch
from PIL import Image, ImageDraw
from torchvision import datasets, transforms
#from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import scipy
import torchvision.models as models
import os
import cv2
import json
# from tencentcloud.common import credential
# from tencentcloud.common.profile.client_profile import ClientProfile
# from tencentcloud.common.profile.http_profile import HttpProfile
# from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
# from tencentcloud.iai.v20200303 import iai_client, models
# from io import BytesIO
# import base64

def search_tencent(imgcode):
    try: 
        cred = credential.Credential("AKIDVnqsxqgqIWVKMPKHKL6hPtANLUaQxvIP", "0kIrD0n4apmUyB7o4oGRN4Vh5C2gHjxO") 
        httpProfile = HttpProfile()
        httpProfile.endpoint = "iai.tencentcloudapi.com"

        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        client = iai_client.IaiClient(cred, "ap-beijing", clientProfile) 

        req = models.SearchFacesRequest()
        params = {
            "GroupIds": [ "1" ],
            "Image": imgcode,
            "MaxPersonNum": 50
        }
        req.from_json_string(json.dumps(params))

        resp = client.SearchFaces(req) 
        return resp.to_json_string()

    except TencentCloudSDKException as err: 
        print(err) 
        return "noface"

def check_tct(image_perturbed,basic=[[0,0],[0,0]],num_classes=5752):
    num = len(image_perturbed)
    typess, percent = [], []
    for i in range(num):
        print('--make prediction for {}-th img--'.format(i),end='\r')
        tage = np.ones((1,num_classes)).astype(np.float32)
        img = image_perturbed[i]
        buff = BytesIO()
        img.save(buff, format="JPEG")
        img_str = base64.b64encode(buff.getvalue())
        imgcode = img_str.decode()
        #print(img_str)
        info = search_tencent(imgcode)
        if(info != "noface"):
            info = json.loads(info)
            people = info["Results"][0]["Candidates"]
            #print(people)
            tage = tage * float(people[len(people)-1]["Score"])
            for sgp in range(len(people)):
                pid = int(people[sgp]["PersonId"][4:])
                pscore = float(people[sgp]["Score"])
                tage[0][pid] = pscore
                           
            cla = [int(people[0]["PersonId"][4:]),int(people[1]["PersonId"][4:])]
            typess.append(cla)
        else:
            typess.append(basic[0])
            tage = basic[1]

        if(i==0):
            percent=tage
        else:
            percent = np.vstack((percent,tage))
    return typess, np.array(percent)

def reward_tct(adv_face,target,truelabel,basic=[[0,0],[0,0]],num_classes=5752):
    buff = BytesIO()
    adv_face.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue())
    imgcode = img_str.decode()
    #print(img_str)
    info = search_tencent(imgcode)
    tage = np.ones((1,num_classes)).astype(np.float32)
    if(info != "noface"):
        info = json.loads(info)
        people = info["Results"][0]["Candidates"]
        #print(people)
        for sgp in range(len(people)):
            pid = int(people[sgp]["PersonId"][4:])
            pscore = float(people[sgp]["Score"])
            tage[0][pid] = pscore
    #return tage[0][target]
    return tage[0][truelabel] - tage[0][target] 
    #return tage[0][5751] - tage[0][target] 


# def predict_tct(image_perturbed,num_classes=5749,basic=[[0,0],[0,0]],num_classes=5749):
def predict_tct(image_perturbed,num_classes=5749,basic=[[0,0],[0,0]]):
    """
        image_perturbed: list[PIL]
    """
    num = len(image_perturbed)
    typess = []
    for i in range(num):
        tage = np.ones((1,num_classes)).astype(np.float32)
        print('--make prediction for {}-th img--'.format(i),end='\r')
        # img_arr = image_perturbed[i]
        # img = Image.fromarray(np.uint8(img_arr))
        # adv_final = (img_arr*255).astype(np.uint8)
        # img = Image.fromarray(adv_final)
        #img.save('a.jpg')

        img = image_perturbed[i]
        buff = BytesIO()
        img.save(buff, format="PNG")
        img_str = base64.b64encode(buff.getvalue())
        imgcode = img_str.decode()
        #print(img_str)
        info = search_tencent(imgcode)
        if(info != "noface"):
            info = json.loads(info)
            people = info["Results"][0]["Candidates"]
            #print(people)
            tage = tage * float(people[len(people)-1]["Score"])
            for sgp in range(len(people)):
                pid = int(people[sgp]["PersonId"][4:])
                pscore = float(people[sgp]["Score"])
                tage[0][pid] = pscore
                           
            cla = [int(people[0]["PersonId"][4:]),int(people[1]["PersonId"][4:])]
            typess.append(cla)
        else:
            typess.append(basic[0])
            tage = basic[1]

        if(i==0):
            percent=tage
        else:
            percent = np.vstack((percent,tage))
    return typess, percent/100
       
if __name__ == '__main__':
    # data_dir = '/home/guoying/patch/celeba_align/CelebA/train'
    # dataset = datasets.ImageFolder(data_dir)
    # dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
    # print(dataset.idx_to_class[4410])
    # log = open('./log.txt','a')
    # with open('/home/guoying/patch/tc_cloud/scope.txt') as f:
    #     inds = [int(line[:-1].strip()) for line in f.readlines()]
    # for i in range(len(inds)):
    #     idx = inds[i]
    #     #idx = 3315
    #     img = dataset[idx][0]
    #     true_label = dataset[idx][1]
    #     #img = Image.open('/home/guoying/patch/tc_cloud/celeba_unt/bs12_3051.jpg')
    #     basic = [[12,11],[100,80]]
    #     typess, percent = predict_type([img],basic)
    #     sticker_name = 'bs12'
    #     log.write('0'+' / '+str(idx)+' / '+str(sticker_name)+' / '+str(true_label)+' / '\
    #             + str(typess)+' / '+str(percent)+'\n')
    #     print(idx,typess, percent)
    # img = Image.open('/home/guoying/patch/tc_cloud/celeba_unt/bs12_25408.jpg')
    #img = dataset[89328][0]
    #img=Image.open('/home/ubuntu/Documents/RY2020/tc_cloud/check/temp_t2400_nst3_5750_gy_29.jpg')
    #img2=Image.open('/home/ubuntu/Documents/RY2020/tc_cloud/check/temp_t8194_nst3_5750_gy_19.jpg')
    face = Image.open('/home/guoying/decouple/physical/GuoYing/1.jpg')

    access = Image.open('/home/guoying/decouple/face/mask/mask3.png')
    access = stick.change_sticker(access,0.9)
    face = stick.make_stick2(backimg=face,sticker=access,x=191,y=216,factor=1)
    #face = Image.open('/home/guoying/decouple/physical/GuoYing/1.jpg')
    basic = [[12,11],[100,80]]
    typess, percent = predict_zoo([face],basic,8210)
    print(typess, percent[0][typess[0][0]],percent[0][typess[0][1]])
    print(percent.shape)

# a = {"Results": [{"Candidates": [{"PersonId": "27", "FaceId": "3912166679774240590", \
#                               "Score": 100, "PersonName": 'null', "Gender": 'null', \
#                               "PersonGroupInfos": 'null'}, \
#                              {"PersonId": "36", "FaceId": "3912166724669027643",\
#                               "Score": 85.72687530517578,"PersonName": 'null',\
#                               "Gender": 'null', "PersonGroupInfos": 'null'}], \
#               "FaceRect": {"X": 42, "Y": 61, "Width": 88, "Height": 122}, "RetCode": 0}], \
#  "FaceNum": 8180, \
#  "FaceModelVersion": "3.0",\
#  "RequestId": "0d5fec22-226b-4940-99d9-240cecb2cc30"}
