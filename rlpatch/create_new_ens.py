from numpy.core.fromnumeric import put
from rl_solve.tct import add_face
import argparse
from attack.tiattack import load_model,crop_imgs
from torchvision import transforms
import torch
import torch.nn as nn
import joblib
import os
from PIL import Image

trans = transforms.Compose([
                transforms.ToTensor(),
            ])

def add_face_local(database_path,pre):
    model_name = ['facenet','arcface34','cosface34','cosface50','arcface50','mobilefacenet']
    device = torch.device('cpu')
    for name in model_name:
        fr_model = load_model(name,device).eval().to(device)
        if pre:
            anchors = joblib.load('stmodels/{}/embeddings_{}_5752.pkl'.format(name,name))        
        dir = database_path
        imgs = []
        for each in os.listdir(dir):
            for each1 in os.listdir(dir+"/"+each):
                filename = dir + "/" + each + "/" + each1
                img = Image.open(filename)
                imgs.append(img)
                break
        if name == 'facenet':
            w,h = 160,160
        else:
            w,h = 112,112
        crops_result, crops_tensor = crop_imgs(imgs,w,h)
        input = torch.stack(crops_tensor).to(device)
        input = nn.functional.interpolate(input, (w, h), mode='bilinear', align_corners=False)
        print(input.shape)
        output = fr_model(input).detach().cpu()
        print(output.shape)
        if pre:
            embedding_sets = torch.cat((anchors,output))
        else:
            embedding_sets = output
        print(embedding_sets.shape)
        
        if not os.path.exists("./stmodels"):
            os.makedirs("./stmodels")
        if pre:
            joblib.dump(embedding_sets, 'stmodels/{}/embeddings_{}_{}.pkl'.format(name,name,str(5751+len(imgs))))
        else:
            joblib.dump(embedding_sets,'stmodels/{}/embeddings_{}_{}.pkl'.format(name,name,len(imgs)))
    return      

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_path', type=str, default='../lfw_images', help='database path')
    parser.add_argument('--new_add', type=int, default=1, help='new or add ens')
    opt = parser.parse_args()
    pre = True if opt.new_add == 1 else False
    add_face(opt.database_path,pre)
    add_face_local(opt.database_path,pre)
