from numpy.core.fromnumeric import put
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

def add_face_local(database_path, name, batch_size):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    fr_model = load_model(name,device).eval().to(device)
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
    output = []
    # for i in range(0,input.shape[0],batch_size):
    #     output.append(fr_model(input[:min(batch_size,input.shape[0]-i)]).detach().cpu())
    for i in range(0,input.shape[0],batch_size):
        output.append(fr_model(input[i:i+min(batch_size,input.shape[0]-i)]).detach().cpu())
    output = torch.cat(output, dim=0)
    print(output.shape)
    embedding_sets = output
    print(embedding_sets.shape)
    
    if not os.path.exists("stmodels"):
        os.makedirs("stmodels")
    if not os.path.exists("stmodels/{}".format(name)):
        os.makedirs("stmodels/{}".format(name))
    joblib.dump(embedding_sets,'stmodels/{}/embeddings_{}.pkl'.format(name,name))
    return      

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_path', type=str, default='../lfw_images', help='database path')
    parser.add_argument('--model_name', type=str, default='facenet', help='model names list')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    opt = parser.parse_args()
    # add_face_local(opt.database_path, opt.model_name, opt.batch_size)
    add_face_local("../lfw", 'arcface50', 32)
