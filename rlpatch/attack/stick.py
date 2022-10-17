from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt
import torch
#import feature

def transparent_back(pic_path):    # make jpg picture's background transparent(png)
    img = cv2.imread(pic_path)     # array
    sticker = Image.open(pic_path) # image
    W,H = sticker.size
    #print(W,H)
    mask = np.zeros(img.shape[:2],np.uint8)
    
    #----------提取前景-------------
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (1,1,450,450)
    # 函数的返回值是更新的 mask, bgdModel, fgdModel
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT) # 提取前景
    #print(mask)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8').transpose()# 前景为1，背景为0
    print('mask2 = ',mask2.shape)

    #print(mask2[200][200])
    sticker = sticker.convert('RGBA')
    for i in range(W):
        for j in range(H):
            color_1 = sticker.getpixel((i,j))
            if(mask2[i][j]==0):   #背景transparent
                color_1 = color_1[:-1] + (0,)
                sticker.putpixel((i,j),color_1)
            else:
                color_1 = color_1[:-1] + (255,)
                sticker.putpixel((i,j),color_1)
    sticker.show()
    sticker.save(pic_path[:-3]+'png')

def make_stick2(backimg,sticker,x,y,factor=1):
    #foreGroundImage = cv2.imread("./lfw_patch/s1.png",-1)
    backimg = np.array(backimg)
    r,g,b = cv2.split(backimg)
    background = cv2.merge([b, g, r])
    #print('background = ',background.shape)
    
    base,_ = make_basemap(background.shape[1],background.shape[0],sticker,x=x,y=y)
    #print('basemap = ',basemap.shape)
    #print('basemap = ',basemap[100][130][3])
    r,g,b,a = cv2.split(base)
    foreGroundImage = cv2.merge([b, g, r,a])
    # cv2.imshow("outImg",foreGroundImage)
    # cv2.waitKey(0)

    ## 先将通道分离
    b,g,r,a = cv2.split(foreGroundImage)
    #得到PNG图像前景部分，在这个图片中就是除去Alpha通道的部分
    foreground = cv2.merge((b,g,r))
    
    #得到PNG图像的alpha通道，即alpha掩模
    alpha = cv2.merge((a,a,a))

    #因为下面要进行乘法运算故将数据类型设为float，防止溢出
    foreground = foreground.astype(float)
    background = background.astype(float)
    
    #将alpha的值归一化在0-1之间，作为加权系数
    alpha = alpha.astype(float)/255
    alpha = alpha * factor
    #print('alpha = ',alpha)
    
    #将前景和背景进行加权，每个像素的加权系数即为alpha掩模对应位置像素的值，前景部分为1，背景部分为0
    foreground = cv2.multiply(alpha,foreground)
    background = cv2.multiply(1-alpha,background)
    
    outarray = foreground + background
    #cv2.imwrite("outImage.jpg",outImage)

    # cv2.imshow("outImg",outImage/255)
    # cv2.waitKey(0)
    b, g, r = cv2.split(outarray)
    outarray = cv2.merge([r, g, b])
    outImage = Image.fromarray(np.uint8(outarray))
    return outImage

def change_sticker(sticker,scale):
    #sticker = Image.open(stickerpath)
    new_weight = int(sticker.size[0]/scale)
    new_height = int(sticker.size[1]/scale)
    #print(new_weight,new_height)
    sticker = sticker.resize((new_weight,new_height),Image.ANTIALIAS)
    return sticker

def make_basemap(width,height,sticker,x,y):
    layer = Image.new('RGBA',(width,height),(255,255,255,0)) # white and transparent
    layer.paste(sticker,(x,y))
    #layer.show()
    base = np.array(layer)
    alpha_matrix = base[:,:,3]
    basemap = np.where(alpha_matrix!=0,1,0)
    return base,basemap

def make_masktensor(width,height,sticker,x,y):
    layer = Image.new('RGBA',(width,height),(255,255,255,0)) # white and transparent
    layer.paste(sticker,(x,y))
    #layer.show()
    base = np.array(layer)
    alpha_matrix = base[:,:,3]
    basemap = np.where(alpha_matrix!=0,1,0)
    rep = cv2.merge((basemap,basemap,basemap))
    #cv2.imwrite("outImage.jpg",rep*255)
    rep = np.transpose(rep,(2,0,1))
    rep = torch.from_numpy(rep)
    rep = torch.unsqueeze(rep,0)
    return rep

#@jit(nopython=True)
def create_space(base):
    #base = sticker
    alpha_matrix = base[:,:,3]
    mask = np.where(alpha_matrix!=0,1,0)
    single_space = int(np.sum(mask))

    searchspace = np.zeros((single_space,2))      # store the coordinate(Image style)
    pack_searchspace = np.ones((base.shape[0],base.shape[1]))*-1 # valid=-1, unvalid=-2
    k = 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if(mask[i][j] == 1):
                searchspace[k] = (j,i)
                pack_searchspace[i][j] = k
                k = k + 1
    return mask,searchspace,pack_searchspace

def collect_index(mask):
    repeat = np.stack((mask,mask,mask),axis=2)
    varlen = repeat.reshape(-1).size
    c = repeat.reshape(-1)       # shape(varlen,)
    var_list = np.where(c==1)[0].astype(np.int32) # shape(varlen,)
    # print(var_list)
    # print('varlen = ',varlen,'varlist=',len(var_list),repeat.shape,c.shape,mask.shape)
    return varlen, var_list

if __name__=="__main__":
    layer = Image.new('RGBA',(100,100),(255,255,255,0)) # white and transparent
    st = Image.new('RGBA',(20,20),(255,255,255,255))
    x,y=torch.tensor(5).numpy(),torch.tensor(7).numpy()
    layer.paste(st,(x,y))
    layer.save('x.png')
    # pic_path = './lfw_patch/s1.jpg'
    # transparent_back(pic_path)

    # pic_dir = "./lfw_patch/6179.jpg"
    # stickerpath = './lfw_patch/s1.png'
    # x = 111
    # y = 75        #Image style
    # scale = 18
    # img = Image.open(pic_dir)

    # facemask = feature.make_mask(img)
    # sticker = change_sticker(stickerpath = stickerpath,scale = scale)
    # base,basemap = make_basemap(width=facemask.shape[1],height=facemask.shape[0],sticker=sticker,x=x,y=y)
    # outImage = make_stick2(backimg=img,sticker=sticker,x=x,y=y,factor=0.2)
    # outImage.show()


    # num_space = np.sum(facemask).astype(int)
    # area = np.sum(basemap)
    # overlap = facemask * basemap
    # retain = np.sum(overlap)
    # print('num_space = ',num_space)
    # print('area = ',area)
    # print('retain = ',retain)

    # searchspace = np.zeros((num_space,2)) # store the coordinate(Image style)
    # k = 0
    # for i in range(facemask.shape[0]):
    #     for j in range(facemask.shape[1]):
    #         if(facemask[i][j] == 1):
    #             searchspace[k] = (j,i)
    #             k = k + 1

    # cv2.imshow("face",facemask)
    # cv2.imshow("st",basemap)
    # # cv2.imshow("outImg",outImage/255)
    # cv2.waitKey(0)

    # backimg = Image.open('/home/guoying/rlpatch/mtcnn_pytorch_master/align.jpg')
    # sticker = Image.new('RGBA',(60,30),(255,255,255,255))
    # x,y = 50,25
    # out = make_stick2(backimg,sticker,x,y,factor=1)
    # out.save('temp.jpg')
    # rep = make_masktensor(160,160,sticker,x,y)
    # print(rep.shape)
    # print(torch.sum(rep))
