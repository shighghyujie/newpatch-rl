from mtcnn_pytorch_master.src.detector import detect_faces
from mtcnn_pytorch_master.src.utils import show_bboxes
# from src.detector import detect_faces
# from src.utils import show_bboxes
from PIL import Image
import numpy as np
from skimage import transform as trans
import cv2
import skimage.io as io

def detect(image):
    bounding_boxes, landmarks = detect_faces(image)
    #print(landmarks)
    image = show_bboxes(image, bounding_boxes, landmarks)
    #image.save('a.jpg')
    #image.show()
    return bounding_boxes, landmarks

def preprocess(img, landmark,w,h):
    image_size = [h,w]
    ruler = float(max(h,w))
    src = ruler/112.*np.array([
		[38.2946, 51.6963],
		[73.5318, 51.5014],
		[56.0252, 71.7366],
		[41.5493, 92.3655],
		[70.7299, 92.2041] ], dtype=np.float32)
    dst = landmark.astype(np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]

    warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)
    return warped

def crop_face(image,w,h):
    #w,h = image.size
    bounding_boxes, landmarks = detect(image)
    #if(len(landmarks)>1):
    landmarks = landmarks[0]
    #print(landmarks)
    landmarks = landmarks.reshape((2,5)).T
    img = np.array(image)
    warped = preprocess(img,landmarks,w,h)
    #print(warped.shape)
    crop = Image.fromarray(np.uint8(warped))
    return crop

if __name__ == "__main__":
    image = Image.open('/home/guoying/decouple/physical/GuoYing/1.jpg')
    w,h = 160,160
    crop = crop_face(image,w,h)
    crop.save('align.jpg')
    
    #io.imsave('aligned2.jpg',warped)
    # pic = Image.open('/home/guoying/rlpatch/mtcnn-pytorch-master/aligned.jpg')
    # mask = feature.make_mask(pic)
    # cv2.imwrite('mask.jpg',mask*255)
