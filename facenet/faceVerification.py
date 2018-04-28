import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from imageio import imread
from skimage.transform import resize
from scipy.spatial import distance
from keras.models import load_model
import requests

image_size = 160

model_path = 'model/keras/model/facenet_keras.h5'
model = load_model(model_path)

def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output


def load_and_align_images(img,im, margin):
    aligned_images = []


    url = 'http://127.0.0.1:9004'
    #files = {'im': open(filepath, 'rb')}
    files = {'im':im}
    res = requests.post(url, files=files)
        # print res.text
    boxes = eval(res.text)

    for box in boxes:
        if box[4] < 0.9:
            continue
        box1 = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
        if box1[2] >= box1[3]:
            diff = box1[2] - box1[3]
            box1[3] = box1[2]
            box1[1] = box1[1] - diff / 2
        else:
            diff = box1[3] - box1[2]
            box1[2] = box1[3]
            box1[0] = box1[0] - diff / 2
        print(box1)

        (x, y, w, h) = box1
        margin = 0

        cropped = img[int(y) - margin // 2:int(y) + int(h) + margin // 2,
                      int(x) - margin // 2:int(x) + int(w) + margin // 2, :]

        aligned = resize(cropped, (image_size, image_size), mode='reflect')
        aligned_images.append(aligned)
    print(len(aligned_images))
    return np.array(aligned_images)

def calc_embs(img,im,margin=10, batch_size=1):
    aligned_images = prewhiten(load_and_align_images(img,im,margin))
    pd = []
    for start in range(0, len(aligned_images), batch_size):
        pd.append(model.predict_on_batch(aligned_images[start:start+batch_size]))
    embs = l2_normalize(np.concatenate(pd))

    return embs

def calc_dist(img_name0, img_name1):
    return distance.euclidean(data[img_name0]['emb'], data[img_name1]['emb'])

def calc_dist_plot(img_name0, img_name1):
    print(calc_dist(img_name0, img_name1))
    plt.subplot(1, 2, 1)
    plt.imshow(imread(data[img_name0]['image_filepath']))
    plt.subplot(1, 2, 2)
    plt.imshow(imread(data[img_name1]['image_filepath']))

# image_dir_basepath = 'data/images/'
# names = ['LarryPage', 'MarkZuckerberg', 'BillGates','Aaron','000','001','002']

def distanceResult(img1,im1,img2,im2):
    a1=calc_embs(img1,im1, margin=10, batch_size=1)
    a2=calc_embs(img2,im2, margin=10, batch_size=1)
    return (distance.euclidean(a1, a2))

img1 = cv2.imread('data/images/000/000_0.bmp')
img2 = cv2.imread('data/images/000/000_2.bmp')
im1=   open('data/images/000/000_0.bmp', 'rb')
im2=   open('data/images/000/000_2.bmp', 'rb')
print(distanceResult(img1,im1,img2,im2))