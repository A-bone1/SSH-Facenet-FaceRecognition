# -*- coding: UTF-8 -*-
#本程序所有基本函数与Veri_test一致
from __future__ import print_function
import tornado.ioloop
import tornado.web
import argparse
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
import urllib3

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

def load_and_align_images(img,box, margin):
    aligned_images = []
    boxes = box
    for box in boxes:
        if box[4] < 0.9:
            continue
        box1 = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
        if box1[2] >= box1[3]:
            diff = box1[2] - box1[3]
            box1[3] = box1[2]
            box1[1] = box1[1] - diff / 2
            if box1[1]<0:
                box1[1]=0
        else:
            diff = box1[3] - box1[2]
            box1[2] = box1[3]
            box1[0] = box1[0] - diff / 2
            if box1[0]<0:
                box1[0]=0
        (x, y, w, h) = box1
        margin = 0
        cropped = img[int(y) - margin // 2:int(y) + int(h) + margin // 2,
                      int(x) - margin // 2:int(x) + int(w) + margin // 2, :]
        aligned = resize(cropped, (image_size, image_size), mode='reflect')
        aligned_images.append(aligned)
    return np.array(aligned_images)

def calc_embs(img,box,margin=10, batch_size=1):
    aligned_images = prewhiten(load_and_align_images(img,box,margin))
    embs = []
    for start in range(0, len(aligned_images), batch_size):
        emb=l2_normalize(model.predict_on_batch(aligned_images[start:start+batch_size]))
        embs.append(emb)
    return embs

def distanceResult(img1,img2,box1,box2):
    a1=calc_embs(img1,box1, margin=10, batch_size=1)
    a2=calc_embs(img2,box2, margin=10, batch_size=1)
    distances=[]
    for emb in a2:
        distances.append(distance.euclidean(a1, emb))
    return (distances)

class MainHandler(tornado.web.RequestHandler):

    def post(self):
        item1 = self.request.files["im1"][0]
        item2 = self.request.files["im2"][0]
        im1 = item1["body"]
        im2 = item2["body"]
        http = urllib3.PoolManager()
        box1 = http.request('POST', params.detect, fields={'im': (item1['filename'], im1)})
        box2 = http.request('POST', params.detect, fields={'im': (item2['filename'], im2)})
        img1 = cv2.imdecode(np.fromstring(im1, np.uint8), cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(np.fromstring(im2, np.uint8), cv2.IMREAD_COLOR)
        box2=eval(box2.data)
        box1=eval(box1.data)
        m=[]
#去除文本框置信率小于0.94的实例
        for i,x in enumerate(box2):
            if x[4]<0.94:
                m.append(i)
        m.sort(reverse=True)
        for x in m:
            box2.remove(box2[x])
        if len(box2)==0:
            result=[100]
        else:
            print("==============")
            result=distanceResult(img1,img2,box1,box2)
        # cls_dets = detect_im(net, im)
        self.set_header("Pragma", "no-cache")
        self.set_header("Expires", "0")
        self.set_header("Cache-Control", "no-cache")
        self.set_header("Cache-Control", "no-store")
        self.write(str(result))


def make_app():
    return tornado.web.Application([
        (r"/", MainHandler)])


if __name__ == "__main__":

    # Loading the network
    image_size = 160
    model_path = 'model/keras/model/facenet_keras.h5'
    #model_path = 'D://project//facenet-tensorflow//model//keras//model//facenet_keras.h5'
    model = load_model(model_path)
   # model.summary()


    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, help="port, default is 9010", default=9010)
    parser.add_argument("--detect", type=str, help="detect api url, default is http://127.0.0.1:9004", default="http://127.0.0.1:9004")
    params = parser.parse_args()
    app = make_app()
    app.listen(params.p)
    print("listen port %d" % params.p)
    tornado.ioloop.IOLoop.instance().start()

