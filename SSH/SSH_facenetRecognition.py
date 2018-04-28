import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

from imageio import imread
from skimage.transform import resize
from scipy.spatial import distance
from keras.models import load_model
import demo_service
import sys
sys.path.append("/home/czh/caffe/python")
sys.path.append("/home/czh/caffe/python/caffe")

# Add caffe and lib to the paths
import caffe
from utils.get_config import cfg




image_dir_basepath = '/home/czh/facenet-tensorflow/notebook/train/'
names = [str(i).zfill(3) for i in range (0,100)]
image_size = 160


model_path = '/home/czh/facenet-tensorflow/model/keras/model/facenet_keras.h5'
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


def load_and_align_images(filepaths, margin):
    aligned_images = []
    for filepath in filepaths:
        img = cv2.imread(filepath)
        cls_dets = demo_service.detect_im(net, img)

        # print res.text
        boxes = list(cls_dets)

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


def calc_embs(filepaths, margin=10, batch_size=1):
    aligned_images = prewhiten(load_and_align_images(filepaths, margin))
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

if not cfg.DEBUG:
    # Suppress Caffe (it does not affect training, only test and demo)
    os.environ['GLOG_minloglevel'] = '3'

    # Loading the network
caffe.set_mode_gpu()
caffe.set_device(0)
assert os.path.isfile("SSH/models/test_ssh.prototxt"), 'Please provide a valid path for the prototxt!'
assert os.path.isfile("data/SSH_models/SSH.caffemodel"), 'Please provide a valid path for the caffemodel!'

print('Loading the network...')
net = caffe.Net("SSH/models/test_ssh.prototxt", "data/SSH_models/SSH.caffemodel", caffe.TEST)
net.name = 'SSH'
print('Done!')


data = {}
for name in names:
    image_dirpath = image_dir_basepath + name

    image_filepaths = [os.path.join(image_dirpath, f) for f in os.listdir(image_dirpath)]
    print(image_filepaths)
    embs = calc_embs(image_filepaths)
    for i in range(len(image_filepaths)):
        data['{}{}'.format(name, i)] = {'image_filepath': image_filepaths[i],
                                        'emb': embs[i]}


def faceRecognition(image_path, data):

    embs = calc_embs(image_path)
    min_dist = 100

    for name in names:
        dist_array = []
        for k, v in data.items():
            if k.startswith(name):
                dist_array.append(calc_dist_self(embs, k))
        a = 0
        for i in dist_array:
            a += i
        dist = a / len(dist_array)

        if dist < min_dist:
            min_dist = dist
            identity = name


    print(min_dist)
    if min_dist > 0.9:
        print("Not in the database.")
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))

    return min_dist, identity


def calc_dist_self(emb, img_name1):
    return distance.euclidean(emb, data[img_name1]['emb'])






image_dir_basepath = '/home/czh/facenet-tensorflow/notebook/test/'
count=0
for name in names:
    image_dirpath = image_dir_basepath + name
    for f in os.listdir(image_dirpath):
        image_filepath = os.path.join(image_dirpath, f)
        #print(image_filepath)
        dist,predict=faceRecognition([str(image_filepath)],data)
        print (predict)
        print (name)
        if predict==name:
            count+=1
print(count)
