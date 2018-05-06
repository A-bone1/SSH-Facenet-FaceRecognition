# SSH Facenet Face Recogniton

## Introduction
This repository includes a face recognition and detection system based on **SSH** face detection network and **facenet** feature extraction network.

This project can run on  Ubuntu 16.04, using the python3.5 or python2.7 environment.The SSH network is built using caffe and the facenet network is built using Tensorflow.**The two networks can be used combination or independently**.

## Contents

1. [SSH Face Detection](#SSH)
2. [Facenet Face Recognition](#Facenet)
3. [Open face recognition and detection remote service](#Open)


<a name="SSH"> </a>
### SSH Face Detection
This project was modified based on the author's [source code](https://github.com/mahyarnajibi/SSH) of the SSH network.

More details of the network can be found in this paper:["SSH: Single Stage Headless Face Detecto"](https://arxiv.org/abs/1708.03979)
#### 1. Install
(1) caffe-GPU and pycaffe are required

(2) Install python requirements:
```
cd SSH
pip install -r requirements.txt
```
(3)  Run ```make``` in the ```SSH\lib``` directory:
```
cd lib
make
```
#### 2. Running the demo
To run the demo, first, you need to download the provided pre-trained *SSH* model. Running the following script downloads the *SSH* model into its default directory path:
```

bash SSH/scripts/download_ssh_model.sh
```
Or you can download it from the [Baidu cloud disk](https://pan.baidu.com/s/1KDiuJ1GA3WqoAMg49aRsiw) I provided.

By default, the model is saved into a folder named ```SSH/data/SSH_models``` .

After downloading the *SSH* model, you can run the demo with the default configuration as follows:
```
cd SSH
python demo.py
```
If everything goes well, the following detections should be saved as ```data/demo/demo_detections_SSH.png```.
![image](https://github.com/A-bone1/SSH-Facenet-FaceRecognition/blob/master/SSH/data/demo/demo_detections_SSH.png)
#### 3.Training a model
If you want to train your own face detection model, you can see a detailed tutorial on this [github](https://github.com/mahyarnajibi/SSH)

<a name="Facenet"></a>
### Facenet Face Recognition
More details of the network can be found in this paper:[FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)
#### 1. Introduction
The face recognition system first uses SSH for face detection, then the cropped picture is input into the facrnet and encoded into a 128-dimensional vector, and then compared with the database, each two faces get a distance.The database face,which has the smallest distance and the distance is less than the preset threshold, is the recognition result.

#### 2. Install
You can quickly start facenet with pretrained Keras model (trained by MS-Celeb-1M dataset).

-Download model from [here](https://pan.baidu.com/s/1KDiuJ1GA3WqoAMg49aRsiw) and save it in model/keras/

You can also create Keras model from pretrained tensorflow model.

-Download model from [here](https://pan.baidu.com/s/1KDiuJ1GA3WqoAMg49aRsiw) and save it in model/tf/
-Convert model for Keras in [facenet\notebook\tf_to_keras.ipynb](https://github.com/A-bone1/SSH-Facenet-FaceRecognition/blob/master/facenet/notebook/tf_to_keras.ipynb)

#### 3. Running the demo
(1) use SSHDetec Network
-Open SSH face detection remote service
```
python SSH/service.py
```
-Run the Jupyter Notebook [SSH_facenet.ipynb](https://github.com/A-bone1/SSH-Facenet-FaceRecognition/blob/master/facenet/notebook/SSH_facenet.ipynb) in facenet\notebook\
(2)Only run Facenet Network

-Run the Jupyter Notebook [demo-images.ipynb](https://github.com/A-bone1/SSH-Facenet-FaceRecognition/blob/master/facenet/notebook/demo-images%20.ipynb) in facenet\notebook\ in facenet\notebook\

#### 3.Training a model
-If you want to train your own face detection model, you can see a detailed tutorial on this [Wiki](https://github.com/davidsandberg/facenet/wiki/Classifier-training-of-inception-resnet-v1)

-Convert model for Keras in [facenet\notebook\tf_to_keras.ipynb](https://github.com/A-bone1/SSH-Facenet-FaceRecognition/blob/master/facenet/notebook/tf_to_keras.ipynb)

<a name="Open"> </a>
### Open face recognition and detection remote service

(1)Open a command line window(ctrl+alt+T/Wins +r),and open the face detection service.
```
python SSH/service.py
```
(2)Open a command line window(ctrl+alt+T/Wins +r),and open the face recognition service.
```
python facenet/Verificationservice.py
```
(3)Use http protocol to access services.The demo is in the [serviceTest\serviceTest.ipynb](https://github.com/A-bone1/SSH-Facenet-FaceRecognition/blob/master/serviceTest/serviceTest.ipynb)
![image](https://github.com/A-bone1/SSH-Facenet-FaceRecognition/blob/master/serviceTest/demoImg/detectService.jpg)
![image](https://github.com/A-bone1/SSH-Facenet-FaceRecognition/blob/master/serviceTest/demoImg/recogService.jpg)
