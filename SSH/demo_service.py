# ------------------------------------------
# SSH: Single Stage Headless Face Detector
# Demo
# by Mahyar Najibi
# ------------------------------------------

from __future__ import print_function

from argparse import ArgumentParser
import sys
# Add caffe and lib to the paths
if not 'caffe-ssh/python' in sys.path:
    sys.path.insert(0,'caffe-ssh/python')
if not 'lib' in sys.path:
    sys.path.insert(0,'lib')
from utils.get_config import cfg

if not cfg.DEBUG:
    import os
    # Suppress Caffe (it does not affect training, only test and demo)
    os.environ['GLOG_minloglevel']='3'

import os

import numpy as np
np.set_printoptions(suppress=True)
from utils.get_config import cfg_from_file, cfg, cfg_print
from utils.get_config import cfg, get_output_dir
from nms.nms_wrapper import nms
from utils.test_utils import _get_image_blob, _compute_scaling_factor, visusalize_detections



sys.path.append("/home/czh/caffe/python")
sys.path.append("/home/czh/caffe/python/caffe")
import caffe
import cv2

def parser():
    parser = ArgumentParser('SSH Demo!')
    parser.add_argument('--im',dest='im_path',help='Path to the image',
                        default='data/demo/demo.jpg',type=str)
    parser.add_argument('--gpu',dest='gpu_id',help='The GPU ide to be used',
                        default=0,type=int)
    parser.add_argument('--proto',dest='prototxt',help='SSH caffe test prototxt',
                        default='SSH/models/test_ssh.prototxt',type=str)
    parser.add_argument('--model',dest='model',help='SSH trained caffemodel',
                        default='data/SSH_models/SSH.caffemodel',type=str)
    parser.add_argument('--out_path',dest='out_path',help='Output path for saving the figure',
                        default='data/demo',type=str)
    parser.add_argument('--cfg',dest='cfg',help='Config file to overwrite the default configs',
                        default='SSH/configs/default_config.yml',type=str)
    return parser.parse_args()

def forward_net(net, blob, im_scale, pyramid='False'):
    """
    :param net: the trained network
    :param blob: a dictionary containing the image
    :param im_scale: the scale used for resizing the input image
    :param pyramid: whether using pyramid testing or not
    :return: the network outputs probs and pred_boxes (the probability of face/bg and the bounding boxes)
    """
    # Adding im_info to the data blob
    blob['im_info'] = np.array(
        [[blob['data'].shape[2], blob['data'].shape[3], im_scale]],
        dtype=np.float32)

    # Reshape network inputs
    net.blobs['data'].reshape(*(blob['data'].shape))
    net.blobs['im_info'].reshape(*(blob['im_info'].shape))

    # Forward the network
    net_args = {'data': blob['data'].astype(np.float32, copy=False),
                      'im_info': blob['im_info'].astype(np.float32, copy=False)}

    blobs_out = net.forward(**net_args)

    if pyramid:
        # If we are in the pyramid mode, return the outputs for different modules separately
        pred_boxes = []
        probs = []
        # Collect the outputs of the SSH detection modules
        for i in range(1,4):
            cur_boxes = net.blobs['m{}@ssh_boxes'.format(i)].data
            # unscale back to raw image space
            cur_boxes = cur_boxes[:, 1:5] / im_scale
            # Repeat boxes
            cur_probs = net.blobs['m{}@ssh_cls_prob'.format(i)].data
            pred_boxes.append(np.tile(cur_boxes, (1, cur_probs.shape[1])))
            probs.append(cur_probs)
    else:
        boxes = net.blobs['ssh_boxes'].data.copy()
        # unscale back to raw image space
        boxes = boxes[:, 1:5] / im_scale
        probs = blobs_out['ssh_cls_prob']
        pred_boxes = np.tile(boxes, (1, probs.shape[1]))

    return probs, pred_boxes


def detect_im(net, im, thresh=0.05):

    sys.stdout.flush()

    im_scale = _compute_scaling_factor(im.shape,cfg.TEST.SCALES[0],cfg.TEST.MAX_SIZE)
    im_blob = _get_image_blob(im,[im_scale])
    probs, boxes = forward_net(net,im_blob[0],im_scale,False)
    boxes = boxes[:, 0:4]

    inds = np.where(probs[:, 0] > thresh)[0]
    probs = probs[inds, 0]
    boxes = boxes[inds, :]
    dets = np.hstack((boxes, probs[:, np.newaxis])) \
            .astype(np.float32, copy=False)
    keep = nms(dets, cfg.TEST.NMS_THRESH)
    cls_dets = dets[keep, :]
    #print(cls_dets)
    return cls_dets




if __name__ == "__main__":

    # Parse arguments
    args = parser()

    # Load the external config
    if args.cfg is not None:
        cfg_from_file(args.cfg)
    # Print config file
    cfg_print(cfg)

    # Loading the network
    cfg.GPU_ID = args.gpu_id
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    assert os.path.isfile(args.prototxt),'Please provide a valid path for the prototxt!'
    assert os.path.isfile(args.model),'Please provide a valid path for the caffemodel!'

    print('Loading the network...', end="")
    net = caffe.Net(args.prototxt, args.model, caffe.TEST)
    net.name = 'SSH'
    print('Done!')

    # Read image
    assert os.path.isfile(args.im_path),'Please provide a path to an existing image!'

    im = cv2.imread(args.im_path)
    # Perform detection
    cls_dets= detect_im(net,im)

    print (cls_dets)





