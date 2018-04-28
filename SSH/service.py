# -*- coding: UTF-8 -*-
from __future__ import print_function
import tornado.ioloop
import tornado.web
import cv2
import sys
import argparse
import numpy as np
#np.set_printoptions(suppress=True)
from SSH.test import detect_im
sys.path.append("/home/czh/caffe/python")
sys.path.append("/home/czh/caffe/python/caffe")

# Add caffe and lib to the paths
import caffe
from utils.get_config import cfg
import os


class MainHandler(tornado.web.RequestHandler):

    def post(self):
        item = self.request.files["im"][0]
        im = item["body"]

#        im_name = item["filename"]

	#img = self.get_argument('im')
	#im = eval(img.decode("base64"))   # dtype为int32
    	#im = np.array(img, dtype=np.uint8)

        im = cv2.imdecode(np.fromstring(im, np.uint8), cv2.IMREAD_COLOR)
	
        cls_dets = detect_im(net, im)
        self.set_header("Pragma", "no-cache")
        self.set_header("Expires", "0")
        self.set_header("Cache-Control", "no-cache")
        self.set_header("Cache-Control", "no-store")
        self.write((str(cls_dets.tolist())))


def make_app():
    return tornado.web.Application([
        (r"/", MainHandler)])


if __name__ == "__main__":
    if not cfg.DEBUG:
        # Suppress Caffe (it does not affect training, only test and demo)
        os.environ['GLOG_minloglevel'] = '3'

    # Loading the network
    caffe.set_mode_gpu()
    caffe.set_device(0)
    assert os.path.isfile("SSH/models/test_ssh.prototxt"), 'Please provide a valid path for the prototxt!'
    assert os.path.isfile("data/SSH_models/SSH.caffemodel"), 'Please provide a valid path for the caffemodel!'

    print('Loading the network...', end="")
    net = caffe.Net("SSH/models/test_ssh.prototxt", "data/SSH_models/SSH.caffemodel", caffe.TEST)
    net.name = 'SSH'
    print('Done!')

    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, help="port, default is 9004", default=9004)
    params = parser.parse_args()
    app = make_app()
    app.listen(params.p)
    print("listen port %d" % params.p)
    tornado.ioloop.IOLoop.instance().start()
