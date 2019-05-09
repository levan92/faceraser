#!/usr/bin/python3
import cv2
import time
import argparse
import pickle
import os
import numpy as np
from copy import deepcopy
from datetime import datetime
from tqdm import tqdm

from utils.shower import Shower
from utils.noise_drawer import noiser
from utils.noise_face import noise_face
from faceDet.mobnet_dlib import Mobnet_FD as FaceDet


parser = argparse.ArgumentParser()
parser.add_argument('cen_images', help='Directory for censored images', type=str)
parser.add_argument('dir_images', help='Filepath to directory of images to censor', type=str)
args = parser.parse_args()

assert os.path.isdir(args.dir_images),'dir given is not a directory!'
img_paths = []
for images in os.listdir(args.dir_images):
    img_paths.append(os.path.join(args.dir_images, images))
print('{} images in all'.format(len(img_paths)))

pose = os.path.basename(args.dir_images)
if not os.path.exists( args.cen_images ):
    os.makedirs( args.cen_images )
pose_path = os.path.join(args.cen_images, pose)
if not os.path.exists( pose_path ):
    os.makedirs( pose_path )

#faceDet network is loaded after faceReg as gpu usage % cannot be specified for faceDet
faceDet = FaceDet(threshold=0.1)

shower = Shower()
show_title = 'Quell the faces!'
shower.start(show_title)
for img_path in img_paths:
    img = cv2.imread(img_path)

    bbs = faceDet.detect_bb(img)
    for bb in bbs:
        noise_face(img, bb, thresh = 0.7)
    shower.show(show_title, img)
    key = cv2.waitKey(0) 
    if key == ord('q'):
        break
    elif key == ord('d'):
        img = noiser(img, shower, radius=5)
    
    dst_fp = os.path.join( pose_path , os.path.basename(img_path) )
    cv2.imwrite(dst_fp, img)

cv2.destroyAllWindows()
