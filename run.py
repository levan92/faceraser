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
from faceDet.mobnet import Mobnet_FD as FaceDet


parser = argparse.ArgumentParser()
parser.add_argument('dir_images', help='Path to directory of images to censor', type=str)
parser.add_argument('out', help='Path to folder of outputed censored images', type=str)
args = parser.parse_args()

assert os.path.isdir(args.dir_images),'dir given is not a directory!'
img_paths = [ os.path.join(args.dir_images, n) for n in os.listdir(args.dir_images) ]
total_num = len(img_paths)
print('{} images in all'.format(total_num))

if not os.path.exists(args.out):
    os.makedirs(args.out)

#faceDet network is loaded after faceReg as gpu usage % cannot be specified for faceDet
faceDet = FaceDet(threshold=0.1)

shower = Shower()
show_title = 'Quell the faces!'
display_title=show_title
shower.start(display_title)
frame_count = 0

while frame_count < len(img_paths):
    img_path = img_paths[frame_count]
    img_name = os.path.basename(img_path)
    img = cv2.imread(img_path)

    bbs = faceDet.detect_bb(img)
    for bb in bbs:
        noise_face(img, bb, thresh = 0.7)
    cv2.destroyWindow(display_title)
    display_title=show_title+'\t'+args.dir_images+'\t'+str(frame_count+1)+' of '+str(total_num)
    shower.start(display_title)
    shower.show(display_title, img)
    key = cv2.waitKey(0) 
    if key == ord('q'):
        print('Quitting')
        break
    elif key == ord('d'):
        img = noiser(img, shower, radius=10)
    elif key == ord('b'): #go back
        frame_count=max(frame_count-1,0)
        continue
    
    frame_count += 1
    out_path = os.path.join(args.out, img_name)
    cv2.imwrite(out_path, img)

print('\nuwu... bye bye')
cv2.destroyAllWindows()
