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
parser.add_argument('dir_images', help='Filepath to folder of images to censor', type=str)
parser.add_argument('folder_name', help='Folder name of images to censor', type=str)
args = parser.parse_args()

if args.dir_images[-1]!='/':
    args.dir_images = args.dir_images+'/'

folder_path = args.dir_images+args.folder_name
assert os.path.isdir(folder_path),'dir given is not a directory!'
img_paths = []
for images in os.listdir(folder_path):
    img_paths.append(os.path.join(folder_path, images))
total_num = len(img_paths)
print('{} images in all'.format(total_num))

#faceDet network is loaded after faceReg as gpu usage % cannot be specified for faceDet
faceDet = FaceDet(threshold=0.1)

shower = Shower()
show_title = 'Quell the faces!'
display_title=show_title
shower.start(display_title)
frame_count = 0


if not os.path.exists(args.folder_name):
    os.makedirs(args.folder_name)

while frame_count < len(img_paths):
    img_path = img_paths[frame_count]
    img = cv2.imread(img_path)

    bbs = faceDet.detect_bb(img)
    for bb in bbs:
        noise_face(img, bb, thresh = 0.7)
    cv2.destroyWindow(display_title)
    display_title=show_title+'\t'+args.folder_name+'\t'+str(frame_count+1)+' of '+str(total_num)
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
    cv2.imwrite('{}/{}.png'.format(args.folder_name,frame_count), img)

print('\nuwu... bye bye')
cv2.destroyAllWindows()
