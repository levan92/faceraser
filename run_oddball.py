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
parser.add_argument('cen_images', help='Directory for directory of censored images (AKA Parent Dir of Pose)', type=str)
parser.add_argument('dir_images', help='Filepath to directory of images to censor (AKA the actual pose Dir)', type=str)
args = parser.parse_args()

assert os.path.isdir(args.dir_images),'dir given is not a directory!'

pose = os.path.basename(args.dir_images)
if not os.path.exists( args.cen_images ):
    os.makedirs( args.cen_images )
pose_path = os.path.join(args.cen_images, pose)
if not os.path.exists( pose_path ):
    os.makedirs( pose_path )

existing_censored = list(os.listdir(pose_path))

img_paths = []
for images in [im for im in os.listdir(args.dir_images) if im not in existing_censored]:
    img_paths.append(os.path.join(args.dir_images, images))
total_num = len(img_paths)
print('{} images in all'.format(total_num))

if total_num <= 0:
    print('No images to censor. Exiting.')
    exit()

#faceDet network is loaded after faceReg as gpu usage % cannot be specified for faceDet
faceDet = FaceDet(threshold=0.1)

shower = Shower()
show_title = 'Quell the faces!'
display_title=show_title
shower.start(display_title)
frame_count = 0

while frame_count < len(img_paths):
    img_path = img_paths[frame_count]
    img = cv2.imread(img_path)

    bbs = faceDet.detect_bb(img)
    for bb in bbs:
        noise_face(img, bb, thresh = 0.7)
    cv2.destroyWindow(display_title)
    display_title=show_title+'\t'+pose+'\t'+str(frame_count+1)+' of '+str(total_num)
    shower.start(display_title)
    shower.show(display_title, img)
    key = cv2.waitKey(0) 
    if key == ord('q'):
        print('Quitting')
        break
    elif key == ord('d'):
        img = noiser(img, shower, radius=5)
    elif key == ord('b'): #go back
        frame_count=max(frame_count-1,0)
        continue

    frame_count+=1
    dst_fp = os.path.join( pose_path , os.path.basename(img_path) )
    cv2.imwrite(dst_fp, img)

print('\nuwu... bye bye')
cv2.destroyAllWindows()
