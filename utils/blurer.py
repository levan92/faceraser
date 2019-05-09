import cv2
import numpy as np

kernel = np.ones((5,5),np.float32)/25

def blur_crop(frame, y0,y1,x0,x1):
	crop = frame[y0:(y1+1), x0:(x1+1)]
	dst = cv2.filter2D(crop,-1,kernel)
	return dst