import cv2
import numpy as np

k=10
kernel = np.ones((k,k),np.float32)/(k**2)

def blur_crop(frame, y0,y1,x0,x1):
	if (y1 - y0) <= 0:
		return None
	elif (x1 - x0) <= 0:
		return None
	crop = frame[y0:(y1+1), x0:(x1+1)]
	dst = cv2.filter2D(crop,-1,kernel)
	return dst