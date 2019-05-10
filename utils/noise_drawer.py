import cv2
import copy 
import numpy as np
import math

from .blurer import blur_crop

mouse_pt = None
click = False

def mouse_events_handler(event, x, y, flags, param):
    global mouse_pt, click, edge_x, edge_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_pt = (x,y)
    if event == cv2.EVENT_LBUTTONDOWN:
        click = True
    if event == cv2.EVENT_LBUTTONUP:
        click = False


def draw_crosshair(frameDC, radius=10):
    global mouse_pt
    if mouse_pt:
        colour = (0,255,0)
        THICC = 2
        # h, w = frameDC.shape[:2]
        frameSHOW = copy.deepcopy(frameDC)
        cv2.circle(frameSHOW, mouse_pt, radius, colour, THICC)
        return frameSHOW
    else:
        return frameDC

def get_points_in_circle(frameDC, frameSHOW, center, radius, frame_size, mode='blur', blurreps=10):
    # y = np.arange(frame_size[0])
    # x = np.arange(frame_size[1])
    # mask = (x[np.newaxis,:]-center[0])**2 + (y[:,np.newaxis]-center[1])**2 < radius**2
    start_x = max(0, center[0] - radius)
    end_x = min(frame_size[1]-1, center[0] + radius)
    start_y = max(0, center[1] - radius)
    end_y = min(frame_size[0]-1, center[1] + radius)

    if mode != 'blur':
        for x in range(start_x, end_x+1):
            for y in range(start_y, end_y+1):
                dist = math.sqrt( (x - center[0])**2 + (y - center[1])**2 )
                if dist > radius:
                    continue
                frameDC[y,x] = np.random.uniform(0, 255, 3)
                frameSHOW[y,x] = np.random.uniform(0, 255, 3)
    else:
        for _ in range(blurreps):
            cropBlurDC = blur_crop(frameDC,start_y,end_y,start_x,end_x)
            cropBlurSHOW = blur_crop(frameSHOW,start_y,end_y,start_x,end_x)

            for x in range(start_x, end_x+1):
                for y in range(start_y, end_y+1):
                    dist = math.sqrt( (x - center[0])**2 + (y - center[1])**2 )
                    if dist > radius:
                        continue
                    frameDC[y,x] = cropBlurDC[y-start_y,x-start_x]
                    frameSHOW[y,x] = cropBlurSHOW[y-start_y,x-start_x]

def process(frameDC, frameSHOW, radius, frame_size):
    global start_click, click
    if click and mouse_pt:
        get_points_in_circle(frameDC, frameSHOW, mouse_pt, radius, frame_size)
        return frameDC, frameSHOW
    else:
        click = None
    return frameDC, frameSHOW

def noiser(frame, shower, radius=10):
    global mouse_pt, click, start_click, frame_size
    mouse_pt = None
    click = None
    start_click = None
    window_name = 'Draw NOISE'
    conf = None
    frame_size = frame.shape[:2]
    shower.start(window_name)
    cv2.setMouseCallback(window_name, mouse_events_handler)
    frameDC = copy.deepcopy(frame)
    while True:
        frameSHOW = draw_crosshair(frameDC, radius)
        frameDC, frameSHOW = process(frameDC, frameSHOW, radius, frame_size)
        # show(window_name, frameDC)
        shower.show(window_name, frameSHOW, wait=-1)
        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key==32: # Enter or Spacebar
            res = frameDC
            break
        elif key == ord('c'):
            res = None
            break
    cv2.destroyWindow(window_name)
    return res

if __name__ == '__main__':
    import time
    from shower import Shower
    import sys

    if len(sys.argv) < 2:
        print('defaulted to bb drawer mode')
        mode = 1
    else:
        if sys.argv[1] == 'd':
            mode = 1        
            print('chosen bb drawer mode')
        elif sys.argv[1] == 'e':
            mode = 2
            print('chosen bb editor mode')
        else:
            mode = 1
            print('defaulted to bb drawer mode')

    img_path =  '/home/angeugn/Pictures/yolo-object-detection-dog.jpg'
    # img_path =  '/home/levan/Pictures/ozil.jpg'
    shower = Shower()
    frame = cv2.imread(img_path)

    final_frame = noiser(frame, shower=shower)

    cv2.imwrite('test.png', final_frame)

    # print('Final bbs: {}'.format(bbs))
    # time.sleep(1)
    # bbs = bbs_editor(frame, bbs = bbs)
    # print('Final final: {}'.format(bbs))
