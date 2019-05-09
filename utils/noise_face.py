import numpy as np

from .blurer import blur_crop

def in_ellipse(x,y,cx,cy,rx,ry):
    eqn = (x-cx)**2 / rx**2 + (y-cy)**2 / ry**2

    if eqn <= 1:
        return True
    else:
        return False

def noise_face(frame, face_bb, thresh = 0.7, mode='blur',blurreps=20):
    if face_bb['confidence'] < 0.7:
        return None

    rx = int(face_bb['rect']['w']/2)
    ry = int(face_bb['rect']['h']/2)
    cx = int(face_bb['rect']['l']) + rx
    cy = int(face_bb['rect']['t']) + ry

    start_y = int(face_bb['rect']['t'])
    end_y = int(face_bb['rect']['b'])+1
    start_x = int(face_bb['rect']['l'])
    end_x = int(face_bb['rect']['r'])+1
    if mode != 'blur':
        for x in range(start_x, end_x):
            for y in range(start_y, end_y):
                if not in_ellipse(x, y, cx, cy, rx, ry):
                    continue
                frame[y,x] = np.random.uniform(0, 255, 3)
    else:
        for _ in range(blurreps):
            cropBlur = blur_crop(frame,start_y,end_y,start_x,end_x)
            for x in range(start_x, end_x):
                for y in range(start_y, end_y):
                    if not in_ellipse(x, y, cx, cy, rx, ry):
                        continue
                    frame[y,x] = cropBlur[y-start_y,x-start_x]
