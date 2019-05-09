import numpy as np

def in_ellipse(x,y,cx,cy,rx,ry):
    eqn = (x-cx)**2 / rx**2 + (y-cy)**2 / ry**2

    if eqn <= 1:
        return True
    else:
        return False

def noise_face(frame, face_bb, thresh = 0.7):
    if face_bb['confidence'] < 0.7:
        return None

    rx = int(face_bb['rect']['w']/2)
    ry = int(face_bb['rect']['h']/2)
    cx = int(face_bb['rect']['l']) + rx
    cy = int(face_bb['rect']['t']) + ry

    for x in range(int(face_bb['rect']['l']), int(face_bb['rect']['r'])+1):
        for y in range(int(face_bb['rect']['t']), int(face_bb['rect']['b'])+1):
            if not in_ellipse(x, y, cx, cy, rx, ry):
                continue
            frame[y,x] = np.random.uniform(0, 255, 3)

