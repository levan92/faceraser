# facemoji
simple tool to erase faces

## Usage
Quick start:
`python3 run.py [path to directory of images]`

1. Goes through each image in your directory
2. Face detection is done and face ellipse is censored with random noise
3. Press `d` to draw noise manually 
4. Press `q` to quit

## My Stack
- CUDA 9.0
- cudnn 7.4.2
- cv2 v3.4.0
- tensorflow-gpu==1.13.1
<!-- - keras==2.1.3 -->
<!-- - dlib==19.9.99  -->

## Acknowledgments
- [dlib](http://dlib.net/)

## TODO
