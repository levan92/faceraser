#!/usr/bin/python3
import os 
import numpy as np
import cv2
import time
import tensorflow as tf 
CURR_DIR = os.path.dirname(os.path.abspath(__file__))

def read_label_map(csv_path):
    label_map = {}
    with open(csv_path,'r') as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            idx, cls_label = line.split(',')
            label_map[cls_label.strip()] = int(idx)
    assert 'face' in label_map,'face not in label_map_csv!'
    return label_map


class Mobnet_TF(object):
    def __init__(self, fd_pb, label_csv, gpu_usage=None, threshold=0.5):
        """Tensorflow detector
        """
        self.frozen_graph = fd_pb
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            self.graphDef = tf.GraphDef()
            with tf.gfile.GFile(fd_pb, 'rb') as fid:
                serialized_graph = fid.read()
                self.graphDef.ParseFromString(serialized_graph)
                tf.import_graph_def(self.graphDef, name='')

            config = tf.ConfigProto()
            if gpu_usage is None:
                config.gpu_options.allow_growth = True
                print('Initalising Mobilenet SSD FD at unlimited gpu usage (allow_growth)..')
            else:
                config.gpu_options.per_process_gpu_memory_fraction = gpu_usage
                print('Initalising Mobilenet SSD FD at {} gpu usage..'.format(gpu_usage))
            self.sess = tf.Session(graph=self.detection_graph, config=config)
            # self.windowNotSet = True
        
        self.label_map = read_label_map(label_csv)
        self.threshold = threshold

    def _post_process(self, boxes, scores, classes, im_size):
        bbs = []
        im_height, im_width = im_size
        for i, score in enumerate(scores):
            if score > self.threshold and int(classes[i]) == self.label_map['face']:
                box = boxes[i]
                t = box[0] * im_height
                l = box[1] * im_width
                b = box[2] * im_height
                r = box[3] * im_width
                w = r - l
                h = b - t
                bb = {'rect':{'t': t, 
                              'l': l,
                              'r': r,
                              'b': b, 
                              'w': w, 
                              'h': h },
                      'confidence': score}
                bbs.append(bb)
        return bbs

    def __call__(self, image):
        """
        image: bgr image
        returns: bbs, list of {'rect':{'t': boxes[0], 
                                       'l': boxes[1],
                                       'r': boxes[3],
                                       'b': boxes[2], 
                                       'w': boxes[3] - boxes[1], 
                                       'h': boxes[2] - boxes[0] },
                                'confidence': score}

        """
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_size = image_np.shape[:2]
        # image_np = image
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        # start_time = time.time()
        (boxes, scores, classes, _) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        # elapsed_time = time.time() - start_time
        # print('inference time cost: {}'.format(elapsed_time))

        return self._post_process(np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes), im_size)

class Mobnet_FD:
    def __init__(self, fd_pb=None, label_csv=None, gpu_usage=None, max_n =None, **kwargs):
        if fd_pb is None:
            fd_pb = os.path.join(CURR_DIR, "mobnet_frozen_graph.pb")
            assert os.path.exists(fd_pb),'{} does not exist'.format(fd_pb)

        if label_csv is None:
            label_csv = os.path.join(CURR_DIR, "mobnet_label_map.csv")
            assert os.path.exists(label_csv),'{} does not exists'.format(label_csv)

        self.detector = Mobnet_TF(fd_pb, label_csv, gpu_usage=gpu_usage)
        self.max_n = max_n
        # warm up
        ret = self.detector(np.zeros((10,10,3), dtype=np.uint8))
        self.i = 0
        print("FACE DETECTION: Mobilenet SSD FD object initalised")

    def detect(self, img3chnl):
        '''
        returns: bbs, list of {'rect':{'t': boxes[0], 
                                       'l': boxes[1],
                                       'r': boxes[3],
                                       'b': boxes[2], 
                                       'w': boxes[3] - boxes[1], 
                                       'h': boxes[2] - boxes[0] },
                                'confidence': score}
        '''
        assert img3chnl is not None,'FD didnt rcv img'
        
        try:
            return self.detector(img3chnl)
        except Exception as e:
            print("WARNING from FD detect: {}".format(e))
            return []

    def detect_bb(self, img3chnl):
        '''
        pass through fn for enrol2phone.py
        '''
        return self.detect(img3chnl)        
    
    def _detect_batch(self, img3chnls):
        '''
        :return: array of bbs.
        '''
        assert img3chnls is not None,'FD didnt rcv img'
        all_bbs = []
        for img3chnl in img3chnls:
            try:
                if img3chnl is None or img3chnl.dtype != np.uint8:
                    all_bbs.append([])
                else:
                    all_bbs.append(self.detector(img3chnl))
            except Exception as e:
                print("WARNING from FD detect_batch: {}".format(e))
                all_bbs.append([])
        return all_bbs

if __name__ == '__main__':
    fd = Mobnet_FD()
    img = cv2.imread('/home/dh/Workspace/FR/Data/pics/IMG_4670.jpeg')
    fd.detect(img)
