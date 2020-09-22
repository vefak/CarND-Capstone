from styx_msgs.msg import TrafficLight

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError #Ros lib for OpenCV

import cv2
import numpy as np
import tensorflow as tf
import datetime

class TLClassifier(object):
    def __init__(self, is_simulation):
      

        if is_simulation:
            self.model_type = 'light_classification/simulation'
        else:
            self.model_type = 'light_classification/site'

        self.path_of_pb = self.model_type + '/inference_graph.pb'

        #load classifier
        # Load a Tensorflow model into memory.
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_of_pb, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.detection_graph.get_tensor_by_name(
                'num_detections:0')
 
        self.session = tf.Session(graph=self.detection_graph)
        self.threshold = 0.5
    

    def get_classification(self, image):
                

        with self.detection_graph.as_default():
            image_expand = np.expand_dims(image, axis=0)
            start_classification_t = datetime.datetime.now()

            (boxes, scores, classes, num_detections) = self.session.run(
                [self.boxes, self.scores, self.classes, self.num_detections],
                feed_dict={self.image_tensor: image_expand})

            end_classification_t = datetime.datetime.now()
            elapsed_time = end_classification_t - start_classification_t

   

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)



        if scores[0] > self.threshold:
            if classes[0] == 1:
                print('Current Traffic Light is: GREEN')
                print('Keep going')
                return TrafficLight.GREEN
            elif classes[0] == 2:
                print('Current Traffic Light is: RED')
                print('Stopping...')
                return TrafficLight.RED
            elif classes[0] == 3:
                print('Current Traffic Light is: YELLOW')
                return TrafficLight.YELLOW

        return TrafficLight.UNKNOWN