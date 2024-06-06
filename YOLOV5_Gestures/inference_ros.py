import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, Subscriber

import cv2
import cv_bridge

import matplotlib.pyplot as plt 
import numpy as np   
import os
import torch
import threading
import time
import copy
from threading import Lock


### ros packaged imports
from YOLOV5_Gestures.utils.match import find_closest_rectangle
from YOLOV5_Gestures.utils.modelWrapper import ModelWrapper
from YOLOV5_Gestures.utils.plots import colors

#############################################################################################
class OakDDetections:
    def __init__(self, detections):
        self.detections = detections
        self.rects = []
        self.depthRects = []
    def process(self, rgb, depth):
        height, width, dim = rgb.shape
        for detection in self.detections:
            if detection.label != 0: #person
                continue
            if detection.confidence < 10:
                continue
            roiData = detection.boundingBoxMapping
            roi = roiData.roi
            roi = roi.denormalize(depth.shape[1], depth.shape[0])
            topLeft = roi.topLeft()
            bottomRight = roi.bottomRight()
            #depth
            xmin = int(topLeft.x)
            ymin = int(topLeft.y)
            xmax = int(bottomRight.x)
            ymax = int(bottomRight.y)
            self.depthRects.append((xmin, ymin, xmax, ymax))
            # rgb
            # Denormalize bounding box
            x1 = int(detection.xmin * width)
            x2 = int(detection.xmax * width)
            y1 = int(detection.ymin * height)
            y2 = int(detection.ymax * height)
            self.rects.append(((x1, y1, x2, y2), detection))
    def matchPointToClosestDetection(self, point):
        index, closest_dist, detectedObject = find_closest_rectangle(point, self.rects)
        return index, closest_dist, detectedObject, self.rects[index], self.depthRects[index]
class Match:
    def __init__(self, cls, classname, conf, rectRgb, rectDepth, xyz):
        self.cls = cls
        self.classname = classname
        self.conf = conf
        self.rectRgb = rectRgb
        self.rectDepth = rectDepth
        self.xyz = xyz

#############################################################################################
class InferenceThread(threading.Thread):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.daemon = True
        self.output = None
        self.mutex = Lock()
        self.clearInputs()
    def run(self):
        while self.running:
            with self.mutex:
                img = self.input_image_buf_1
            if img:
                ### do inference stuff
                start = time.time()
                with torch.no_grad(): ### no bp & gradients compute during inference
                    predictions, inputImg = self.model.run(self.input_image_buf_1)
                    self.output = predictions
                dt = time.time() - start
                print("dt: %s" % (dt))
            else:
                ### safety delay for thread to not run amok
                time.sleep(0.001)
                
    def stop(self):
        self.running = False
    def setInputs(self, img):
        with self.mutex:
            self.input_image_buf_1 = img.copy
    def getOutput(self):
        return self.output
    def clearInputs(self):
        self.input_image_buf_1 = None
#############################################################################################
class Annotator:
    font = cv2.FONT_HERSHEY_TRIPLEX
    size = 0.75
    # textcolor = (0,255,0)
    @staticmethod
    def draw(rgb, depth, matches):
        if not matches:
            return
        rgb_img = rgb.copy()
        depth_img = depth.copy()
        for m in matches:
            label = f'{m.classname} {m.conf:.2f}')
            classColor = colors(m.cls, True)
            x,y,z = m.xyz
            ### annotate rgb image rect and depth info
            center = centerFromRect(m.rectRgb)
            centerx = center[0]
            centery = center[1]
            x1, y1, x2, y2 = m.rectRgb
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(rgb_img, p1, p2, classColor, thickness=1, lineType=cv2.LINE_AA)
            cv2.putText(rgb_img, label, (p1[0], p1[1] - 2), 0, 0.5, classColor, thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(rgb_img, "{:.2f}".format(m.conf), (centerx + 10, centery + 35), Annotator.font, Annotator.size, classColor)
            cv2.putText(rgb_img, f"X: {int(x)} mm", (centerx + 10, centery + 50), Annotator.font, Annotator.size, classColor)
            cv2.putText(rgb_img, f"Y: {int(y)} mm", (centerx + 10, centery + 65), Annotator.font, Annotator.size, classColor)
            cv2.putText(rgb_img, f"Z: {int(z)} mm", (centerx + 10, centery + 80), Annotator.font, Annotator.size, classColor)
            #### annotate depth image rect
            xmin, ymin, xmax, ymax = m.rectDepth
            cv2.rectangle(depth_img, (xmin, ymin), (xmax, ymax), (255,255,255), 2)
        return rgb_img, depth_img
#############################################################################################                  
def centerFromRect(rect):
    p1, p2 = (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3]))
    center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    return center

class InferenceRosNode(Node):
    def __init__(self):
        super().__init__('inference_ros_node')
        #################################################
        ### Declare and get parameters
        # self.declare_parameter('model_config', 'config.yml')
        self.declare_parameter('visualize', False)
        self.declare_parameter('model_checkpoint', '')
        self.declare_parameter('oakd_nn_blob', os.path.abspath("models/yolo-v4-tiny-tf_openvino_2021.4_6shave.blob"))
        self.declare_parameter('rgb_topic', '/rgb/image_raw')
        self.declare_parameter('depth_topic', '/depth/image_raw')   
        self.declare_parameter('pixel_match_threshold', 200)   
        # configPath = self.get_parameter('model_config').get_parameter_value().string_value
        self.visualize = self.get_parameter('visualize').get_parameter_value().bool_value
        modelCheckpoint = self.get_parameter('model_checkpoint').get_parameter_value().string_value
        rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.pixel_match_threshold = self.get_parameter('pixel_match_threshold').get_parameter_value().double_value
        #################################################
        ### Create publishers/subscribers for the two image topics
        self.rgb_pub = self.create_publisher(Image, '~/rgb/labeled', 1)
        self.depth_pub = self.create_publisher(Image, '~/depth/colored', 1)
        self.rgb_sub = Subscriber(self, Image, rgb_topic)
        self.depth_sub = Subscriber(self, Image, depth_topic)
        ### Create the ApproximateTimeSynchronizer
        self.ts = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )
        ### Register the callback to be called when both images are received
        self.ts.registerCallback(self.image_callback)
        #################################################
        ### model stuff
        self.model = ModelWrapper(modelCheckpoint)
        #################################################
        ### inference thread
        self.inference = InferenceThread(self.model)
        #################################################
        ### misc
        self.bridge = cv_bridge.CvBridge()
        self.rgb_buf = []
        self.depth_buf = []
        
    def image_callback(self, rgb0, depth0):
        self.get_logger().debug('Synchronized images received')
        ### prepare images 
        rgb = self.model.transformInputRgb(rgb0)
        depth = self.model.transformInputDepth(depth0) ## will be colored
        ###
        self.inference.setInputs(rgb)
        ### grab last available output from model (could be old)
        predictions = self.inference.getOutput()
        ### calculate proper rectangles inside rgb and depth from neural network output of oak
        self.detections.process(rgb, depth)
        ### for every prediction, match the closest oak detection
        matches = []
        for rect, conf, cls in predictions:
            classname = self.model.lookupName(cls)
            center = centerFromRect(rect)
            index, closest_dist, detectedObject, rgbRect, depthRect = self.detections.matchPointToClosestDetection(center)
            if closest_dist < self.pixel_match_threshold and conf > 15:
                x,y,z = (int(detectedObject.spatialCoordinates.x), int(detectedObject.spatialCoordinates.y), int(detectedObject.spatialCoordinates.z))
                matches.append(Match(cls, classname, conf, rgbRect, depthRect, (x,y,z)))
        ### annotate and visualize 
        if self.visualize:
            rgb, depth = Annotator.draw(rgb, depth, matches)
        ### publish output images containing last available model output
        rgb_img_msg = self.bridge.cv2_to_imgmsg(rgb, "bgr8")
        depth_img_msg = self.bridge.cv2_to_imgmsg(depth, "bgr8")
        self.rgb_pub.publish(rgb_img_msg)
        self.depth_pub.publish(depth_img_msg)
    def nn_callback(self, detections):
        self.detections = OakDDetections(detections)


    def destroy_node(self):
        self.inference.stop()
        super().destroy_node()
#############################################################################################
def main(args=None):
    rclpy.init(args=args)
    node = InferenceRosNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
