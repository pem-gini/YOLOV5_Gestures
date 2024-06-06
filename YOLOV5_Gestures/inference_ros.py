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
from YOLOV5_Gestures.utils.modelWrapper import ModelWrapper

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
    @staticmethod
    def draw(rgb, depth, predicitons, oakddetections=None):
        rgb_img = rgb.copy()
        depth_img = depth.copy()
        return rgb_img, depth_img
        
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
        # configPath = self.get_parameter('model_config').get_parameter_value().string_value
        self.visualize = self.get_parameter('visualize').get_parameter_value().bool_value
        modelCheckpoint = self.get_parameter('model_checkpoint').get_parameter_value().string_value
        rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
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
        ### annotate and visualize 
        if self.visualize:
            Annotator.draw(rgb, depth, predictions)
        ### publish output images containing last available model output
        rgb_img_msg = self.bridge.cv2_to_imgmsg(rgb, "bgr8")
        depth_img_msg = self.bridge.cv2_to_imgmsg(depth, "bgr8")
        self.rgb_pub.publish(rgb_img_msg)
        self.depth_pub.publish(depth_img_msg)

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
