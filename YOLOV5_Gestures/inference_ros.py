import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from depthai_ros_msgs.msg import SpatialDetectionArray, SpatialDetection
from vision_msgs.msg import ObjectHypothesis
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
class OakDDetection:
    def __init__(self, center, size, conf, xyz):
        self.center = center
        self.size_x, self.size_y = size
        self.confidence =  conf
        self.x, self.y, self.z = xyz
        ### bounding box
        self.xmin, self.ymin, self.xmax, self.ymax = calculateRectEdges(center, size)
    def __repr__(self) -> str:
        return "%s (%s, %s) : %s" % (self.center, self.size_x, self.size_y, self.confidence)
class OakDDetections:
    def __init__(self, detections):
        self.detections = detections
    def valid(self):
        return len(self.detections) > 0
    def scale(self, rgb, depth):
        height, width, dim = rgb.shape
        detection : OakDDetection
        for i, detection in enumerate(self.detections):
            pass
    def matchPointToClosestDetection(self, point):
        rects = [(((d.xmin, d.ymin, d.xmax, d.ymax), d)) for d in self.detections]
        depthrects = [(((d.xmin, d.ymin, d.xmax, d.ymax), d)) for d in self.detections]
        index, closest_dist, detectedObject = find_closest_rectangle(point, rects)
        resultRgbRect = None
        resultDepthRect = None
        if index != None:
            resultRgbRect = rects[index][0]
            resultDepthRect = depthrects[index][0]
        return index, closest_dist, detectedObject, resultRgbRect, resultDepthRect
    def __repr__(self) -> str:
        return "%s" % [str(x) for x in self.detections] 
class Match:
    def __init__(self, cls, classname, conf, rectRgb, rectDepth, xyz):
        self.cls = cls
        self.classname = classname
        self.conf = conf
        self.rectRgb = rectRgb
        self.rectDepth = rectDepth
        self.xyz = xyz
    def __repr__(self) -> str:
        return str(self.__dict__)

#############################################################################################
class InferenceThread(threading.Thread):
    def __init__(self, model : ModelWrapper):
        super().__init__()
        self.model = model
        self.daemon = True
        self.output = None
        self.mutex = Lock()
        self.running = True
        self.clearInputs()
    def run(self):
        while self.running:
            with self.mutex:
                img = self.input_image_buf_1
            if img.any():
                ### do inference stuff
                start = time.time()
                with torch.no_grad(): ### no bp & gradients compute during inference
                    predictions = self.model.run(self.input_image_buf_1)
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
            self.input_image_buf_1 = img.copy()
    def getOutput(self):
        return self.output
    def clearInputs(self):
        self.input_image_buf_1 = np.empty((0,0))
#############################################################################################
class Annotator:
    font = cv2.FONT_HERSHEY_TRIPLEX
    size = 0.75
    # textcolor = (0,255,0)
    @staticmethod
    def drawMatches(rgb, depth, matches):
        if not matches:
            return rgb, depth
        rgb_img = rgb.copy()
        depth_img = depth.copy()
        m : Match
        for m in matches:
            label = f'{m.classname} {m.conf:.2f}'
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
            cv2.putText(rgb_img, f"X: {x:.3f} m", (centerx + 10, centery + 50), Annotator.font, Annotator.size, classColor)
            cv2.putText(rgb_img, f"Y: {y:.3f} m", (centerx + 10, centery + 65), Annotator.font, Annotator.size, classColor)
            cv2.putText(rgb_img, f"Z: {z:.3f} m", (centerx + 10, centery + 80), Annotator.font, Annotator.size, classColor)
            #### annotate depth image rect
            x1, y1, x2, y2 = m.rectDepth
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(depth_img, p1, p2, (255,255,255), 2)
        return rgb_img, depth_img
    @staticmethod
    def drawPrediction(rgb, label, box, color=(128, 128, 128), txt_color=(255, 255, 255), lw=1):
        rgb_img = rgb.copy()
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        cv2.rectangle(rgb_img, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.circle(rgb_img, center, 5, (0, 0, 255), -1)
            cv2.rectangle(rgb_img, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(rgb_img, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
        return rgb_img
    @staticmethod
    def drawDetection(rgb, depth, detection : OakDDetection):
        rgb_img = rgb.copy()
        depth_img = depth.copy()
        # Draw a rectangle on the RGB depth image
        p1 = (int(detection.xmin), int(detection.ymin))
        p2 = (int(detection.xmax), int(detection.ymax))
        color = (255, 255, 255)  # Color of the rectangle (Green in BGR format)
        thickness = 2  # Thickness of the rectangle border
        # cv2.rectangle(rgb_img, p1, p2, color, thickness, cv2.LINE_AA)
        cv2.rectangle(depth_img, p1, p2, color, thickness, cv2.LINE_AA)
        return rgb_img, depth_img
#############################################################################################                  
def centerFromRect(rect):
    p1, p2 = (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3]))
    center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    return center
def calculateRectEdges(center, size):
    center_x, center_y = center
    width, height = size
    xmin = center_x - width / 2
    ymin = center_y - height / 2
    xmax = center_x + width / 2
    ymax = center_y + height / 2
    return xmin, ymin, xmax, ymax

class InferenceRosNode(Node):
    def __init__(self):
        super().__init__('inference_ros_node')
        #################################################
        ### Declare and get parameters
        # self.declare_parameter('model_config', 'config.yml')
        self.declare_parameter('visualize', False)
        self.declare_parameter('model_checkpoint', '')
        # self.declare_parameter('oakd_nn_blob', os.path.abspath("models/yolo-v4-tiny-tf_openvino_2021.4_6shave.blob"))
        self.declare_parameter('rgb_topic', '/color/image')
        self.declare_parameter('depth_topic', '/stereo/depth')
        self.declare_parameter('nn_topic', '/color/yolov4_Spatial_detections')
        self.declare_parameter('pixel_match_threshold', 200)   
        self.declare_parameter('prediction_score_treshold', 0.2)   
        self.declare_parameter('detection_score_treshold', 0.2)   
        # configPath = self.get_parameter('model_config').get_parameter_value().string_value
        self.visualize = self.get_parameter('visualize').get_parameter_value().bool_value
        modelCheckpoint = self.get_parameter('model_checkpoint').get_parameter_value().string_value
        rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        nn_topic = self.get_parameter('nn_topic').get_parameter_value().string_value
        self.pixel_match_threshold = self.get_parameter('pixel_match_threshold').get_parameter_value().integer_value
        self.prediction_score_treshold = self.get_parameter('prediction_score_treshold').get_parameter_value().double_value
        self.detection_score_treshold = self.get_parameter('detection_score_treshold').get_parameter_value().double_value
        #################################################
        ### Create publishers/subscribers for the two image topics
        if self.visualize:
            self.rgb_pub = self.create_publisher(Image, '~/rgb/labeled', 1)
            self.depth_pub = self.create_publisher(Image, '~/depth/colored', 1)
        self.rgb_sub = Subscriber(self, Image, rgb_topic)
        self.depth_sub = Subscriber(self, Image, depth_topic)
        self.nn_sub = Subscriber(self, SpatialDetectionArray, nn_topic)
        # self.nn_sub = self.create_subscription(TrackDetection2DArray, nn_topic, self.nn_callback, 1)

        ### Create the ApproximateTimeSynchronizer
        self.ts = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.nn_sub],
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
        self.inference.start()
        #################################################
        ### misc
        self.bridge = cv_bridge.CvBridge()
        self.rgb_buf = []
        self.depth_buf = []        
    def image_callback(self, rgb_msg, depth_msg, nn_msg):
        self.get_logger().debug('Synchronized images received')
        rgb0 = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        depth0 = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        ### prepare images 
        rgb_rescaled, rgb_input = self.model.transformInputRgb(rgb0)
        depth = self.model.transformInputDepth(depth0) ## will be colored
        ###
        self.inference.setInputs(rgb_input)
        ### grab last available output from model (could be old)
        predictions = self.inference.getOutput()
        if predictions == None:
            return
        ### calculate proper rectangles inside rgb and depth from neural network output of oak
        detections = self.parseNNMsg(nn_msg)
        # detections.scale(rgb_rescaled, depth)
        ### debug draw detections
        if self.visualize:
            for detec in detections.detections:
                rgb_rescaled, depth = Annotator.drawDetection(rgb_rescaled, depth, detec)
        ### for every prediction, match the closest oak detection
        matches = []
        for rect, conf, cls in predictions:
            classname = self.model.lookupName(cls)
            center = centerFromRect(rect)
            if self.visualize:
                rgb_rescaled = Annotator.drawPrediction(rgb_rescaled, classname, rect)
            index, closest_dist, detectedObject, rgbRect, depthRect = detections.matchPointToClosestDetection(center)
            if closest_dist < self.pixel_match_threshold and conf > self.prediction_score_treshold:
                x,y,z = (detectedObject.x, detectedObject.y, detectedObject.z)
                m = Match(cls, classname, conf, rgbRect, depthRect, (x,y,z))
                matches.append(m)
        ### annotate and visualize 
        if self.visualize:
            rgb_rescaled, depth = Annotator.drawMatches(rgb_rescaled, depth, matches)
        ### publish output images containing last available model output
        rgb_img_msg = self.bridge.cv2_to_imgmsg(rgb_rescaled, "bgr8")
        depth_img_msg = self.bridge.cv2_to_imgmsg(depth, "bgr8")
        self.rgb_pub.publish(rgb_img_msg)
        self.depth_pub.publish(depth_img_msg)
    def parseNNMsg(self, msg : SpatialDetectionArray) -> OakDDetections:
        ### parse detections
        extracted = []
        detection:SpatialDetection
        for detection in msg.detections:
            ### only take tracked objects instead of new or lost ones
            # if int(detection.tracking_status) != 1:
            #     continue
            ### go throug all possible classes and scores
            result : ObjectHypothesis
            personResult = None
            for result in detection.results:
                if int(result.class_id) == 0 and result.score > self.detection_score_treshold:
                    personResult = result
                    break
            if personResult:
                extracted.append(OakDDetection((detection.bbox.center.position.x, detection.bbox.center.position.y),
                                               (detection.bbox.size_x, detection.bbox.size_y), 
                                               personResult.score, 
                                               (detection.position.x, detection.position.y, detection.position.z)))
        return OakDDetections(extracted)
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
