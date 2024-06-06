
import os
import sys
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn

from YOLOV5_Gestures.models.common import DetectMultiBackend
from YOLOV5_Gestures.utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from YOLOV5_Gestures.utils.torch_utils import select_device, time_sync
from YOLOV5_Gestures.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from YOLOV5_Gestures.utils.plots import Annotator, colors


# class OakDDetections:
#     def __init__(self, rgb, depth, detections):
#         self.detections = detections
#         self.rects = []
#         self.depthRects = []
#         height, width, dim = rgb.shape
#         for detection in self.detections:
#             if detection.label != 0: #person
#                 continue
#             if detection.confidence < 10:
#                 continue
#             roiData = detection.boundingBoxMapping
#             roi = roiData.roi
#             roi = roi.denormalize(depth.shape[1], depth.shape[0])
#             topLeft = roi.topLeft()
#             bottomRight = roi.bottomRight()
#             #depth
#             xmin = int(topLeft.x)
#             ymin = int(topLeft.y)
#             xmax = int(bottomRight.x)
#             ymax = int(bottomRight.y)
#             self.depthRects.append((xmin, ymin, xmax, ymax))
#             # rgb
#             # Denormalize bounding box
#             x1 = int(detection.xmin * width)
#             x2 = int(detection.xmax * width)
#             y1 = int(detection.ymin * height)
#             y2 = int(detection.ymax * height)
#             self.rects.append(((x1, y1, x2, y2), detection))
#     def find_closest_rectangle(self, point):

#                         OakDDetections(rgb, depth, oak_d_detections)
#                         if rects:
#                             index, closest_dist, detectedObject = find_closest_rectangle(center, rects)
#                             if closest_dist < 300:
#                                 x1 = center[0]
#                                 y1 = center[1]
#                                 cv2.putText(im0, "{:.2f}".format(detectedObject.confidence), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0,0,255))
#                                 cv2.putText(im0, f"X: {int(detectedObject.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0,255,0))
#                                 cv2.putText(im0, f"Y: {int(detectedObject.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0,255,0))
#                                 cv2.putText(im0, f"Z: {int(detectedObject.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0,255,0))
#                         if depthRects:
#                             for xmin, ymin, xmax, ymax in depthRects:
#                                 cv2.rectangle(depth0, (xmin, ymin), (xmax, ymax), (255,255,255), 2)
#                         ##########################################################################                

class ModelWrapper:
    def __init__(self, weights, device="", imgsz=640, dnn=True):
        # Load model
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device, dnn=dnn)
        self.stride, self.names, self.pt, self.jit, self.onnx = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size
        # Half
        self.half = self.pt and self.device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if self.pt:
            self.model.model.half() if self.half else self.model.model.float()
        ### warmup
        if self.pt and self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, *self.imgsz).to(self.device).type_as(next(self.model.model.parameters())))
        self.configurePrediction()
    def configurePrediction(self, conf_thres=0.25, iou_thres=0.45, max_det=1000, agnostic_nms=False, classes=None):
        self.conf_thres = conf_thres,  # confidence threshold
        self.iou_thres = iou_thres,  # NMS IOU threshold
        self.max_det = max_det # maximum detections per image
        self.agnostic_nms = agnostic_nms # class-agnostic NMS
        self.classes = classes # filter by classes eg --classes 0 1 2
    def transformInputRgb(self, rgb):
        img = rgb.copy()
        img = letterbox(img, self.imgsz, stride=self.stride)
        img = img[..., ::-1].transpose((0, 3, 1, 2))
        img = np.ascontiguousarray(img)
        return img
    def transformInputDepth(self, depth):
        img = depth.copy()
        img = letterbox(img, self.imgsz, stride=self.stride)
        downscaled = img[0][::4]
        if np.all(downscaled == 0):
            min_depth = 0  # Set a default minimum depth value when all elements are zero
        else:
            min_depth = np.percentile(downscaled[downscaled != 0], 1)
        max_depth = np.percentile(downscaled, 99)
        depthFrameColor = np.interp(depth[0], (min_depth, max_depth), (0, 255)).astype(np.uint8)
        img = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
        img = np.ascontiguousarray(img)  
        return img
    def run(self, rgb):
        # Inference
        img = rgb.copy()
        img = torch.from_numpy(rgb).to(self.device)
        img = rgb.half() if self.half else rgb.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0，normalization
        # if len(img.shape) == 3:
        #     img = img[None]  # expand for batch dim
        pred = self.model(img, augment=False, visualize=False)
        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        predictions = []
        for det in enumerate(pred):
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(rgb.shape[2:], det[:, :4], rgb.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    predictions.append(xyxy, conf, cls)
        return predictions, img
def annotateImages(rgb, depth, predictions, oakddetections):
    # annotator = None
    # if liveAnnotation:
    #     line_thickness=3,  # bounding box thickness (pixels)
    #     annotator = Annotator(rgb, line_width=line_thickness, example=str(self.names))
    pass