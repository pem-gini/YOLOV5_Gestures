
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
        cudnn.benchmark = True  # set True to speed up constant image size inference
        self.configurePrediction()
    def configurePrediction(self, conf_thres=0.25, iou_thres=0.45, max_det=1000, agnostic_nms=False, classes=None):
        self.conf_thres = conf_thres  # confidence threshold
        self.iou_thres = iou_thres  # NMS IOU threshold
        self.max_det = max_det # maximum detections per image
        self.agnostic_nms = agnostic_nms # class-agnostic NMS
        self.classes = classes # filter by classes eg --classes 0 1 2
    def transformInputRgb(self, rgb):
        img = rgb.copy()
        # img = np.array([img])
        resized = [letterbox(img, self.imgsz, stride=self.stride)[0]]
        img = np.stack(resized, 0)
        img = img[..., ::-1].transpose((0, 3, 1, 2))
        img = np.ascontiguousarray(img)
        return resized[0], img 
    def transformInputDepth(self, depth0):
        depth = depth0.copy()
        ### Normalize the 16-bit depth image to an 8-bit range
        depth = cv2.convertScaleAbs(depth, alpha=(255.0/65535.0))
        ### Convert the single-channel 8-bit image to a 3-channel 8-bit image by replicating the gray values across the 3 channels
        depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
        depth = letterbox(depth, self.imgsz, stride=self.stride)[0]
        downscaled = depth[::4]
        if np.all(downscaled == 0):
            min_depth = 0  # Set a default minimum depth value when all elements are zero
        else:
            min_depth = np.percentile(downscaled[downscaled != 0], 1)
        max_depth = np.percentile(downscaled, 99)
        depthFrameColor = np.interp(depth, (min_depth, max_depth), (0, 255)).astype(np.uint8)
        img = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
        img = np.ascontiguousarray(img)  
        return img
    def run(self, rgb):
        # Inference
        img = rgb.copy()
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0，normalization
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        pred = self.model(img, augment=False, visualize=False)
        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        predictions = []
        for i, det in enumerate(pred):
            if len(det):
                ### Rescale boxes from img_size to im0 size
                # tensor([[212.40894, 135.32660, 391.62476, 456.87207]]) (3, 640, 640)
                ### def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
                # det[:, :4] = scale_coords(img.shape[2:], boxcords, rgb[0].shape).round()
                for *xyxy, conf, cls in reversed(det):
                    predictions.append((xyxy, conf, int(cls)))
        return predictions
    def lookupName(self, cls):
        if cls < len(self.names):
            return self.names[cls]
        return ""