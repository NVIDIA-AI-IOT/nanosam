import cv2
import json
import PIL.Image
import torch
import numpy as np
import torchvision.transforms as transforms
import trt_pose.coco
import trt_pose.models
import matplotlib.pyplot as plt
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
from nanosam.utils.predictor import Predictor

class PoseDetector(object):

    def __init__(self, 
            weights,
            config,
            arch="densenet121_baseline_att",
            shape=(256, 256)
        ):

        with open(config, 'r') as f:
            human_pose = json.load(f)

        self.keypoints = human_pose['keypoints']
        num_parts = len(human_pose['keypoints'])
        num_links = len(human_pose['skeleton'])

        self.topology = trt_pose.coco.coco_category_to_topology(human_pose)

        if arch == "densenet121_baseline_att":
            self.model = trt_pose.models.densenet121_baseline_att(num_parts, 2 * num_links).cuda().eval()

        self.model.load_state_dict(torch.load(weights))

        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        self.parse_objects = ParseObjects(self.topology)
        self.draw_objects = DrawObjects(self.topology)
        self.shape = shape

    @torch.no_grad()
    def preprocess(self, image):
        image = transforms.functional.to_tensor(image).cuda()
        image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return image[None, ...]
    
    @torch.no_grad()
    def postprocess(self, image_shape, cmap, paf):
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        object_counts, objects, normalized_peaks = self.parse_objects(cmap, paf)
        topology = self.topology
        height = image_shape[1]
        width = image_shape[0]
        
        detections = []

        count = int(object_counts[0])
        for i in range(count):
            obj = objects[0][i]
            C = obj.shape[0]
            pose = {'keypoints': []}
            for j in range(C):
                k = int(obj[j])
                if k >= 0:
                    peak = normalized_peaks[0][j][k]
                    x = float(peak[1]) * width
                    y = float(peak[0]) * height
                    kp = {
                        'name': self.keypoints[j],
                        'index': j,
                        'x': x,
                        'y': y
                    }
                    pose['keypoints'].append(kp)
            detections.append(pose)

        return detections
    
    @torch.no_grad()
    def predict(self, image):
        width, height = image.width, image.height
        image = image.resize(self.shape)
        data = self.preprocess(image)
        cmap, paf = self.model(data)
        return self.postprocess((width, height), cmap, paf)


def pose_to_sam_points(pose, fg_kps, bg_kps):

    points = []
    point_labels = []

    for kp in pose['keypoints']:
        if kp['name'] in fg_kps:
            points.append([kp['x'], kp['y']])
            point_labels.append(1)
        if kp['name'] in bg_kps:
            points.append([kp['x'], kp['y']])
            point_labels.append(0)

    points = np.array(points)
    point_labels = np.array(point_labels)

    return points, point_labels
