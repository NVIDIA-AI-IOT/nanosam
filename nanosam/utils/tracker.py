# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
from .predictor import Predictor, upscale_mask
import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

def bbox2points(box):
    return np.array([[box[0], box[1]], [box[2], box[3]]]), np.array([2, 3])


def down_to_64(x):
    return F.interpolate(x, (64, 64), mode="area")


def up_to_256(x):
    return F.interpolate(x, (256, 256), mode="bilinear")

def mask_to_box(mask):
    mask = mask[0, 0] > 0
    mask = mask.detach().cpu().numpy()
    mask_pts = np.argwhere(mask)
    min_y = np.min(mask_pts[:, 0])
    min_x = np.min(mask_pts[:, 1])
    max_y = np.max(mask_pts[:, 0])
    max_x = np.max(mask_pts[:, 1])
    next_box = np.array([min_x, min_y, max_x, max_y])
    return next_box

def mask_to_centroid(mask):
    mask = mask[0, 0] > 0
    mask = mask.detach().cpu().numpy()
    mask_pts = np.argwhere(mask)
    center_y = np.median(mask_pts[:, 0])
    center_x = np.median(mask_pts[:, 1])
    return np.array([center_x, center_y])

def mask_to_sample_points(mask):
    mask = mask[0, 0] > 0
    mask = mask.detach().cpu().numpy()
    fg_mask_pts = np.argwhere(mask)
    fg_mask_pts_selected = np.random.choice(len(fg_mask_pts), 1)
    bg_mask_pts = np.argwhere(mask == False)
    bg_mask_pts_selected = np.random.choice(len(bg_mask_pts), 1)
    return fg_mask_pts[fg_mask_pts_selected], bg_mask_pts[bg_mask_pts_selected]

class Tracker(object):

    def __init__(self,
            predictor: Predictor
        ):
        self.predictor = predictor
        self.target_mask = None
        self.token = None
        self._targets = []
        self._features = []

    def set_image(self, image):
        self.predictor.set_image(image)

    def predict_mask(self, points=None, point_labels=None, box=None, mask_input=None):
        
        if box is not None:
            points, point_labels = bbox2points(box)

        mask_high, iou_pred, mask_raw = self.predictor.predict(points, point_labels, mask_input=mask_input)

        idx = int(iou_pred.argmax())
        mask_raw = mask_raw[:, idx:idx+1, :, :]
        mask_high = mask_high[:, idx:idx+1, :, :]
        return mask_high, mask_raw, down_to_64(mask_raw)
    

    def fit_token(self, features, mask_low):
        """
        Finds token that when dot-producted with features minimizes MSE with low 
        resolution masks.

        Args:
            features (Nx256x64x64)
            mask (Nx1x64x64) - Should be logits type
        """
        with torch.no_grad():
            N = features.shape[0]
            assert N == mask_low.shape[0]
            A = features.permute(0, 2, 3, 1).reshape(N * 64 * 64, 256)
            B = mask_low.permute(0, 2, 3, 1).reshape(N * 64 * 64, 1)
            X = torch.linalg.lstsq(A, B).solution.reshape(1, 256, 1, 1)
        return X.detach()

    
    def apply_token(self, features, token):
        return up_to_256(torch.sum(features * token, dim=(1), keepdim=True))
    
    @torch.no_grad()
    def init(self, image, point=None, box=None):
        self.set_image(image)

        if point is not None:
            mask_high, mask_raw, mask_low = self.predict_mask(np.array([point]), np.array([1]))

        self.token = self.fit_token(self.predictor.features, mask_low)
        self.init_token = self.token

        return mask_high
        
    def reset(self):
        self._features = []
        self._targets = []
        self.token = None

    @torch.no_grad()
    def update(self, image):
        self.set_image(image)
        mask_token = self.apply_token(self.predictor.features, self.token)
        mask_token_up = upscale_mask(mask_token, (image.height, image.width))
        if torch.count_nonzero(mask_token_up > 0.0) > 1:
            # fg_points, bg_points = mask_to_sample_points(mask_token_up)
            # points = np.concatenate([fg_points, bg_points], axis=0)
            # point_labels = np.concatenate([np.ones((len(fg_points),), dtype=np.int64), np.zeros((len(bg_points),), dtype=np.int64)], axis=0)
            # box = mask_to_box(mask_token_up)
            points = np.array([mask_to_centroid(mask_token_up)])
            point_labels = np.array([1])
            mask_high, mask_raw, mask_low = self.predict_mask(points, point_labels, mask_input=mask_token)
            self.token = 0.995 * self.token + 0.005 * self.fit_token(self.predictor.features, mask_low)
            self._result = mask_high, points[0]
            self._result = mask_token_up, points[0]
            return mask_high
        else:
            return None