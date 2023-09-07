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

from torch2trt import TRTModule
from typing import Tuple
import tensorrt as trt
import PIL.Image
import torch
import numpy as np
import torch.nn.functional as F

def load_mask_decoder_engine(path: str):
    
    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(path, 'rb') as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)

    mask_decoder_trt = TRTModule(
        engine=engine,
        input_names=[
            "image_embeddings",
            "point_coords",
            "point_labels",
            "mask_input",
            "has_mask_input"
        ],
        output_names=[
            "iou_predictions",
            "low_res_masks"
        ]
    )

    return mask_decoder_trt


def load_image_encoder_engine(path: str):

    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(path, 'rb') as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)

    image_encoder_trt = TRTModule(
        engine=engine,
        input_names=["image"],
        output_names=["image_embeddings"]
    )

    return image_encoder_trt


def preprocess_image(image, size: int = 512):

    if isinstance(image, np.ndarray):
        image = PIL.Image.fromarray(image)

    image_mean = torch.tensor([123.675, 116.28, 103.53])[:, None, None]
    image_std = torch.tensor([58.395, 57.12, 57.375])[:, None, None]

    image_pil = image
    aspect_ratio = image_pil.width / image_pil.height
    if aspect_ratio >= 1:
        resize_width = size
        resize_height = int(size / aspect_ratio)
    else:
        resize_height = size
        resize_width = int(size * aspect_ratio)

    image_pil_resized = image_pil.resize((resize_width, resize_height))
    image_np_resized = np.asarray(image_pil_resized)
    image_torch_resized = torch.from_numpy(image_np_resized).permute(2, 0, 1)
    image_torch_resized_normalized = (image_torch_resized.float() - image_mean) / image_std
    image_tensor = torch.zeros((1, 3, size, size))
    image_tensor[0, :, :resize_height, :resize_width] = image_torch_resized_normalized

    return image_tensor.cuda()


def preprocess_points(points, image_size, size: int = 1024):
    scale = size / max(*image_size)
    points = points * scale
    return points


def run_mask_decoder(mask_decoder_engine, features, points=None, point_labels=None, mask_input=None):
    if points is not None:
        assert point_labels is not None
        assert len(points) == len(point_labels)

    image_point_coords = torch.tensor([points]).float().cuda()
    image_point_labels = torch.tensor([point_labels]).float().cuda()

    if mask_input is None:
        mask_input = torch.zeros(1, 1, 256, 256).float().cuda()
        has_mask_input = torch.tensor([0]).float().cuda()
    else:
        has_mask_input = torch.tensor([1]).float().cuda()


    iou_predictions, low_res_masks = mask_decoder_engine(
        features,
        image_point_coords,
        image_point_labels,
        mask_input,
        has_mask_input
    )

    return iou_predictions, low_res_masks


def upscale_mask(mask, image_shape, size=256):
    
    if image_shape[1] > image_shape[0]:
        lim_x = size
        lim_y = int(size * image_shape[0] / image_shape[1])
    else:
        lim_x = int(size * image_shape[1] / image_shape[0])
        lim_y = size

    mask[:, :, :lim_y, :lim_x]
    mask = F.interpolate(mask[:, :, :lim_y, :lim_x], image_shape, mode='bilinear')
    
    return mask


class Predictor(object):

    def __init__(self,
            image_encoder_engine: str,
            mask_decoder_engine: str,
            image_encoder_size: int = 1024,
            orig_image_encoder_size: int = 1024,
        ):
        self.image_encoder_engine = load_image_encoder_engine(image_encoder_engine)
        self.mask_decoder_engine = load_mask_decoder_engine(mask_decoder_engine)
        self.image_encoder_size = image_encoder_size
        self.orig_image_encoder_size = orig_image_encoder_size

    def set_image(self, image):
        self.image = image
        self.image_tensor = preprocess_image(image, self.image_encoder_size)
        self.features = self.image_encoder_engine(self.image_tensor)

    def predict(self, points, point_labels, mask_input=None):
        points = preprocess_points(
            points, 
            (self.image.height, self.image.width),
            self.orig_image_encoder_size
        )
        mask_iou, low_res_mask = run_mask_decoder(
            self.mask_decoder_engine,
            self.features,
            points,
            point_labels,
            mask_input
        )

        hi_res_mask = upscale_mask(
            low_res_mask, 
            (self.image.height, self.image.width)                           
        )

        return hi_res_mask, mask_iou, low_res_mask