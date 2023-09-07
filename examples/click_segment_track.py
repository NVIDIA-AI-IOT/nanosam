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

import PIL.Image
import cv2
import numpy as np
import argparse
from nanosam.utils.predictor import Predictor
from nanosam.utils.tracker import Tracker

parser = argparse.ArgumentParser()
parser.add_argument("--image_encoder", type=str, default="data/resnet18_image_encoder.engine")
parser.add_argument("--mask_decoder", type=str, default="data/mobile_sam_mask_decoder.engine")
args = parser.parse_args()

def cv2_to_pil(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return PIL.Image.fromarray(image)

predictor = Predictor(
    args.image_encoder,
    args.mask_decoder
)

tracker = Tracker(predictor)

mask = None
point = None

cap = cv2.VideoCapture(0)


def init_track(event,x,y,flags,param):
    global mask, point
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mask = tracker.init(image_pil, point=(x, y))
        point = (x, y)


cv2.namedWindow('image')
cv2.setMouseCallback('image',init_track)

while True:

    re, image = cap.read()


    if not re:
        break

    image_pil = cv2_to_pil(image)

    if tracker.token is not None:
        mask, point = tracker.update(image_pil)
    
    # Draw mask
    if mask is not None:
        bin_mask = (mask[0,0].detach().cpu().numpy() < 0)
        green_image = np.zeros_like(image)
        green_image[:, :] = (0, 185, 118)
        green_image[bin_mask] = 0

        image = cv2.addWeighted(image, 0.4, green_image, 0.6, 0)

    # Draw center
    if point is not None:

        image = cv2.circle(
            image,
            point,
            5,
            (0, 185, 118),
            -1
        )

    cv2.imshow("image", image)

    ret = cv2.waitKey(1)

    if ret == ord('q'):
        break
    elif ret == ord('r'):
        tracker.reset()
        mask = None
        box = None


cv2.destroyAllWindows()
