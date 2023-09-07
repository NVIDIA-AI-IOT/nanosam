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
from nanosam.utils.trt_pose import PoseDetector, pose_to_sam_points

parser = argparse.ArgumentParser()
parser.add_argument("--image_encoder", type=str, default="data/resnet18_image_encoder.engine")
parser.add_argument("--mask_decoder", type=str, default="data/mobile_sam_mask_decoder.engine")
args = parser.parse_args()

def get_torso_points(pose):
    return pose_to_sam_points(
        pose,
        ["left_shoulder", "right_shoulder"],
        ["nose", "left_ear", "right_ear", "right_wrist", "left_wrist", "left_knee", "right_knee"]
    )

def cv2_to_pil(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return PIL.Image.fromarray(image)

pose_model = PoseDetector(
    "data/densenet121_baseline_att_256x256_B_epoch_160.pth",
    "assets/human_pose.json"
)

predictor = Predictor(
    args.image_encoder,
    args.mask_decoder
)

mask = None

cap = cv2.VideoCapture(0)

cv2.namedWindow('image')

while True:

    re, image = cap.read()


    if not re:
        break

    image_pil = cv2_to_pil(image)

    detections = pose_model.predict(image_pil)

    if len(detections) > 0:
        pose = detections[0]

        points, point_labels = get_torso_points(detections[0])

        predictor.set_image(image_pil)
        mask, _, _ = predictor.predict(points, point_labels)
    
        # Draw mask
        if mask is not None:
            bin_mask = (mask[0,0].detach().cpu().numpy() < 0)
            green_image = np.zeros_like(image)
            green_image[:, :] = (0, 185, 118)
            green_image[bin_mask] = 0

            image = cv2.addWeighted(image, 0.4, green_image, 0.6, 0)


        for pt, lab in zip(points, point_labels):
            xy = (int(pt[0]), int(pt[1]))
            if lab == 1:
                cv2.circle(image, xy, 8, (0, 185, 118), -1)
            else:
                cv2.circle(image, xy, 8, (0, 0, 185), -1)


    cv2.imshow("image", image)

    ret = cv2.waitKey(1)

    if ret == ord('q'):
        break


cv2.destroyAllWindows()
