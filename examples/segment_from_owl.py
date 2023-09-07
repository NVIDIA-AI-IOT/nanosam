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
import matplotlib.pyplot as plt
import numpy as np
import argparse
from nanosam.utils.owlvit import OwlVit
from nanosam.utils.predictor import Predictor


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str)
    parser.add_argument("--prompt", nargs='+', required=True)
    parser.add_argument("--image_encoder", type=str, default="data/resnet18_image_encoder.engine")
    parser.add_argument("--mask_decoder", type=str, default="data/mobile_sam_mask_decoder.engine")
    parser.add_argument("--thresh", type=float, default=0.1)
    args = parser.parse_args()
        
    def bbox2points(bbox):
        points = np.array([
            [bbox[0], bbox[1]],
            [bbox[2], bbox[3]]
        ])

        point_labels = np.array([2, 3])

        return points, point_labels

    def draw_bbox(bbox):
        x = [bbox[0], bbox[2], bbox[2], bbox[0], bbox[0]]
        y = [bbox[1], bbox[1], bbox[3], bbox[3], bbox[1]]
        plt.plot(x, y, 'g-')


    detector = OwlVit(args.thresh)

    image = PIL.Image.open(args.image)

    detections = detector.predict(image, texts=args.prompt)

    sam_predictor = Predictor(
        args.image_encoder,
        args.mask_decoder
    )

    sam_predictor.set_image(image)
    N = len(detections)

    def subplot_notick(a, b, c):
        ax = plt.subplot(a, b, c)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.axis('off')

    def draw_detection(index):
        subplot_notick(2, N, index + 1)
        bbox = detections[index]['bbox']
        points, point_labels = bbox2points(bbox)
        mask, _, _ = sam_predictor.predict(points, point_labels)
        plt.imshow(image)
        draw_bbox(bbox)
        subplot_notick(2, N, N + index + 1)
        plt.imshow(image)
        plt.imshow(mask[0, 0].detach().cpu() > 0, alpha=0.5)


    AR = image.width / image.height
    plt.figure(figsize=(25, 10))
    for i in range(N):
        draw_detection(i)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("out.png", bbox_inches="tight")