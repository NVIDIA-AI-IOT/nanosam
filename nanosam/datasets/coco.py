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
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO
import PIL.Image as Image
import os
import numpy as np


class CocoDetectionWithAlbumentations(Dataset):
    def __init__(self, root_dir, annotation_file, transforms=None, cat_ids=[], max_boxes=64):
        self.coco = COCO(annotation_file)
        self.root_dir = root_dir
        self.transforms = transforms
        self.cat_ids = cat_ids
        self.max_boxes = max_boxes

    def __len__(self):
        return len(self.coco.getImgIds())

    def __getitem__(self, idx):
        img_id = self.coco.getImgIds()[idx]
        # print(img_id)
        img_info = self.coco.loadImgs(img_id)[0]
        image_path = os.path.join(self.root_dir, img_info['file_name'])
        image = Image.open(image_path).convert("RGB")
        image = np.asarray(image)
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids)
        annotations = self.coco.loadAnns(ann_ids)

        # Extract bounding boxes and class labels from annotations
        bboxes = [ann['bbox'] for ann in annotations]
        labels = [ann['category_id'] for ann in annotations]

        target = {
            'bboxes': bboxes,
            'labels': labels
        }

        if self.transforms:
            transformed = self.transforms(image=image, bboxes=target['bboxes'], labels=target['labels'])
            image = transformed['image']
            target['bboxes'] = np.array(transformed['bboxes'])
            target['labels'] = np.array(transformed['labels'])
        
        # pad
        labels = -np.ones_like(target['labels'], shape=(self.max_boxes,))
        nlabels = min(self.max_boxes, len(target['labels']))
        if nlabels > 0:
            labels[:nlabels] = target['labels'][:nlabels]

        bboxes = np.zeros_like(target['bboxes'], shape=(self.max_boxes, 4))
        nboxes = min(self.max_boxes, len(target['bboxes']))
        if nboxes> 0:
            bboxes[:nboxes] = target['bboxes'][:nboxes]

        target['labels'] = labels
        target['bboxes'] = bboxes

        return image, target