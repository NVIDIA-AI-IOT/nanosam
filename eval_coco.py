import pycocotools.mask as m
import json
import os
from torchvision.datasets import CocoDetection
import pycocotools.coco as coco
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from mobile_sam.predictor_trt import PredictorTRT

dataset = CocoDetection(
    root="data/coco_val/val2017",
    annFile="data/coco_val/annotations/instances_val2017.json"
)

mobile_sam_predictor = PredictorTRT(
    image_encoder_engine="data/mobile_sam_image_encoder.engine",
    mask_decoder_engine="data/mobile_sam_mask_decoder.engine",
    image_encoder_size=1024
)

nano_sam_predictor = PredictorTRT(
    image_encoder_engine="data/resnet18_huber_1024_v5.engine",
    mask_decoder_engine="data/mobile_sam_mask_decoder.engine",
    image_encoder_size=1024
)


def predict_box(predictor, image, box, set_image=True):

    if set_image:
        predictor.set_image(image)

    points = np.array([
        [box[0], box[1]],
        [box[2], box[3]]
    ])
    point_labels = np.array([2, 3])

    mask, iou_preds, low_res_mask = predictor.predict(
        points=points,
        point_labels=point_labels
    )

    mask = mask[0, iou_preds.argmax()].detach().cpu().numpy() > 0
    
    return mask

def predict_box_roi(predictor, image: PIL.Image.Image, box, scale):
    w = box[2] - box[0]
    h = box[3] - box[1]
    s = scale * max(w, h)
    cx = box[0] + w / 2
    cy = box[1] + h / 2
    roi = [
        int(cx - s / 2),
        int(cy - s / 2),
        int(cx + s / 2),
        int(cy + s / 2)
    ]
    dw = (s - w) / 2
    dh = (s - h) / 2
    box_in_roi = [
        dw,
        dh,
        s - dw,
        s - dh
    ]
    
    image_crop = image.crop(roi)
    mask = predict_box(predictor, image_crop, box_in_roi)
    mask = PIL.Image.fromarray(mask.astype(np.uint8))
    mask_image = np.zeros((image.height, image.width), dtype=np.uint8)
    mask_image = PIL.Image.fromarray(mask_image)
    mask_image.paste(mask, roi)

    return np.asarray(mask_image) > 0
    
def box_xywh_to_xyxy(box):
    return [box[0], box[1], box[0] + box[2], box[1] + box[3]]

def draw_box(box):
    x = [box[0], box[0], box[2], box[2], box[0]]
    y = [box[1], box[3], box[3], box[1], box[1]]
    plt.plot(x, y, 'g-')
    
def iou(mask_a, mask_b):
    intersection = np.count_nonzero(mask_a & mask_b)
    union = np.count_nonzero(mask_a | mask_b)
    return intersection / union

ann = dataset[0][1][0]

results = []
for i in tqdm.tqdm(range(len(dataset))):
    image, anns = dataset[i]
    for j, ann in enumerate(anns):
        id = ann['id']
        area = ann['area']
        category_id = ann['category_id']
        iscrowd = ann['iscrowd']
        image_id = ann['image_id']
        box = box_xywh_to_xyxy(ann['bbox'])
        mask = dataset.coco.annToMask(ann)
        mask_coco = (mask > 0)
        mask_nano_sam = predict_box(nano_sam_predictor, image, box, set_image=(j==0))
        mask_mobile_sam = predict_box(mobile_sam_predictor, image, box, set_image=(j==0))

        result = {
            "id": ann['id'],
            "area": ann['area'],
            "category_id": ann['category_id'],
            "iscrowd": ann['iscrowd'],
            "image_id": ann["image_id"],
            "box": box,
            "iou_nanosam_coco": iou(mask_nano_sam, mask_coco),
            "iou_mobilesam_coco": iou(mask_mobile_sam, mask_coco),
            "iou_mobilesam_nanosam": iou(mask_mobile_sam, mask_nano_sam)
        }
        results.append(result)

with open("eval_coco_results.json", 'w') as f:
    json.dump(results, f)

# print(ann)
# results = []

# for i in 
# mask = dataset.coco.annToMask(ann)

# mask_coco = (mask > 0)
# box = box_xywh_to_xyxy(ann['bbox'])
# mask_nanosam = predict_box(nano_sam_predictor, image, box)
# mask_mobilesam = predict_box(mobile_sam_predictor, image, box)
# # mask_mobilesam_roi = predict_box_roi(mobile_sam_predictor, image, box, scale=1.5)
# # mask_nanosam_roi = predict_box_roi(nano_sam_predictor, image, box, scale=1.5)

# print(f"IOU(mobilesam, coco): {iou(mask_mobilesam, mask_coco)}")
# print(f"IOU(nanosam, coco): {iou(mask_nanosam, mask_coco)}")
# print(f"IOU(mobilesam, nanosam): {iou(mask_mobilesam, mask_nanosam)}")
# # print(f"IOU(mobilesam_roi, coco): {iou(mask_mobilesam_roi, mask_coco)}")
# # print(f"IOU(nanosam_roi, coco): {iou(mask_nanosam_roi, mask_coco)}")
# # print(f"IOU(mobilesam_roi, nanosam_roi): {iou(mask_mobilesam_roi, mask_nanosam_roi)}")

# plt.subplot(131)
# plt.imshow(image)
# plt.imshow(mask_coco, alpha=0.5)
# # draw_box(box)
# plt.subplot(132)
# plt.imshow(image)
# plt.imshow(mask_mobilesam, alpha=0.5)
# plt.subplot(133)
# plt.imshow(image)
# plt.imshow(mask_nanosam, alpha=0.5)
# # plt.subplot(234)
# # plt.imshow(image)
# # plt.imshow(mask_mobilesam_roi, alpha=0.5)
# # plt.subplot(234)
# # plt.imshow(image)
# # plt.imshow(mask_nanosam_roi, alpha=0.5)

# plt.savefig("mask.png")