import json
from torchvision.datasets import CocoDetection
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import argparse
from nanosam.utils.predictor import Predictor


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_root", type=str, default="data/coco/val2017")
    parser.add_argument("--coco_ann", type=str, default="data/coco/annotations/instances_val2017.json")
    parser.add_argument("--image_encoder", type=str, default="data/mobile_sam_image_encoder.engine")
    parser.add_argument("--mask_decoder", type=str, default="data/mobile_sam_mask_decoder.engine")
    parser.add_argument("--output", type=str, default="data/mobile_sam_coco_results.json")
    args = parser.parse_args()

        
    dataset = CocoDetection(
        root=args.coco_root,
        annFile=args.coco_ann
    )

    predictor = Predictor(
        image_encoder_engine=args.image_encoder,
        mask_decoder_engine=args.mask_decoder,
        image_encoder_size=1024
    )

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
            mask_sam = predict_box(predictor, image, box, set_image=(j==0))

            result = {
                "id": ann['id'],
                "area": ann['area'],
                "category_id": ann['category_id'],
                "iscrowd": ann['iscrowd'],
                "image_id": ann["image_id"],
                "box": box,
                "iou": iou(mask_sam, mask_coco)
            }

            results.append(result)

    with open(args.output, 'w') as f:
        json.dump(results, f)
