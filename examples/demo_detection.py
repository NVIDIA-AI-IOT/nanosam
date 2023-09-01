import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
from nanosam.utils.owlvit import OwlVit
from nanosam.utils.predictor import Predictor


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


detector = OwlVit()

image = PIL.Image.open("assets/john_1.jpg")

detections = detector.predict(image, texts=["a tree"])

sam_predictor = Predictor(
    "data/resnet18_image_encoder.engine",
    "data/mobile_sam_mask_decoder.engine"
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