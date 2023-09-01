import PIL.Image
import cv2
import numpy as np
import argparse
from nanosam.utils.owlvit import OwlVit
from nanosam.utils.predictor import Predictor
from nanosam.utils.tracker_online_learning import TrackerOnline

parser = argparse.ArgumentParser()
parser.add_argument("prompt")
args = parser.parse_args()
prompt = args.prompt

def cv2_to_pil(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return PIL.Image.fromarray(image)


# owlvit = OwlVit(threshold=0.1)

predictor = Predictor(
    "data/resnet18_image_encoder.engine",
    "data/mobile_sam_mask_decoder.engine"
)

tracker = TrackerOnline(predictor)

mask = None
point = None

cap = cv2.VideoCapture(0)


def init_track(event,x,y,flags,param):
    global mask, point
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mask = tracker.init(image_pil, point=(x, y))


cv2.namedWindow('image')
cv2.setMouseCallback('image',init_track)

while True:

    re, image = cap.read()


    if not re:
        break

    image_pil = cv2_to_pil(image)

    # if box is None:

    #     detections = owlvit.predict(image_pil, texts=[prompt])

    #     if len(detections) > 0:
    #         box = detections[0]['box']
    #         mask, box = tracker.init(image_pil, box)
        
    if tracker.object_model is not None:
        mask = tracker.update(image_pil)
    
    if mask is not None:
        bin_mask = mask[0,0].detach().cpu().numpy() < 0
        image[bin_mask] = (0.1 * image[bin_mask]).astype(np.uint8)
    else:
        image = (0.1 * image).astype(np.uint8)
    # if point is not None:
    #     cv2.circle(image, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
    cv2.imshow("image", image)

    ret = cv2.waitKey(1)

    if ret == ord('q'):
        break
    elif ret == ord('r'):
        tracker.reset()
        mask = None
        box = None


cv2.destroyAllWindows()
