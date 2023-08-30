import torch
import torch.nn.functional as F
from nanosam.predictor import Predictor, upscale_mask
import numpy as np
import PIL.Image

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
    mask_pts = np.argwhere(mask)
    mask_pts_selected = np.random.choice(len(mask_pts), 10)
    return mask_pts[mask_pts_selected]

class Tracker(object):

    def __init__(self,
            predictor: Predictor
        ):
        self.predictor = predictor
        self.target_mask = None
        self.token = None

    def set_image(self, image):
        self.predictor.set_image(image)

    def predict_mask(self, points, point_labels, mask_input=None):
        
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
    def init_point(self, image, point):
        self.set_image(image)
        mask_high, mask_raw, mask_low = self.predict_mask_point(point)
        box = mask_to_box(mask_high)
        self.token = self.fit_token(self.predictor.features, mask_low)
        mask_token = self.apply_token(self.predictor.features, self.token)
        mask_high, mask_raw, mask_low = self.predict_mask(box, mask_input=mask_token)
        self.mask = mask_high
        return mask_high, mask_to_centroid(mask_high)

    @torch.no_grad()
    def init(self, image, box):
        self.set_image(image)
        mask_high, mask_raw, mask_low = self.predict_mask(box)
        self.token = self.fit_token(self.predictor.features, mask_low)
        mask_token = self.apply_token(self.predictor.features, self.token)
        mask_high, mask_raw, mask_low = self.predict_mask(box, mask_input=mask_token)
        self.mask = mask_high
        return mask_high, mask_to_box(mask_high)

    @torch.no_grad()
    def update(self, image):
        self.set_image(image)
        mask_token = self.apply_token(self.predictor.features, self.token)
        mask_token_up = upscale_mask(mask_token, (image.height, image.width))
        if torch.count_nonzero(mask_token_up > 0) > 1:
            box = mask_to_box(mask_token_up)
            mask_high, mask_raw, mask_low = self.predict_mask(box, mask_input=mask_token)
            self.token = self.fit_token(self.predictor.features, mask_low)
            result = mask_high, mask_to_box(mask_high)
            return result
        else:
            return None, None

        
    @torch.no_grad()
    def update_point(self, image):
        self.set_image(image)
        mask_token = self.apply_token(self.predictor.features, self.token)
        mask_token_up = upscale_mask(mask_token, (image.height, image.width))
        if torch.count_nonzero(mask_token_up > 0) > 1:
            point = mask_to_centroid(mask_token_up)
            mask_high, mask_raw, mask_low = self.predict_mask_point(point, mask_input=mask_token)
            self.token = 0.99 * self.token + 0.01 * self.fit_token(self.predictor.features, mask_low)
            result = mask_high, mask_to_centroid(mask_high)
            self._result = result
            return result
        else:
            return self._result