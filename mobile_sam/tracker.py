import cv2
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


def read_image(path: str):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


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

class SamTracker:
    
    def __init__(self, checkpoint, token_history=1):
        model_type = "vit_t"
        sam_checkpoint = checkpoint
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        mobile_sam.to(device=device)
        mobile_sam.eval()
        self.predictor = SamPredictor(mobile_sam)
        self._feature_buffer = []
        self._mask_buffer = []
        self._token_history = token_history

    def compute_features(self, image):
        self.predictor.set_image(image)
        return self.predictor.features
        
    def compute_features_and_mask(self, image, bbox):
        self.predictor.set_image(image)
        _, _, mask = self.predictor.predict(
            box=bbox,
            multimask_output=False
        )
        return self.predictor.features, torch.from_numpy(mask[None, ...]).cuda()

    def refine_mask(self, image, bbox, mask, scale=50., iters=2, set_image=True):
        if set_image:
            self.predictor.set_image(image)
        mask = scale * mask[0] # remove batch dim
        for i in range(iters):
            _, _, mask = self.predictor.predict(
                box=bbox,
                multimask_output=False,
                mask_input=mask
            )
            mask = mask * scale
        return torch.from_numpy(mask[None, ...]).cuda() # add batch dim

    def fit_lstsq_mask_token(self, features, mask):
        """
        Finds token that when dot-producted with features minimizes MSE with low 
        resolution masks.

        Args:
            features (Nx256x64x64)
            mask (Nx1x64x64) - Should be logits type
        """
        with torch.no_grad():
            N = features.shape[0]
            assert N == mask.shape[0]
            A = features.permute(0, 2, 3, 1).reshape(N * 64 * 64, 256)
            B = mask.permute(0, 2, 3, 1).reshape(N * 64 * 64, 1)
            X = torch.linalg.lstsq(A, B).solution.reshape(1, 256, 1, 1)
        return X.detach()
    
    def apply_token(self, features, token):
        return up_to_256(torch.sum(features * token, dim=(1), keepdim=True))

    def init(self, image, bbox, refine_scale=1., refine_iters=1):
        scale = max(image.shape[0], image.shape[1]) / 256.
        features, mask = self.compute_features_and_mask(image, bbox)
        token = self.fit_lstsq_mask_token(features, down_to_64(mask))
        self.token = token
        mask_token = self.apply_token(features, token)
        mask_bbox = mask_to_box(mask_token) * scale
        mask_token_refined = self.refine_mask(image, mask_bbox, mask_token, scale=refine_scale, iters=refine_iters, set_image=False)
        self._feature_buffer = []
        self._mask_buffer = []
        self._feature_buffer.append(features)
        self._mask_buffer.append(down_to_64(mask_token_refined))
        return mask_bbox, mask_token_refined

    def update(self, image, refine_scale=1., refine_iters=1):
        if not hasattr(self, 'token'):
            raise RuntimeError("No token found.  Did you call tracker.init()?")
        scale = max(image.shape[0], image.shape[1]) / 256.
        features = self.compute_features(image)
        mask_token = self.apply_token(features, self.token)
        mask_bbox = mask_to_box(mask_token) * scale
        mask_token_refined = self.refine_mask(image, mask_bbox, mask_token, scale=refine_scale, iters=refine_iters)
        reg_f = torch.cat(self._feature_buffer, dim=0)
        mask_f = torch.cat(self._mask_buffer, dim=0)
        self.token = self.fit_lstsq_mask_token(reg_f, mask_f)
        self._feature_buffer.append(features)
        self._mask_buffer.append(down_to_64(mask_token_refined))
        if len(self._feature_buffer) > self._token_history:
            self._feature_buffer = self._feature_buffer[1:]
            self._mask_buffer = self._mask_buffer[1:]
        return mask_bbox, mask_token_refined