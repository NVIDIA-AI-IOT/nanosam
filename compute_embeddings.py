import argparse
import cv2
import PIL.Image
import numpy as np
import glob
import tqdm
import torch
import os
import albumentations as A
import timm
from mobile_sam import sam_model_registry
from albumentations.pytorch.transforms import ToTensorV2
from typing import Optional
from torch.utils.data import DataLoader


def sam_transform():
    size = 1024
    return A.Compose([
        A.LongestMaxSize(size),
        A.PadIfNeeded(
            min_height=size, 
            min_width=size, 
            border_mode=0, 
            value=(0,0,0),
            position=A.PadIfNeeded.PositionType.TOP_LEFT
        ),
        A.Normalize(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            max_pixel_value=1.
        ),
        ToTensorV2()
    ])


class ImageDataset:
    def __init__(self, root: str, transform: Optional[A.Compose] = None):
        self.root = root
        image_paths = glob.glob(os.path.join(root, "*.jpg"))
        image_paths += glob.glob(os.path.join(root, "*.png"))
        print(root, len(image_paths))
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, index: int):
        image = PIL.Image.open(self.image_paths[index]).convert("RGB")
        image = np.asarray(image)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, index


parser = argparse.ArgumentParser()
parser.add_argument("images", type=str)
parser.add_argument("output_dir", type=str)
parser.add_argument("--model_type", type=str, default="vit_t")
parser.add_argument("--checkpoint", type=str, default="./weights/mobile_sam.pt")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--num_feature_samples", type=int, default=64)
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

mobile_sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
mobile_sam = mobile_sam.cuda().eval()

def sample_grid_coordinates(h, w, k):
    mx, my = np.meshgrid(np.arange(w), np.arange(h))
    mxy = np.stack([mx, my], axis=-1)
    mxy = np.reshape(mxy, (h * w, -1))
    sample = np.random.choice(h * w, size=k, replace=False)
    return mxy[sample]


def sample_features(features, k):
    h = features.shape[1]
    w = features.shape[2]
    coords = sample_grid_coordinates(h, w, k)
    features_at_coords = features[:, coords[:, 1], coords[:, 0]]
    return {
        "features": features_at_coords,
        "coords": coords
    }

dataset = ImageDataset(args.images, transform=sam_transform())

loader = DataLoader(
    dataset=dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers
)

for image, index in tqdm.tqdm(iter(loader)):

    image = image.cuda()

    with torch.no_grad():

        features = mobile_sam.image_encoder(image)

        features = features.detach().cpu().numpy()

        for i in range(len(index)):
            idx = int(index[i])
            filename = os.path.basename(dataset.image_paths[idx]).split('.')[0] + '.npy'
            path = os.path.join(args.output_dir, filename)

            features_subsampled = sample_features(features[i], k=args.num_feature_samples)

            np.save(path, features_subsampled)

