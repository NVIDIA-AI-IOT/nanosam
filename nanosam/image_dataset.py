import glob
import PIL.Image
import os
from torchvision.transforms import Compose, ToTensor, Normalize, RandomResizedCrop

def default_transform():
    transform = Compose([
        RandomResizedCrop((1024, 1024)),
        ToTensor(),
        Normalize(
            mean=[123.675/255, 116.28/255, 103.53/255],
            std=[58.395/255, 57.12/255, 57.375/255]
        )
    ])
    return transform

class ImageDataset:
    def __init__(self, root: str, transform = None):
        self.root = root
        image_paths = glob.glob(os.path.join(root, "*.jpg"))
        image_paths += glob.glob(os.path.join(root, "*.png"))
        self.image_paths = image_paths

        if transform is None:
            transform = default_transform()

        self.transform = transform


    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image = PIL.Image.open(self.image_paths[index]).convert("RGB")
        image = self.transform(image)
        return image