import timm
import torch
import torch.nn as nn
from typing import Tuple


class ImageEncoderCNN(nn.Module):
    def __init__(self, 
            model_name: str,
            pretrained: bool = False,
            feature_dim: int = 256,
            feature_shape: Tuple[int, int] = (64, 64),
            pos_embedding: bool = True):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True
        )

        self.up_1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, 2, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.up_2 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 3, 2, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.proj = nn.Conv2d(
            256 + 128, 
            feature_dim, 
            3,
            padding=1
        )

        if pos_embedding:
            self.register_parameter(
                "pos_embedding", 
                nn.Parameter(torch.randn(1, feature_dim, *feature_shape))
            )
        else:
            self.pos_embedding = None
        
        
    def forward(self, x):
        x = self.backbone(x)
        z = torch.cat([x[-2], self.up_1(x[-1])], dim=1)
        z = torch.cat([x[-3], self.up_2(z)], dim=1)
        x = self.proj(z)
        if self.pos_embedding is not None:
            x = x + self.pos_embedding
        return x
