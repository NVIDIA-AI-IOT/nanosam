import torch
import torch.nn as nn
from typing import Tuple
import timm

class ImageEncoderCNN_256(nn.Module):
    def __init__(self, 
            model_name: str = "resnet18",
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

        self.up_3 = nn.Sequential(
            nn.Conv2d(256 + 128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 3, 2, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.proj = nn.Conv2d(
            256 + 64, 
            feature_dim, 
            3,
            padding=1
        )

        if pos_embedding:
            self.register_parameter(
                "pos_embedding", 
                nn.Parameter(1e-5*torch.randn(1, feature_dim, *feature_shape))
            )
        else:
            self.pos_embedding = None
        
        
    def forward(self, x):
        x = self.backbone(x)
        z = torch.cat([x[-2], self.up_1(x[-1])], dim=1)
        z = torch.cat([x[-3], self.up_2(z)], dim=1)
        z = torch.cat([x[-4], self.up_3(z)], dim=1)
        x = self.proj(z)
        if self.pos_embedding is not None:
            x = x + self.pos_embedding
        return x
    

class ImageEncoderCNN_512(nn.Module):
    def __init__(self, 
            model_name: str = "resnet18",
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

        channels = self.backbone.feature_info.channels()

        self.up_1 = nn.Sequential(
            nn.Conv2d(channels[-1], 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 3, 2, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.up_2 = nn.Sequential(
            nn.Conv2d(256 + channels[-2], 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 3, 2, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.proj = nn.Sequential(
            nn.Conv2d(256 + channels[-3], 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, feature_dim, 1, padding=0)
        )

        if pos_embedding:
            self.register_parameter(
                "pos_embedding", 
                nn.Parameter(1e-5*torch.randn(1, feature_dim, *feature_shape))
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
    

class ImageEncoderCNN_1024(nn.Module):
    def __init__(self, 
            model_name: str = "resnet18",
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

        channels = self.backbone.feature_info.channels()

        self.up_1 = nn.Sequential(
            nn.Conv2d(channels[-1], 256, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(256, 256, 3, 2, 1, 1),
            nn.GELU()
        )

        self.proj = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            # nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, feature_dim, 1, padding=0)
        )

        if pos_embedding:
            self.register_parameter(
                "pos_embedding", 
                nn.Parameter(1e-5*torch.randn(1, feature_dim, *feature_shape))
            )
        else:
            self.pos_embedding = None
        
        
    def forward(self, x):
        x = self.backbone(x)
        # z = torch.cat([x[-2], self.up_1(x[-1])], dim=1)
        x = self.proj(self.up_1(x[-1]))
        if self.pos_embedding is not None:
            x = x + self.pos_embedding
        return x