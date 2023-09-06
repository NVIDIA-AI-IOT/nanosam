import torch
import torch.nn as nn
from typing import Tuple
import timm
from .registry import register_model


class TimmImageEncoder(nn.Module):
    def __init__(self, 
            model_name: str = "resnet18",
            pretrained: bool = False,
            feature_dim: int = 256,
            feature_shape: Tuple[int, int] = (64, 64),
            neck_channels: int = 256,
            pos_embedding: bool = True):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True
        )

        channels = self.backbone.feature_info.channels()

        self.up_1 = nn.Sequential(
            nn.Conv2d(channels[-1], neck_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(neck_channels, neck_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(neck_channels, neck_channels, 3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(neck_channels, neck_channels, 3, 2, 1, 1),
            nn.GELU()
        )

        self.proj = nn.Sequential(
            nn.Conv2d(neck_channels, neck_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(neck_channels, feature_dim, 1, padding=0)
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
    

@register_model("resnet18")
def resnet18():
    return TimmImageEncoder('resnet18', pretrained=True)


@register_model("resnet34")
def resnet34():
    return TimmImageEncoder('resnet34', pretrained=True)


@register_model("resnet50")
def resnet50():
    return TimmImageEncoder('resnet50', pretrained=True)


@register_model("efficientvit_b0")
def resnet18():
    return TimmImageEncoder('efficientvit_b0', pretrained=True)