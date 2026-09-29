from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import torch
from torch import nn

@dataclass(frozen=True)
class BackboneSpec:
    name: str
    constructor: Callable
    weights_enum: object


def _specs():
    from torchvision.models import (
        resnet50, ResNet50_Weights,
        densenet121, DenseNet121_Weights,
        efficientnet_b0, EfficientNet_B0_Weights,
        convnext_tiny, ConvNeXt_Tiny_Weights,
    )
    return {
        "resnet50": BackboneSpec("resnet50", resnet50, ResNet50_Weights.IMAGENET1K_V2),
        "densenet121": BackboneSpec("densenet121", densenet121, DenseNet121_Weights.IMAGENET1K_V1),
        "efficientnet_b0": BackboneSpec("efficientnet_b0", efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1),
        "convnext_tiny": BackboneSpec("convnext_tiny", convnext_tiny, ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
    }


def available_backbones() -> list[str]:
    return list(_specs())


def build_pretrained(name: str, out_dim: int, pretrained: bool = True) -> nn.Module:
    specs = _specs()
    if name not in specs:
        raise ValueError(f"Unknown pretrained backbone {name}; choose from {sorted(specs)}")
    spec = specs[name]
    weights = spec.weights_enum if pretrained else None
    net = spec.constructor(weights=weights)
    if name.startswith("resnet"):
        dim = net.fc.in_features
        net.fc = nn.Identity()
    elif name.startswith("densenet"):
        dim = net.classifier.in_features
        net.classifier = nn.Identity()
    elif name.startswith("efficientnet"):
        dim = net.classifier[1].in_features
        net.classifier = nn.Identity()
    elif name.startswith("convnext"):
        dim = net.classifier[-1].in_features
        net.classifier = nn.Sequential(*list(net.classifier[:-1]), nn.Identity())
    else:
        raise RuntimeError(name)
    return PretrainedMultiLabel(net, dim, out_dim, name)


class PretrainedMultiLabel(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int, out_dim: int, name: str):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = int(feature_dim)
        self.name = name
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.25),
            nn.Linear(feature_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ImageNet backbones expect 3 channels; DXA is grayscale, so the normalized
        # monochrome image is repeated across RGB channels without inventing color.
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.head(self.backbone(x))


def imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    return (x - mean) / std


def freeze_backbone(model: PretrainedMultiLabel) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True


def unfreeze_last_stage(model: PretrainedMultiLabel) -> None:
    # Unfreeze only the late semantic stage to reduce overfitting on the 100-study set.
    for p in model.backbone.parameters():
        p.requires_grad = False
    b = model.backbone
    if model.name.startswith("resnet"):
        modules = [b.layer4]
    elif model.name.startswith("densenet"):
        modules = [b.features.denseblock4, b.features.norm5]
    elif model.name.startswith("efficientnet"):
        modules = [b.features[-2], b.features[-1]]
    elif model.name.startswith("convnext"):
        modules = [b.features[-2], b.features[-1]]
    else:
        modules = []
    for module in modules:
        for p in module.parameters():
            p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True
