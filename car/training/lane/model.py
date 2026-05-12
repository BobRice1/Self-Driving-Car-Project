import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class LaneFollower(nn.Module):
    """MobileNetV3-Small with a single-value steering regression head."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.2) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)

        self.features = backbone.features
        self.avgpool = backbone.avgpool

        in_features = backbone.classifier[0].in_features
        self.head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x.squeeze(-1)
