from typing import Dict

import timm
import torch
import torch.nn as nn


class SelfDrivingRegressor(nn.Module):
    def __init__(
        self,
        backbone: str = "tf_efficientnetv2_s",
        pretrained: bool = True,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        num_features = getattr(self.backbone, "num_features", None)
        if num_features is None:
            raise ValueError(f"Unable to infer feature dimension for backbone '{backbone}'.")

        self.shared_head = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.angle_head = nn.Linear(hidden_dim, 17)
        self.speed_head = nn.Linear(hidden_dim, 1)

        self.register_buffer(
            "bin_values",
            torch.linspace(0.0, 1.0, 17, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        hidden = self.shared_head(features)

        angle_logits = self.angle_head(hidden)
        speed_logit = self.speed_head(hidden).squeeze(-1)

        angle_probs = torch.softmax(angle_logits, dim=1)
        angle_pred = torch.sum(angle_probs * self.bin_values.unsqueeze(0), dim=1)
        speed_pred = torch.sigmoid(speed_logit)

        return {
            "angle_logits": angle_logits,
            "speed_logit": speed_logit,
            "angle_pred": angle_pred,
            "speed_pred": speed_pred,
        }

