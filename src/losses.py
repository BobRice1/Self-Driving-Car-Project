from typing import Dict

import torch
import torch.nn.functional as F


def weighted_mse_loss(
    angle_pred: torch.Tensor,
    speed_pred: torch.Tensor,
    speed_logit: torch.Tensor,
    angle_true: torch.Tensor,
    speed_true: torch.Tensor,
    angle_loss_weight: float = 2.5,
    speed_loss_weight: float = 1.0,
    angle_sample_weight_alpha: float = 0.0,
    speed_loss_type: str = "mse",
    speed_pos_weight: float = 1.0,
    speed_neg_weight: float = 1.0,
    speed_bce_weight: float = 1.0,
    speed_mse_weight: float = 0.0,
) -> Dict[str, torch.Tensor]:
    angle_true = angle_true.float().view(-1)
    speed_true = speed_true.float().view(-1)
    angle_pred = angle_pred.float().view(-1)
    speed_pred = speed_pred.float().view(-1)
    speed_logit = speed_logit.float().view(-1)

    angle_sq_error = (angle_pred - angle_true) ** 2
    if angle_sample_weight_alpha > 0.0:
        weights = 1.0 + angle_sample_weight_alpha * torch.abs(angle_true - 0.5)
        mse_angle = torch.mean(weights * angle_sq_error)
    else:
        mse_angle = torch.mean(angle_sq_error)

    mse_speed = torch.mean((speed_pred - speed_true) ** 2)
    bce_raw = F.binary_cross_entropy_with_logits(speed_logit, speed_true, reduction="none")
    speed_class_weights = torch.where(speed_true > 0.5, speed_pos_weight, speed_neg_weight)
    speed_bce = torch.mean(speed_class_weights * bce_raw)

    speed_loss_type = str(speed_loss_type).lower()
    if speed_loss_type == "mse":
        speed_loss = mse_speed
    elif speed_loss_type == "bce":
        speed_loss = speed_bce
    elif speed_loss_type == "bce_mse":
        speed_loss = speed_bce_weight * speed_bce + speed_mse_weight * mse_speed
    else:
        raise ValueError(
            f"Unknown speed_loss_type='{speed_loss_type}'. Expected one of: mse, bce, bce_mse."
        )

    loss = angle_loss_weight * mse_angle + speed_loss_weight * speed_loss
    kaggle_mse = 0.5 * (mse_angle + mse_speed)

    return {
        "loss": loss,
        "kaggle_mse": kaggle_mse,
        "mse_angle": mse_angle,
        "mse_speed": mse_speed,
        "speed_bce": speed_bce,
        "speed_loss": speed_loss,
    }
