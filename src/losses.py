from typing import Dict

import torch


def weighted_mse_loss(
    angle_pred: torch.Tensor,
    speed_pred: torch.Tensor,
    angle_true: torch.Tensor,
    speed_true: torch.Tensor,
    angle_loss_weight: float = 2.5,
    speed_loss_weight: float = 1.0,
    angle_sample_weight_alpha: float = 0.0,
) -> Dict[str, torch.Tensor]:
    angle_true = angle_true.float().view(-1)
    speed_true = speed_true.float().view(-1)
    angle_pred = angle_pred.float().view(-1)
    speed_pred = speed_pred.float().view(-1)

    angle_sq_error = (angle_pred - angle_true) ** 2
    if angle_sample_weight_alpha > 0.0:
        weights = 1.0 + angle_sample_weight_alpha * torch.abs(angle_true - 0.5)
        mse_angle = torch.mean(weights * angle_sq_error)
    else:
        mse_angle = torch.mean(angle_sq_error)

    mse_speed = torch.mean((speed_pred - speed_true) ** 2)
    loss = angle_loss_weight * mse_angle + speed_loss_weight * mse_speed

    return {
        "loss": loss,
        "mse_angle": mse_angle,
        "mse_speed": mse_speed,
    }

