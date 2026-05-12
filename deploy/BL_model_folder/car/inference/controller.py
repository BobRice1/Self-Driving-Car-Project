"""Priority-based decision controller for the self-driving car."""
from __future__ import annotations

CRUISE_SPEED = 35

LEFT_BIAS = -15
RIGHT_BIAS = 15

ANGLE_MIN = 50
ANGLE_MAX = 120
ANGLE_STRAIGHT = 90


def normalised_to_car_angle(angle_norm: float) -> int:
    """Convert [0, 1] normalised angle to car units [50, 120]."""
    return int(round(ANGLE_MIN + max(0.0, min(1.0, angle_norm)) * (ANGLE_MAX - ANGLE_MIN)))


def decide(
    lane_angle_norm: float,
    arrow: str = "none",
    obstacle_in_lane: bool = False,
    cruise_speed: int = CRUISE_SPEED,
) -> tuple[int, int]:
    """Return (angle_car_units, speed_car_units) using priority rules.

    Priority (highest first):
        1. Obstacle in lane  -> stop
        2. Arrow left/right   -> apply steering bias
        3. Default            -> lane-following at cruise speed
    """
    angle = normalised_to_car_angle(lane_angle_norm)
    speed = cruise_speed

    if obstacle_in_lane:
        return angle, 0

    if arrow == "left":
        angle = max(ANGLE_MIN, angle + LEFT_BIAS)
    elif arrow == "right":
        angle = min(ANGLE_MAX, angle + RIGHT_BIAS)

    return angle, speed
