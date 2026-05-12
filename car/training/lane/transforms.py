import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

LANE_HEIGHT = 120
LANE_WIDTH = 160

CROP_RATIO = 0.4


def build_train_transforms(height: int = LANE_HEIGHT, width: int = LANE_WIDTH):
    return A.Compose(
        [
            A.Resize(height=height, width=width),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomGamma(gamma_limit=(85, 115), p=0.3),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=3, p=1.0),
                    A.GaussianBlur(blur_limit=(3, 3), p=1.0),
                ],
                p=0.15,
            ),
            A.GaussNoise(p=0.1),
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(-0.03, 0.03),
                rotate=(-3, 3),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.3,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def build_valid_transforms(height: int = LANE_HEIGHT, width: int = LANE_WIDTH):
    return A.Compose(
        [
            A.Resize(height=height, width=width),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
