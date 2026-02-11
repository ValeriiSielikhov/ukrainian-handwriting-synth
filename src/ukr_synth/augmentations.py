"""Image augmentation pipeline for synthetic handwriting.

Uses albumentations for standard transforms and includes custom transforms
ported from the legacy codebase (gradient backgrounds, stains, morphology).
"""

import random

import cv2
import numpy as np
from albumentations import (
    CLAHE,
    Compose,
    GaussNoise,
    HueSaturationValue,
    ImageCompression,
    ISONoise,
    MedianBlur,
    MotionBlur,
    MultiplicativeNoise,
    OneOf,
    Perspective,
    RandomBrightnessContrast,
    RandomGamma,
    Rotate,
    SafeRotate,
    Sharpen,
    ToGray,
    ToSepia,
)
from albumentations.core.transforms_interface import ImageOnlyTransform

# ---------------------------------------------------------------------------
# Custom transforms (ported from old/transforms.py)
# ---------------------------------------------------------------------------


class Erosion(ImageOnlyTransform):
    """Morphological erosion -- makes strokes thinner."""

    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(p=p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return cv2.erode(img, np.ones((2, 2), np.uint8), iterations=1)


class Dilation(ImageOnlyTransform):
    """Morphological dilation -- makes strokes thicker."""

    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(p=p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return cv2.dilate(img, np.ones((2, 2), np.uint8), iterations=1)


class GradientBackground(ImageOnlyTransform):
    """Overlay a gradient onto the background."""

    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(p=p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        light = random.choice([True, False])
        rotate = random.randint(0, 3)
        color = random.randint(100, 200)
        h, w = img.shape[:2]

        # Create gradient directly at (h, w) size instead of (w, w) square
        # Start with a 1D gradient
        if rotate in [0, 2]:
            # Vertical gradient
            gradient_1d = np.linspace(1, 0, h)[:, np.newaxis]  # (h, 1)
            gradient = np.broadcast_to(gradient_1d, (h, w))  # (h, w)
        else:
            # Horizontal gradient
            gradient_1d = np.linspace(1, 0, w)[np.newaxis, :]  # (1, w)
            gradient = np.broadcast_to(gradient_1d, (h, w))  # (h, w)

        # Rotate if needed
        if rotate == 1:
            gradient = np.rot90(gradient, 1)
        elif rotate == 2:
            gradient = np.rot90(gradient, 2)
        elif rotate == 3:
            gradient = np.rot90(gradient, 3)

        # Expand to 3 channels
        gradient = gradient[:, :, np.newaxis]  # (h, w, 1)
        gradient = np.broadcast_to(gradient, (h, w, 3))  # (h, w, 3)

        # Create background
        bg = np.ones((h, w, 3), dtype=np.uint8) * 255
        bg = (gradient * bg + (1 - gradient) * color).astype(np.uint8)
        bg = 255 - bg

        if light:
            return cv2.add(img, bg)
        return 255 - cv2.add(255 - img, bg)


class RandomStains(ImageOnlyTransform):
    """Add random blob-shaped stains to the image."""

    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(p=p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        light = random.choice([True, False])
        color = random.randint(100, 200)
        h, w = img.shape[:2]
        rng = np.random.default_rng(seed=random.randint(0, 1000))

        noise = rng.integers(0, 255, (h, w), np.uint8, True)
        blur = cv2.GaussianBlur(
            noise, (0, 0), sigmaX=15, sigmaY=15, borderType=cv2.BORDER_DEFAULT
        )

        # Rescale intensity (replaces skimage.exposure.rescale_intensity)
        bmin, bmax = float(blur.min()), float(blur.max())
        if bmax - bmin > 0:
            stretch = ((blur.astype(np.float32) - bmin) / (bmax - bmin) * 255).astype(
                np.uint8
            )
        else:
            stretch = blur

        thresh = cv2.threshold(stretch, 175, 255, cv2.THRESH_BINARY)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.merge([mask, mask, mask])
        result = np.where(mask > 0, 255 - color, 0).astype(np.uint8)

        if light:
            return cv2.add(img, result)
        return 255 - cv2.add(255 - img, result)


class RandomBlurredStains(ImageOnlyTransform):
    """Add smooth blurred stains to the background."""

    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(p=p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        light = random.choice([True, False])
        max_color = random.randint(100, 200)
        h, w = img.shape[:2]
        rng = np.random.default_rng(seed=random.randint(0, 1000))

        noise = rng.integers(0, 255, (h, w), np.uint8, True)
        blur = cv2.GaussianBlur(
            noise, (0, 0), sigmaX=15, sigmaY=15, borderType=cv2.BORDER_DEFAULT
        )
        bg = cv2.merge([blur, blur, blur]).astype(np.float32)
        bg_min, bg_max = bg.min(), bg.max()
        if bg_max - bg_min > 0:
            bg = ((bg - bg_min) * max_color / (bg_max - bg_min)).astype(np.uint8)
        else:
            bg = bg.astype(np.uint8)

        if light:
            return cv2.add(img, bg)
        return 255 - cv2.add(255 - img, bg)


class RandomShadow(ImageOnlyTransform):
    """Cast a random diagonal shadow across the image."""

    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(p=p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        h, w = img.shape[:2]
        top_y = w * np.random.uniform()
        bot_y = w * np.random.uniform()

        img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        # Use broadcasting instead of np.mgrid to avoid allocating full arrays
        rows = np.arange(h)[:, np.newaxis]  # (h, 1)
        cols = np.arange(w)[np.newaxis, :]  # (1, w)

        shadow_mask = (rows - 0) * (bot_y - top_y) - (h - 0) * (cols - top_y) >= 0
        random_bright = 0.25 + 0.7 * np.random.uniform()

        if np.random.randint(2) == 1:
            img_hls[:, :, 1][shadow_mask] = (
                img_hls[:, :, 1][shadow_mask] * random_bright
            ).astype(np.uint8)
        else:
            img_hls[:, :, 1][~shadow_mask] = (
                img_hls[:, :, 1][~shadow_mask] * random_bright
            ).astype(np.uint8)

        result = cv2.cvtColor(img_hls, cv2.COLOR_HLS2RGB)
        return np.clip(result, 0, 255).astype(np.uint8)


class CutCharacters(ImageOnlyTransform):
    """Simulate overlapping lines by cutting and pasting a strip."""

    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(p=p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        under = random.choice([True, False])
        inv = 255 - img
        y = max(int(0.3 * inv.shape[0]), 1)
        cut = inv[:y, :] if under else inv[-y:, :]
        cut = np.roll(cut, random.randint(0, cut.shape[1]), axis=1)

        if under:
            inv[-y:, :] = cv2.add(inv[-y:, :], cut)
        else:
            inv[:y, :] = cv2.add(inv[:y, :], cut)
        return 255 - inv


class ChangeWidth(ImageOnlyTransform):
    """Randomly stretch or compress image width."""

    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(p=p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        proportion = random.uniform(0.5, 2.0)
        new_w = max(int(img.shape[1] * proportion), 1)
        return cv2.resize(img, (new_w, img.shape[0]), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Pipeline constructors
# ---------------------------------------------------------------------------


def get_augmentation_pipeline(prob: float = 0.5) -> Compose:
    """Build the full augmentation pipeline.

    Each stage is a ``OneOf`` group so that exactly one transform from each
    category is applied per image.
    """
    return Compose(
        [
            # 1. Simulate overlapping lines from adjacent rows
            CutCharacters(p=prob),
            # 2. Background artefacts
            OneOf(
                [
                    RandomStains(p=1.0),
                    RandomBlurredStains(p=1.0),
                    GradientBackground(p=1.0),
                    RandomShadow(p=1.0),
                ],
                p=prob,
            ),
            # 3. Morphology
            OneOf(
                [
                    Dilation(p=1.0),
                    Erosion(p=1.0),
                ],
                p=prob,
            ),
            # 4. Noise / compression / sharpening
            OneOf(
                [
                    CLAHE(p=1.0),
                    GaussNoise(p=1.0),
                    ISONoise(p=1.0),
                    MultiplicativeNoise(multiplier=(0.85, 1.15), p=1.0),
                    ImageCompression(quality_range=(60, 90), p=1.0),
                    Sharpen(p=1.0),
                    MotionBlur(blur_limit=7, p=1.0),
                    MedianBlur(blur_limit=5, p=1.0),
                ],
                p=prob,
            ),
            # 5. Geometric (lightweight transforms only - removed expensive ElasticTransform/GridDistortion/OpticalDistortion)
            OneOf(
                [
                    Rotate(limit=2, p=1.0),
                    SafeRotate(limit=5, p=1.0),
                    Perspective(fit_output=True, p=1.0),
                    ChangeWidth(p=1.0),
                ],
                p=prob,
            ),
            # 6. Color / brightness
            OneOf(
                [
                    RandomBrightnessContrast(p=1.0),
                    RandomGamma(gamma_limit=(50, 150), p=1.0),
                    HueSaturationValue(p=1.0),
                    ToGray(p=1.0),
                    ToSepia(p=1.0),
                ],
                p=prob,
            ),
        ]
    )
