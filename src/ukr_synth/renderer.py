"""Render a single text line into a cropped handwriting-style image."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ukr_synth.config import DEFAULT_CONFIG


def render_line(
    text: str,
    font_path: str | Path,
    font_size: int,
    config: dict | None = None,
) -> np.ndarray:
    """Render *text* using *font_path* at *font_size* and return a cropped image.

    Returns an RGB ``np.ndarray`` with white background and dark text,
    cropped tightly around the text with a small margin.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    margin = cfg["margin"]
    page_color: tuple[int, int, int] = cfg["page_color"]
    text_color: tuple[int, int, int] = cfg["text_color"]

    font = ImageFont.truetype(str(font_path), font_size)

    x, y = 0, 0
    x_min, y_min, x_max, y_max = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox(
        (x, y), text, font=font
    )
    text_w = x_max - x_min
    text_h = y_max - y_min

    canvas_w = text_w + margin[0] * 2
    canvas_h = text_h + margin[1] * 2
    canvas_w = max(canvas_w, 10)
    canvas_h = max(canvas_h, 10)
    img_arr = np.full((canvas_h, canvas_w, 3), fill_value=page_color, dtype=np.uint8)
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)
    draw.text(
        (-x_min + margin[0], -y_min + margin[1]), text, fill=text_color, font=font
    )
    return np.array(img)


def apply_skew(img: np.ndarray, angle: float) -> np.ndarray:
    """Apply a horizontal shear (skew) to the image.

    Parameters
    ----------
    img:
        3-channel image with white background (255) and dark text.
    angle:
        Shear factor in range roughly [-1.5, 1.5].  Positive values
        skew the text to the right.
    """
    # Invert so text is bright on dark, avoids white artefacts at borders
    inv = 255 - img
    h, w = inv.shape[:2]

    extra = int(h * abs(angle))
    if angle > 0:
        padded = np.concatenate([inv, np.zeros((h, extra, 3), dtype=np.uint8)], axis=1)
    else:
        padded = np.concatenate([np.zeros((h, extra, 3), dtype=np.uint8), inv], axis=1)

    M = np.float32([[1, -angle, 0], [0, 1, 0]])
    skewed = cv2.warpAffine(
        padded,
        M,
        (padded.shape[1], padded.shape[0]),
        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
    )
    # Invert back
    return (255 - skewed).astype(np.uint8)
