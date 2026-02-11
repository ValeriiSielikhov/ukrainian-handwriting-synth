"""Render a single text line into a cropped handwriting-style image."""

import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ukr_synth.config import DEFAULT_CONFIG

# Cached background textures by directory path (str -> list of RGB ndarrays)
_texture_cache: dict[str, list[np.ndarray]] = {}


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


def load_background_textures(dir_path: Path) -> list[np.ndarray]:
    """Load all PNG images from dir_path as RGB arrays. Cached per directory."""
    key = str(dir_path.resolve())
    if key in _texture_cache:
        return _texture_cache[key]
    out: list[np.ndarray] = []
    if dir_path.is_dir():
        for p in sorted(dir_path.glob("*.png")):
            arr = cv2.imread(str(p))
            if arr is not None:
                out.append(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    _texture_cache[key] = out
    return out


def apply_background_texture(
    img: np.ndarray,
    textures: list[np.ndarray],
    page_color: tuple[int, int, int],
    background_threshold: int = 40,
    edge_margin_frac: float = 0.08,
) -> np.ndarray:
    """Replace background pixels in img with a random patch from one of the texture images.

    Pixels close to page_color are considered background and replaced; text pixels are kept.
    Patches are cropped away from texture edges (by edge_margin_frac) to avoid shadows/scan artifacts.
    """
    if not textures:
        return img
    h, w = img.shape[:2]
    page = np.array(page_color, dtype=np.int32)
    # Background mask: pixel close to page_color (L1 distance per channel)
    diff = np.abs(img.astype(np.int32) - page)
    background_mask = diff.sum(axis=2) <= background_threshold * 3
    background_mask = np.broadcast_to(background_mask[:, :, np.newaxis], (h, w, 3))

    tex = random.choice(textures)
    th, tw = tex.shape[:2]
    if th <= h or tw <= w:
        patch = cv2.resize(tex, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        margin_y = max(0, min(int(th * edge_margin_frac), (th - h) // 2))
        margin_x = max(0, min(int(tw * edge_margin_frac), (tw - w) // 2))
        y_lo = margin_y
        y_hi = th - h - margin_y
        x_lo = margin_x
        x_hi = tw - w - margin_x
        y0 = random.randint(y_lo, y_hi) if y_hi >= y_lo else 0
        x0 = random.randint(x_lo, x_hi) if x_hi >= x_lo else 0
        patch = tex[y0 : y0 + h, x0 : x0 + w].copy()

    if patch.shape[0] != h or patch.shape[1] != w:
        patch = cv2.resize(patch, (w, h), interpolation=cv2.INTER_LINEAR)

    result = np.where(background_mask, patch, img)
    return result.astype(np.uint8)
