"""Main dataset generation orchestrator."""

import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from ukr_synth.augmentations import get_augmentation_pipeline
from ukr_synth.config import (
    DEFAULT_CONFIG,
    INK_COLOR_JITTER,
    INK_COLORS,
    PAGE_COLORS,
)
from ukr_synth.fonts import get_fonts_for_text, get_valid_fonts
from ukr_synth.logger import get_logger
from ukr_synth.renderer import (
    apply_background_texture,
    apply_skew,
    load_background_textures,
    render_line,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Per-worker state (module-level for multiprocessing)
# ---------------------------------------------------------------------------

_worker_pipeline = None
_worker_backgrounds_dir: Path | None = None
_worker_background_texture_prob: float = 0.0


def _worker_init(
    augment_prob: float,
    backgrounds_dir: str | Path | None = None,
    background_texture_prob: float = 0.0,
):
    """Initialize worker process with cached augmentation pipeline and background settings."""
    global _worker_pipeline, _worker_backgrounds_dir, _worker_background_texture_prob
    if augment_prob > 0:
        _worker_pipeline = get_augmentation_pipeline(prob=augment_prob)
    else:
        _worker_pipeline = None
    _worker_backgrounds_dir = Path(backgrounds_dir) if backgrounds_dir else None
    _worker_background_texture_prob = background_texture_prob


# ---------------------------------------------------------------------------
# Worker function (top-level so it is picklable for multiprocessing)
# ---------------------------------------------------------------------------


def _generate_single(
    idx: int,
    text: str,
    font_path: str,
    font_size: int,
    skew_angle: float,
    page_color: tuple[int, int, int],
    text_color: tuple[int, int, int],
    augment_prob: float,
    output_dir: str,
    subfolder: str,
) -> dict | None:
    """Render one sentence and save the image. Returns metadata dict or None."""
    try:
        config = {"page_color": page_color, "text_color": text_color}
        img = render_line(text, font_path, font_size, config=config)
        img = apply_skew(img, skew_angle)

        if _worker_backgrounds_dir is not None and random.random() < _worker_background_texture_prob:
            textures = load_background_textures(_worker_backgrounds_dir)
            if textures:
                img = apply_background_texture(img, textures, page_color)

        if _worker_pipeline is not None:
            img = _worker_pipeline(image=img)["image"]

        font_name = Path(font_path).stem
        filename = f"{font_name}_{idx:06d}.jpg"
        img_path = Path(output_dir) / "images" / subfolder / filename
        cv2.imwrite(
            str(img_path),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        return {"filename": f"images/{subfolder}/{filename}", "text": text}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _jitter_color(color: tuple[int, int, int], max_jitter: int) -> tuple[int, int, int]:
    """Add random per-channel offset; clamp to [0, 255]."""
    return tuple(
        max(0, min(255, c + random.randint(-max_jitter, max_jitter))) for c in color
    )


_TaskTuple = tuple[
    int, str, str, int, float, str, tuple[int, int, int], tuple[int, int, int]
]


def _prepare_tasks(
    sentences: list[str],
    valid_fonts: list[Path],
    num_per_sentence: int,
    config: dict,
    subfolder: str,
) -> list[_TaskTuple]:
    """Prepare all generation tasks with per-sample page and ink colors."""
    tasks: list[_TaskTuple] = []
    idx = 0
    for sentence in sentences:
        usable = get_fonts_for_text(valid_fonts, sentence)
        if not usable:
            continue
        for _ in range(num_per_sentence):
            font_path = str(random.choice(usable))
            font_size = random.randint(config["font_size_min"], config["font_size_max"])
            skew_angle = random.uniform(config["skew_min"], config["skew_max"])
            page_color = random.choice(PAGE_COLORS)
            text_color = _jitter_color(
                random.choice(INK_COLORS), INK_COLOR_JITTER
            )
            tasks.append(
                (
                    idx,
                    sentence,
                    font_path,
                    font_size,
                    skew_angle,
                    subfolder,
                    page_color,
                    text_color,
                )
            )
            idx += 1
    return tasks


def _generate_single_process(
    tasks: list[_TaskTuple],
    augment_prob: float,
    output_dir: Path,
    backgrounds_dir: Path | None,
    background_texture_prob: float,
) -> list[dict]:
    """Generate images in single-process mode."""
    logger.info("Generating images in single-process mode")
    _worker_init(
        augment_prob,
        backgrounds_dir=backgrounds_dir,
        background_texture_prob=background_texture_prob,
    )
    results: list[dict] = []
    for task in tqdm(tasks, desc="Generating images"):
        i, text, fp, fs, sa, subfolder, page_color, text_color = task
        res = _generate_single(
            i, text, fp, fs, sa, page_color, text_color, augment_prob, str(output_dir), subfolder
        )
        if res is not None:
            results.append(res)
    return results


def _generate_multi_process(
    tasks: list[_TaskTuple],
    augment_prob: float,
    output_dir: Path,
    workers: int,
    backgrounds_dir: Path | None,
    background_texture_prob: float,
) -> list[dict]:
    """Generate images in multi-process mode."""
    logger.info(f"Generating images in multi-process mode with {workers} workers")
    results: list[dict] = []
    initargs = (
        augment_prob,
        str(backgrounds_dir) if backgrounds_dir else None,
        background_texture_prob,
    )
    futures = {}
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_worker_init, initargs=initargs
    ) as pool:
        for task in tasks:
            i, text, fp, fs, sa, subfolder, page_color, text_color = task
            fut = pool.submit(
                _generate_single,
                i,
                text,
                fp,
                fs,
                sa,
                page_color,
                text_color,
                augment_prob,
                str(output_dir),
                subfolder,
            )
            futures[fut] = i

        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="Generating images"
        ):
            res = fut.result()
            if res is not None:
                results.append(res)
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_dataset(
    sentences: dict[str, list[str]] | list[str],
    fonts_dir: str | Path,
    output_dir: str | Path,
    num_per_sentence: int = 10,
    augment_prob: float = 0.5,
    seed: int | None = None,
    workers: int = 4,
    backgrounds_dir: str | Path | None = None,
    background_texture_prob: float = 0.3,
) -> Path:
    """Generate the full dataset.

    Parameters
    ----------
    sentences:
        Dict mapping filename to list of Ukrainian text lines, or list of text lines.
    fonts_dir:
        Directory containing .ttf/.otf font files.
    output_dir:
        Root output directory. Images are saved under ``output_dir/images/``.
    num_per_sentence:
        How many image variants to create for each sentence.
    augment_prob:
        Probability of applying each augmentation stage (0 = no aug).
    seed:
        Random seed for reproducibility.
    workers:
        Number of parallel worker processes.
    backgrounds_dir:
        Directory with PNG texture backgrounds (lined/grid/kraft). When provided,
        a fraction of images will use these as background instead of plain color.
    background_texture_prob:
        Probability (0–1) of applying a texture background per image when
        backgrounds_dir is set. Default 0.3.

    Returns
    -------
    Path to the generated ``labels.csv`` file.
    """
    logger.info("Generating dataset...")
    output_dir = Path(output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {output_dir / 'images'}")

    if seed is not None:
        logger.info(f"Setting random seed to {seed}")
        random.seed(seed)

    valid_fonts = get_valid_fonts(fonts_dir)
    if not valid_fonts:
        raise RuntimeError(
            f"No fonts with Ukrainian character support found in '{fonts_dir}'. "
        )
    logger.info(f"Found {len(valid_fonts)} valid font(s) in '{fonts_dir}' directory")

    creation_date = datetime.now().strftime("%Y%m%d")
    all_tasks: list[_TaskTuple] = []
    backgrounds_dir_resolved: Path | None = (
        Path(backgrounds_dir) if backgrounds_dir else None
    )

    if isinstance(sentences, dict):
        for filename, file_sentences in sentences.items():
            subfolder = f"{filename}_{creation_date}"
            (output_dir / "images" / subfolder).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created subfolder: {subfolder}")

            tasks = _prepare_tasks(
                file_sentences,
                valid_fonts,
                num_per_sentence,
                DEFAULT_CONFIG,
                subfolder,
            )
            all_tasks.extend(tasks)
            logger.info(f"Prepared {len(tasks)} tasks for {filename}")
    else:
        subfolder = f"dataset_{creation_date}"
        (output_dir / "images" / subfolder).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created subfolder: {subfolder}")

        all_tasks = _prepare_tasks(
            sentences,
            valid_fonts,
            num_per_sentence,
            DEFAULT_CONFIG,
            subfolder,
        )

    logger.info(f"Generating {len(all_tasks)} images total")

    # Auto-reduce workers when augmentation is heavy to prevent system overload
    cpu_count = multiprocessing.cpu_count()
    if augment_prob > 0 and workers > max(1, cpu_count // 2):
        original_workers = workers
        workers = max(1, cpu_count // 2)
        logger.info(
            f"Augmentation enabled: reducing workers from {original_workers} to {workers} "
            f"to prevent system overload"
        )

    if workers <= 1:
        all_results = _generate_single_process(
            all_tasks,
            augment_prob,
            output_dir,
            backgrounds_dir_resolved,
            background_texture_prob,
        )
    else:
        all_results = _generate_multi_process(
            all_tasks,
            augment_prob,
            output_dir,
            workers,
            backgrounds_dir_resolved,
            background_texture_prob,
        )

    all_results.sort(key=lambda r: r["filename"])

    labels_path = output_dir / "labels.csv"
    df = pd.DataFrame(all_results)
    df.to_csv(labels_path, sep="\t", index=False, header=False)
    logger.info(f"Done. {len(all_results)} images saved. Labels: {labels_path}")
    return labels_path
