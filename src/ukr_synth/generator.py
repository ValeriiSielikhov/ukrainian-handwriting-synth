"""Main dataset generation orchestrator."""

import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from ukr_synth.augmentations import get_augmentation_pipeline
from ukr_synth.config import DEFAULT_CONFIG
from ukr_synth.fonts import get_fonts_for_text, get_valid_fonts
from ukr_synth.logger import get_logger
from ukr_synth.renderer import apply_skew, render_line

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Worker function (top-level so it is picklable for multiprocessing)
# ---------------------------------------------------------------------------


def _generate_single(
    idx: int,
    text: str,
    font_path: str,
    font_size: int,
    skew_angle: float,
    augment_prob: float,
    output_dir: str,
    subfolder: str,
) -> dict | None:
    """Render one sentence and save the image. Returns metadata dict or None."""
    try:
        img = render_line(text, font_path, font_size)
        img = apply_skew(img, skew_angle)

        if augment_prob > 0:
            pipeline = get_augmentation_pipeline(prob=augment_prob)
            img = pipeline(image=img)["image"]

        font_name = Path(font_path).stem
        filename = f"{font_name}_{idx:06d}.png"
        img_path = Path(output_dir) / "images" / subfolder / filename
        cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return {"filename": f"images/{subfolder}/{filename}", "text": text}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _prepare_tasks(
    sentences: list[str],
    valid_fonts: list[Path],
    num_per_sentence: int,
    config: dict,
    subfolder: str,
) -> list[tuple[int, str, str, int, float, str]]:
    """Prepare all generation tasks."""
    tasks: list[tuple[int, str, str, int, float, str]] = []
    idx = 0
    for sentence in sentences:
        usable = get_fonts_for_text(valid_fonts, sentence)
        if not usable:
            continue
        for _ in range(num_per_sentence):
            font_path = str(random.choice(usable))
            font_size = random.randint(config["font_size_min"], config["font_size_max"])
            skew_angle = random.uniform(config["skew_min"], config["skew_max"])
            tasks.append((idx, sentence, font_path, font_size, skew_angle, subfolder))
            idx += 1
    return tasks


def _generate_single_process(
    tasks: list[tuple[int, str, str, int, float, str]],
    augment_prob: float,
    output_dir: Path,
) -> list[dict]:
    """Generate images in single-process mode."""
    logger.info("Generating images in single-process mode")
    results: list[dict] = []
    for task in tqdm(tasks, desc="Generating images"):
        i, text, fp, fs, sa, subfolder = task
        res = _generate_single(
            i, text, fp, fs, sa, augment_prob, str(output_dir), subfolder
        )
        if res is not None:
            results.append(res)
    return results


def _generate_multi_process(
    tasks: list[tuple[int, str, str, int, float, str]],
    augment_prob: float,
    output_dir: Path,
    workers: int,
) -> list[dict]:
    """Generate images in multi-process mode."""
    logger.info(f"Generating images in multi-process mode with {workers} workers")
    results: list[dict] = []
    futures = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for task in tasks:
            i, text, fp, fs, sa, subfolder = task
            fut = pool.submit(
                _generate_single,
                i,
                text,
                fp,
                fs,
                sa,
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
            "Run scripts/download_fonts.sh first."
        )
    logger.info(f"Found {len(valid_fonts)} valid font(s) in '{fonts_dir}' directory")

    creation_date = datetime.now().strftime("%Y%m%d")
    all_tasks: list[tuple[int, str, str, int, float, str]] = []

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

    if workers <= 1:
        all_results = _generate_single_process(all_tasks, augment_prob, output_dir)
    else:
        all_results = _generate_multi_process(
            all_tasks, augment_prob, output_dir, workers
        )

    all_results.sort(key=lambda r: r["filename"])

    labels_path = output_dir / "labels.csv"
    df = pd.DataFrame(all_results)
    df.to_csv(labels_path, sep="\t", index=False, header=False)
    logger.info(f"Done. {len(all_results)} images saved. Labels: {labels_path}")
    return labels_path
