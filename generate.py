#!/usr/bin/env python3
"""CLI entry point for Ukrainian synthetic handwriting dataset generation."""

import argparse
import multiprocessing

from ukr_synth.config import BACKGROUNDS_DIR
from ukr_synth.corpus import SENTENCES
from ukr_synth.corpus_reader import corpus_reader
from ukr_synth.generator import generate_dataset


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic handwritten text dataset for Ukrainian language.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="output",
        help="Root directory to save images and labels.csv (default: output)",
    )
    parser.add_argument(
        "--fonts-dir",
        "-f",
        type=str,
        default="fonts",
        help="Directory containing .ttf/.otf font files (default: fonts)",
    )
    parser.add_argument(
        "--num-per-sentence",
        "-n",
        type=int,
        default=1,
        help="Number of image variants per sentence (default: 1)",
    )
    parser.add_argument(
        "--augment-prob",
        "-a",
        type=float,
        default=0.5,
        help="Probability of each augmentation stage, 0 to disable (default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=max(1, multiprocessing.cpu_count() - 2),
        help="Number of parallel worker processes (default: CPU count)",
    )
    parser.add_argument(
        "--backgrounds-dir",
        "-b",
        type=str,
        default=None,
        help="Directory with PNG texture backgrounds (default: use config BACKGROUNDS_DIR)",
    )
    parser.add_argument(
        "--background-texture-prob",
        type=float,
        default=0.3,
        help="Probability of applying a texture background when --backgrounds-dir is set (default: 0.3)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    sentences = corpus_reader()
    if not sentences:
        sentences = SENTENCES
    backgrounds_dir = args.backgrounds_dir or BACKGROUNDS_DIR
    generate_dataset(
        sentences=sentences,
        fonts_dir=args.fonts_dir,
        output_dir=args.output_dir,
        num_per_sentence=args.num_per_sentence,
        augment_prob=args.augment_prob,
        seed=args.seed,
        workers=args.workers,
        backgrounds_dir=backgrounds_dir,
        background_texture_prob=args.background_texture_prob,
    )


if __name__ == "__main__":
    main()
