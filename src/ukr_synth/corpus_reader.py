import glob
import json
import os
from pathlib import Path
from random import shuffle

from ukr_synth.logger import get_logger

logger = get_logger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = _ROOT / "src" / "data" / "ukr_text_corpuses"
JSONL_EXT = "*.jsonl"
MAX_LINE_LENGTH = 150
MAX_SENTENCES = 100


def corpus_reader():
    sentences = []
    jsonl_files = glob.glob(os.path.join(CORPUS_PATH, "**", JSONL_EXT), recursive=True)
    logger.info(f"Found {len(jsonl_files)} jsonl files in {CORPUS_PATH}")
    for jsonl_file in jsonl_files:
        logger.info(f"Reading {jsonl_file}")
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for example in f:
                data = json.loads(example)["text_plain"].split("\n")
                for line in data:
                    if line and line.strip():
                        sentences.append(line.strip()[:MAX_LINE_LENGTH])
    shuffle(sentences)
    sentences = sentences[:MAX_SENTENCES]
    logger.info(f"Loaded {len(sentences)} sentences")
    return sentences
