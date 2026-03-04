import glob
import json
import os
from pathlib import Path

from ukr_synth.config import _global_skipped_letters_set, is_text_allowed
from ukr_synth.logger import get_logger

logger = get_logger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = _ROOT / "src" / "data" / "ukr_text_corpuses"
JSONL_EXT = "*.jsonl"
MAX_LINE_LENGTH = 60


def corpus_reader(num_sentences: int = -1) -> dict[str, list[str]]:
    sentences: dict[str, list[str]] = {}
    jsonl_files = glob.glob(os.path.join(CORPUS_PATH, "**", JSONL_EXT), recursive=True)
    logger.info(f"Found {len(jsonl_files)} jsonl files in {CORPUS_PATH}")
    total_sentences = 0
    for jsonl_file in jsonl_files:
        logger.info(f"Reading {jsonl_file}")
        file_sentences = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for example in f:
                data = json.loads(example)["text_plain"].split("\n")
                for line in data:
                    if line and line.strip():
                        cleaned_line = line.strip()[:MAX_LINE_LENGTH]
                        if is_text_allowed(cleaned_line):
                            file_sentences.append(cleaned_line)
                            total_sentences += 1
                            if num_sentences > 0 and total_sentences >= num_sentences:
                                logger.info(f"Reached {num_sentences} sentences, stopping")
                                filename = Path(jsonl_file).stem
                                sentences[filename] = file_sentences
                                return sentences
                        else:
                            logger.debug(
                                f"Skipped sentence with disallowed characters: {cleaned_line[:50]}..."
                            )
        filename = Path(jsonl_file).stem
        sentences[filename] = file_sentences
        logger.info(f"Loaded {len(file_sentences)} sentences from {filename}")
    logger.info(
        f"Skipped {len(_global_skipped_letters_set)} letters: {_global_skipped_letters_set}"
    )
    return sentences
