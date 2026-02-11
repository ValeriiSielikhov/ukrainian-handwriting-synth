# Synthetic Handwriting Dataset Generator

A tool for generating synthetic handwritten text images from digital fonts — designed for training OCR and HTR models.

<img src="images/example_1.png" alt="Example output" style="width: 100%; max-width: 100%;">

<table style="width: 100%;">
<tr>
<td style="width: 50%;"><img src="images/example_2.png" alt="Example 2" style="width: 100%;"></td>
<td style="width: 50%;"><img src="images/example_3.png" alt="Example 3" style="width: 100%;"></td>
</tr>
<tr>
<td><img src="images/example_4.png" alt="Example 4" style="width: 100%;"></td>
<td><img src="images/example_5.png" alt="Example 5" style="width: 100%;"></td>
</tr>
<tr>
<td><img src="images/example_6.png" alt="Example 6" style="width: 100%;"></td>
<td><img src="images/example_7.png" alt="Example 7" style="width: 100%;"></td>
</tr>
<tr>
<td><img src="images/example_8.png" alt="Example 8" style="width: 100%;"></td>
<td><img src="images/example_9.png" alt="Example 9" style="width: 100%;"></td>
</tr>
</table>

## Overview

This project renders text strings using handwriting-style fonts and produces labeled image datasets. Each image is a single line of text with randomized font size, skew, and optional augmentations applied. The output is ready for use as training data for optical character recognition (OCR) or handwritten text recognition (HTR) pipelines.

The generator is built with Ukrainian in mind but can be adapted to any language by providing the appropriate text corpus and fonts.

## Features

- **Font validation** — automatically discovers and validates fonts against required character sets using `fontTools` cmap tables (no fallback glyphs)
- **Randomized rendering** — each sample varies font, size, and horizontal skew for realistic diversity
- **Augmentation pipeline** — morphological transforms, noise, geometric distortions, background artifacts, shadows, and color shifts via `albumentations` and custom transforms
- **Parallel generation** — multiprocess workers for fast dataset creation
- **Structured output** — images saved as PNGs with a tab-separated labels file mapping each image to its source text

## Project Structure

```mermaid
---
config:
  layout: dagre
---
flowchart TD
    classDef root fill:#1976d2,stroke:#0d47a1,stroke-width:3px,color:#fff
    classDef entry fill:#388e3c,stroke:#1b5e20,stroke-width:2px,color:#fff
    classDef config fill:#f57c00,stroke:#e65100,stroke-width:2px,color:#fff
    classDef module fill:#7b1fa2,stroke:#4a148c,stroke-width:2px,color:#fff
    classDef data fill:#c2185b,stroke:#880e4f,stroke-width:2px,color:#fff

    Root["."]:::root

    Root --> EntryGroup["Entry Points"]
    Root --> ConfigGroup["Configuration"]
    Root --> SourceGroup["Source Code"]
    Root --> DataGroup["Data"]

    subgraph EntryGroup[" "]
        direction TB
        Generate["generate.py<br/>CLI entry point"]:::entry
        Run["run.sh<br/>Quick-run helper"]:::entry
    end

    subgraph ConfigGroup[" "]
        direction TB
        PyProject["pyproject.toml<br/>Project metadata"]:::config
    end

    subgraph SourceGroup[" "]
        direction TB
        Src["src/"]
        UkrSynth["ukr_synth/"]:::module
        Src --> UkrSynth
        UkrSynth --> Init["__init__.py"]:::module
        UkrSynth --> ConfigPy["config.py"]:::module
        UkrSynth --> CorpusPy["corpus.py"]:::module
        UkrSynth --> FontsPy["fonts.py"]:::module
        UkrSynth --> RendererPy["renderer.py"]:::module
        UkrSynth --> AugPy["augmentations.py"]:::module
        UkrSynth --> GeneratorPy["generator.py"]:::module
        UkrSynth --> LoggerPy["logger.py"]:::module
    end

    subgraph DataGroup[" "]
        direction TB
        Fonts["fonts/<br/>Font files"]:::data
        Images["images/<br/>Examples"]:::data
    end
```

## Architecture

```mermaid
---
config:
  layout: dagre
---
flowchart TD
    classDef input fill:#1976d2,stroke:#0d47a1,stroke-width:3px,color:#fff
    classDef process fill:#f57c00,stroke:#e65100,stroke-width:2px,color:#fff
    classDef render fill:#388e3c,stroke:#1b5e20,stroke-width:2px,color:#fff
    classDef augment fill:#c2185b,stroke:#880e4f,stroke-width:2px,color:#fff
    classDef output fill:#00796b,stroke:#004d40,stroke-width:2px,color:#fff

    subgraph Input["Input Data"]
        Text["Text Corpus<br/>(sentences/phrases)"]:::input
        FontDir["Font Files<br/>(.ttf / .otf)"]:::input
    end

    subgraph Processing["Processing Pipeline"]
        Validation["Font Validation<br/>(fontTools cmap check)"]:::process
        Selection["Font Selection<br/>(per sentence)"]:::process
        Render["Rendering<br/>(PIL ImageFont)"]:::render
        Skew["Apply Skew<br/>(OpenCV warpAffine)"]:::render
        Augment{"Augmentation<br/>(optional)"}:::augment
        AugOut["Augmented Image"]:::augment
    end

    subgraph Output["Output"]
        Images["images/<br/>PNG files"]:::output
        Labels["labels.csv<br/>Tab-separated"]:::output
    end

    Text --> Validation
    FontDir --> Validation
    Validation --> Selection
    Selection --> Render
    Render --> Skew
    Skew --> Augment
    Augment -->|"apply transforms"| AugOut
    Augment -->|"skip"| AugOut
    AugOut --> Images
    AugOut --> Labels
```

## Requirements

- Python >= 3.14
- Dependencies (installed automatically):
  - `pillow` — text rendering
  - `opencv-python` — image transforms, skew
  - `numpy` — array operations
  - `albumentations` — augmentation pipeline
  - `fonttools` — font glyph validation
  - `pandas` — labels file writing
  - `tqdm` — progress bars

## Installation

Using [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

## Usage

Run generation via CLI:

```bash
python generate.py \
    --output-dir output \
    --fonts-dir fonts \
    --num-per-sentence 10 \
    --augment-prob 0.5 \
    --seed 42 \
    --workers 4
```

| Flag | Description | Default |
|---|---|---|
| `--output-dir`, `-o` | Root directory for images and labels | `output` |
| `--fonts-dir`, `-f` | Directory containing `.ttf` / `.otf` font files | `fonts` |
| `--num-per-sentence`, `-n` | Number of image variants per sentence | `10` |
| `--augment-prob`, `-a` | Probability of each augmentation stage (0 to disable) | `0.5` |
| `--seed`, `-s` | Random seed for reproducibility | `None` |
| `--workers`, `-w` | Number of parallel worker processes | CPU count |

Or use the helper script:

```bash
bash run.sh
```

## Input Data

- **Text** — a list of sentences or phrases to render. Provided as a Python list in the corpus module or any iterable of strings.
- **Fonts** — `.ttf` or `.otf` files placed in the fonts directory. Each font is automatically validated to ensure it contains proper glyphs for the target language characters — fonts with missing glyphs are excluded.

## Output

```
output/
├── images/
│   ├── FontName_000000.png
│   ├── FontName_000001.png
│   └── ...
└── labels.csv
```

- **images/** — rendered PNG images, named `{FontName}_{index}.png`
- **labels.csv** — tab-separated file with columns: image path, source text (no header)

## License

MIT
