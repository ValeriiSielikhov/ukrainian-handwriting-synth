#!/usr/bin/env bash

uv run generate.py \
            --output-dir output \
            --fonts-dir fonts \
            --num-per-sentence 1 \
            --augment-prob 0.5 \
            --seed 42 \
            --workers 8 \
            --background-texture-prob 0.5