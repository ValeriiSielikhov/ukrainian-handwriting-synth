#!/usr/bin/env bash

# Set error handling
set -e

# 1. Install uv if not installed
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add uv to PATH for current session (important!)
    # uv is usually installed in $HOME/.cargo/bin
    export PATH="$HOME/.cargo/bin:$PATH"
fi

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# 2. Run the generator
echo "Running the generator..."
uv run generate.py \
            --output-dir output \
            --fonts-dir fonts \
            --num-per-sentence 1 \
            --augment-prob 0.5 \
            --seed 42 \
            --workers 8 \
            --background-texture-prob 0.3 \
            --image-mode rgb