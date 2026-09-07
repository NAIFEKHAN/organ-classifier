#!/usr/bin/env bash
# One-command setup for macOS/Linux.
#
# Requires a Kaggle API token first (one-time, ~1 minute) — see README
# "Step 1" or run `python src/download_datasets.py` on its own to get
# the instructions printed out.
#
# Usage (from the project root):
#     chmod +x setup.sh && ./setup.sh

set -e

echo "==> Installing dependencies..."
pip install -r requirements.txt

echo "==> Downloading datasets from Kaggle (brain, lungs, heart, liver, kidney, pancreas)..."
python src/download_datasets.py

echo "==> Splitting into train/val/test..."
python src/preprocessing/organize_dataset.py --raw_dir data/raw --out_dir data/processed

echo ""
echo "Setup complete. Next: train the model (ideally in Google Colab with a GPU):"
echo "  python src/train.py --data_dir data/processed --model_out model --img_size 224 --batch_size 32 --epochs_head 10 --epochs_finetune 10"
