# One-command setup for Windows PowerShell.
#
# Requires a Kaggle API token first (one-time, ~1 minute) — see README
# "Step 1" or run `python src\download_datasets.py` on its own to get
# the instructions printed out.
#
# Usage (from the project root):
#     .\setup.ps1
#
# If PowerShell blocks the script, run this once in an elevated prompt:
#     Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

$ErrorActionPreference = "Stop"

Write-Host "==> Installing dependencies..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "==> Downloading datasets from Kaggle (brain, lungs, heart, liver, kidney, pancreas)..." -ForegroundColor Cyan
python src/download_datasets.py

Write-Host "==> Splitting into train/val/test..." -ForegroundColor Cyan
python src/preprocessing/organize_dataset.py --raw_dir data/raw --out_dir data/processed

Write-Host "`nSetup complete. Next: train the model (ideally in Google Colab with a GPU):" -ForegroundColor Green
Write-Host "  python src/train.py --data_dir data/processed --model_out model --img_size 224 --batch_size 32 --epochs_head 10 --epochs_finetune 10"
