"""
download_datasets.py

Automatically downloads a real, ready-to-use Kaggle image dataset for
each organ and drops the images straight into data/raw/<organ>/, so you
don't have to manually search Kaggle and unzip things per organ.

Every dataset below is plain JPG/PNG images (not raw video or 3D .nii
volumes), so no extra conversion step is needed for this default set.
(extract_ct_slices.py and extract_echo_frames.py are still here if you
ever swap in a raw-volume/video dataset instead.)

Since this project only classifies which ORGAN a scan shows (not any
disease), any disease/subtype subfolders in a dataset (tumor, cyst,
stone, normal, etc.) all just get dumped together into that organ's
folder, and mask/segmentation-label images are skipped.

Requires a free Kaggle account + API token — see README "Step 1" for
setup. Then:

    python src/download_datasets.py
    python src/download_datasets.py --organs brain lungs
    python src/download_datasets.py --max_per_organ 2000
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import kagglehub

# organ -> Kaggle dataset handle ("owner/dataset-slug")
DATASETS = {
    "brain": {
        "slug": "masoudnickparvar/brain-tumor-mri-dataset",
        "note": "Brain Tumor MRI Dataset",
    },
    "lungs": {
        "slug": "paultimothymooney/chest-xray-pneumonia",
        "note": "Chest X-Ray Images (Pneumonia)",
    },
    "heart": {
        "slug": "toygarr/camus-dataset",
        "note": "CAMUS echocardiography images",
    },
    "liver": {
        "slug": "andrewmvd/lits-png",
        "note": "LiTS liver CT, pre-sliced to PNG",
    },
    "kidney": {
        "slug": "nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone",
        "note": "CT KIDNEY Dataset (Normal-Cyst-Tumor-Stone)",
    },
    "pancreas": {
        "slug": "jayaprakashpondy/pancreatic-ct-images",
        "note": "Pancreatic CT Images",
    },
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
# Path substrings that mark a file as a segmentation mask / label rather
# than an actual scan image, so we skip it.
EXCLUDE_KEYWORDS = ("mask", "segmentation", "label", "ground_truth", "groundtruth", "_gt", "gt_")


def has_kaggle_credentials() -> bool:
    # Current default: a single KGAT_... access token.
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    if (Path.home() / ".kaggle" / "access_token").exists():
        return True
    if (Path.home() / ".kaggle" / "access_token.txt").exists():
        return True
    # Legacy: username + key (kaggle.json or its two env vars).
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def print_credentials_help():
    print("No Kaggle API credentials found.\n")
    print("Get a free API token (one-time setup):")
    print("  1. Log in at https://www.kaggle.com -> click your profile -> Settings -> API")
    print("  2. Click 'Create New Token' -> this shows a token starting with KGAT_...")
    print("  3. Either:")
    print("       - set it as an environment variable:")
    print("           macOS/Linux:      export KAGGLE_API_TOKEN=KGAT_xxxxxxxx")
    print("           Windows PowerShell: $env:KAGGLE_API_TOKEN=\"KGAT_xxxxxxxx\"")
    print(r"       - or save it to ~/.kaggle/access_token")
    print(r"         (Windows PowerShell: C:\Users\<you>\.kaggle\access_token)")
    print("\n  (Older accounts can instead use 'Legacy API Credentials' -> kaggle.json,")
    print(r"   placed at ~/.kaggle/kaggle.json, or KAGGLE_USERNAME + KAGGLE_KEY env vars.)")


def collect_images(root: Path, max_images=None):
    found = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        lower_path = str(path).lower()
        if any(k in lower_path for k in EXCLUDE_KEYWORDS):
            continue
        found.append(path)
    if max_images:
        found = found[:max_images]
    return found


def download_organ(organ: str, slug: str, out_root: Path, max_images=None) -> int:
    print(f"\n--- {organ} ({slug}) ---")
    cache_path = Path(kagglehub.dataset_download(slug))
    images = collect_images(cache_path, max_images)

    if not images:
        print(f"  WARNING: no images found under {cache_path}. "
              f"The dataset's internal layout may have changed — open "
              f"https://www.kaggle.com/datasets/{slug} to check it manually.")
        return 0

    out_dir = out_root / organ
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img_path in enumerate(images):
        dest = out_dir / f"{organ}_{i:06d}{img_path.suffix.lower()}"
        shutil.copy2(img_path, dest)

    print(f"  Copied {len(images)} images -> {out_dir}")
    return len(images)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--organs", nargs="+", choices=list(DATASETS.keys()),
                         default=list(DATASETS.keys()),
                         help="Which organs to download (default: all 6)")
    parser.add_argument("--data_dir", type=str, default="data/raw",
                         help="Where to write data/raw/<organ>/ folders")
    parser.add_argument("--max_per_organ", type=int, default=None,
                         help="Cap the number of images kept per organ "
                              "(default: no cap, keep everything found)")
    args = parser.parse_args()

    if not has_kaggle_credentials():
        print_credentials_help()
        sys.exit(1)

    out_root = Path(args.data_dir)
    summary = {}
    for organ in args.organs:
        cfg = DATASETS[organ]
        try:
            summary[organ] = download_organ(organ, cfg["slug"], out_root, args.max_per_organ)
        except Exception as e:
            print(f"  FAILED for {organ}: {e}")
            summary[organ] = 0

    print("\n=== Download summary ===")
    print(f"{'Organ':<10}{'Images':<10}Source")
    for organ, count in summary.items():
        print(f"{organ:<10}{count:<10}{DATASETS[organ]['note']}")

    failed = [o for o, c in summary.items() if c == 0]
    if failed:
        print(f"\n{len(failed)} organ(s) got 0 images: {failed}.")
        print("Kaggle datasets occasionally get renamed/restructured — check the")
        print("dataset page for that organ and update its slug in DATASETS if needed.")
        sys.exit(1)

    print("\nAll organs downloaded. Next step:")
    print("  python src/preprocessing/organize_dataset.py --raw_dir data/raw --out_dir data/processed")


if __name__ == "__main__":
    main()
