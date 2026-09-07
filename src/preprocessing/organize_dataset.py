"""
organize_dataset.py

Takes raw images organized as:
    data/raw/<organ_name>/*.png|jpg|jpeg

and splits them into:
    data/processed/train/<organ_name>/...
    data/processed/val/<organ_name>/...
    data/processed/test/<organ_name>/...

Usage:
    python organize_dataset.py --raw_dir ../../data/raw --out_dir ../../data/processed \
        --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15
"""

import argparse
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def collect_images(class_dir: Path):
    return [p for p in class_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, required=True,
                         help="Path to data/raw containing one folder per organ class")
    parser.add_argument("--out_dir", type=str, required=True,
                         help="Path to data/processed where train/val/test folders will be created")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6, \
        "train/val/test ratios must sum to 1.0"

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    class_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        raise SystemExit(f"No class folders found in {raw_dir}. "
                          f"Expected structure: {raw_dir}/<organ_name>/*.png")

    print(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")

    summary = {}
    for class_dir in class_dirs:
        class_name = class_dir.name
        images = collect_images(class_dir)
        if len(images) < 3:
            print(f"  WARNING: '{class_name}' has only {len(images)} images — skipping (need at least 3).")
            continue

        train_imgs, temp_imgs = train_test_split(
            images, train_size=args.train_ratio, random_state=args.seed
        )
        relative_val_size = args.val_ratio / (args.val_ratio + args.test_ratio)
        val_imgs, test_imgs = train_test_split(
            temp_imgs, train_size=relative_val_size, random_state=args.seed
        )

        splits = {"train": train_imgs, "val": val_imgs, "test": test_imgs}
        for split_name, split_imgs in splits.items():
            split_dir = out_dir / split_name / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for img_path in tqdm(split_imgs, desc=f"{class_name}/{split_name}", leave=False):
                shutil.copy2(img_path, split_dir / img_path.name)

        summary[class_name] = {
            "total": len(images),
            "train": len(train_imgs),
            "val": len(val_imgs),
            "test": len(test_imgs),
        }

    print("\nDataset split complete:")
    print(f"{'Class':<15}{'Total':<8}{'Train':<8}{'Val':<8}{'Test':<8}")
    for class_name, counts in summary.items():
        print(f"{class_name:<15}{counts['total']:<8}{counts['train']:<8}{counts['val']:<8}{counts['test']:<8}")


if __name__ == "__main__":
    main()
