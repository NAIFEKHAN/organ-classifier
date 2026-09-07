"""
extract_ct_slices.py

Many CT datasets (LiTS liver, Pancreas-CT, etc.) ship as 3D volumes in NIfTI
(.nii / .nii.gz) format rather than ready-to-use 2D images. This script pulls
out the axial slices that actually contain the organ and saves them as PNGs
that organize_dataset.py can then split into train/val/test.

Usage:
    python extract_ct_slices.py --volume_dir path/to/nifti_volumes \
        --out_dir ../../data/raw/liver \
        --organ_label liver \
        --slice_step 3

Notes:
- If your dataset also includes segmentation masks (label volumes), pass
  --mask_dir so slices with no organ pixels present are skipped. This
  filters out empty/irrelevant slices automatically.
- Without a mask, the script takes the middle third of slices in each
  volume, which is a reasonable heuristic for many organs since datasets
  are typically cropped to the abdomen already.
"""

import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm


def normalize_to_uint8(slice_2d: np.ndarray, window_min=-200, window_max=250) -> np.ndarray:
    """Apply a CT windowing (soft tissue window by default) and scale to 0-255."""
    clipped = np.clip(slice_2d, window_min, window_max)
    scaled = (clipped - window_min) / (window_max - window_min) * 255.0
    return scaled.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume_dir", type=str, required=True,
                         help="Folder containing .nii or .nii.gz CT volumes")
    parser.add_argument("--mask_dir", type=str, default=None,
                         help="Optional folder of matching segmentation masks (same filenames)")
    parser.add_argument("--out_dir", type=str, required=True,
                         help="Where to save extracted PNG slices (should be data/raw/<organ_name>)")
    parser.add_argument("--organ_label", type=str, required=True)
    parser.add_argument("--slice_step", type=int, default=3,
                         help="Save every Nth slice to avoid near-duplicate frames")
    parser.add_argument("--window_min", type=int, default=-200, help="CT windowing lower bound (HU)")
    parser.add_argument("--window_max", type=int, default=250, help="CT windowing upper bound (HU)")
    args = parser.parse_args()

    volume_dir = Path(args.volume_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = Path(args.mask_dir) if args.mask_dir else None

    volume_files = sorted(list(volume_dir.glob("*.nii")) + list(volume_dir.glob("*.nii.gz")))
    if not volume_files:
        raise SystemExit(f"No .nii/.nii.gz files found in {volume_dir}")

    print(f"Found {len(volume_files)} volumes for class '{args.organ_label}'")
    saved_count = 0

    for vol_path in tqdm(volume_files, desc="Volumes"):
        img = nib.load(str(vol_path))
        data = img.get_fdata()  # shape: (H, W, num_slices)
        num_slices = data.shape[2]

        mask_data = None
        if mask_dir is not None:
            mask_path = mask_dir / vol_path.name
            if mask_path.exists():
                mask_data = nib.load(str(mask_path)).get_fdata()

        if mask_data is not None:
            slice_indices = [i for i in range(num_slices) if np.any(mask_data[:, :, i] > 0)]
        else:
            # Heuristic: use the middle third of the volume where the target
            # organ is most likely to appear, since these datasets are
            # generally pre-cropped to the abdomen/chest region.
            start, end = num_slices // 3, 2 * num_slices // 3
            slice_indices = list(range(start, end))

        for i in slice_indices[::args.slice_step]:
            slice_2d = data[:, :, i]
            img_uint8 = normalize_to_uint8(slice_2d, args.window_min, args.window_max)
            out_path = out_dir / f"{vol_path.stem.replace('.nii', '')}_slice{i:04d}.png"
            Image.fromarray(img_uint8).save(out_path)
            saved_count += 1

    print(f"Saved {saved_count} slices to {out_dir}")


if __name__ == "__main__":
    main()
