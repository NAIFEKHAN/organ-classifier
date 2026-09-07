"""
extract_echo_frames.py

Heart data (e.g. EchoNet-Dynamic, CAMUS) comes as ultrasound video clips,
not static images. This pulls individual frames out of each video so they
can be used as training images for the "heart" class.

Usage:
    python extract_echo_frames.py --video_dir path/to/echo_videos \
        --out_dir ../../data/raw/heart \
        --frame_step 10
"""

import argparse
from pathlib import Path
import cv2
from tqdm import tqdm

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True,
                         help="Should be data/raw/heart")
    parser.add_argument("--frame_step", type=int, default=10,
                         help="Save one frame every N frames to avoid near-duplicate images")
    parser.add_argument("--max_frames_per_video", type=int, default=15,
                         help="Cap frames per video so one long clip doesn't dominate the dataset")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_files = [p for p in video_dir.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS]
    if not video_files:
        raise SystemExit(f"No video files found in {video_dir}")

    print(f"Found {len(video_files)} echo videos")
    saved_count = 0

    for video_path in tqdm(video_files, desc="Videos"):
        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0
        saved_for_this_video = 0

        while cap.isOpened() and saved_for_this_video < args.max_frames_per_video:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % args.frame_step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                out_path = out_dir / f"{video_path.stem}_frame{frame_idx:05d}.png"
                cv2.imwrite(str(out_path), gray)
                saved_count += 1
                saved_for_this_video += 1
            frame_idx += 1

        cap.release()

    print(f"Saved {saved_count} frames to {out_dir}")


if __name__ == "__main__":
    main()
