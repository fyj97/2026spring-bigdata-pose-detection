#!/usr/bin/env python3
"""
Draw COCO-17 skeletons from annotations.jsonl produced by annotate_yolo_pose.py.

  pip install -r requirements-yolo.txt

Image mode (each record's "source" is an image path):
  python visualize_yolo_annotations.py --annotations out\\annotations.jsonl

  Writes JPGs (and with --video, <stem>_viz.mp4) to the annotations folder unless --output_dir is set.

Video mode:
  python visualize_yolo_annotations.py --annotations out\\annotations.jsonl --video clip.mp4

  MP4: one encoded frame per JSONL line, FPS from the video; same folder as the frame JPGs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2

from cv_path import imread as imread_unicode
from cv_path import imwrite as imwrite_unicode

# COCO-17 limb pairs (indices)
COCO17_LIMBS = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
]


def draw_person(
    img: Any,
    kps: list[list[float]],
    color: tuple[int, int, int],
    conf_thr: float,
) -> None:
    for a, b in COCO17_LIMBS:
        if a >= len(kps) or b >= len(kps):
            continue
        xa, ya, ca = kps[a]
        xb, yb, cb = kps[b]
        if ca < conf_thr or cb < conf_thr:
            continue
        cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), color, 2, cv2.LINE_AA)
    for x, y, c in kps:
        if c < conf_thr:
            continue
        cv2.circle(img, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)


def colors_for_n(n: int) -> list[tuple[int, int, int]]:
    base = [
        (0, 255, 0),
        (255, 128, 0),
        (0, 128, 255),
        (255, 0, 255),
        (0, 255, 255),
        (180, 105, 255),
        (128, 255, 0),
        (255, 0, 128),
    ]
    return [base[i % len(base)] for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize YOLO pose JSONL on images or video.")
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--video", type=Path, default=None, help="Video path when records have frame_index.")
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Where to write JPGs and mp4. Default: folder containing --annotations.",
    )
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Process at most this many JSONL lines (0 = all). Handy for smoke tests.",
    )
    args = ap.parse_args()

    ann = args.annotations.expanduser().resolve()
    out_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else ann.parent
    )
    if not ann.is_file():
        print(f"Missing: {ann}", file=sys.stderr)
        print(
            "Create it by running annotate first, for example:\n"
            '  python annotate_yolo_pose.py --source ".\\your_video.mp4" --output_dir ".\\out_yolo"',
            file=sys.stderr,
        )
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = None
    vpath: Path | None = None
    v_fps = 30.0
    viz_writer: Any | None = None
    viz_video_tmp: Path | None = None
    viz_video_out: Path | None = None
    if args.video is not None:
        vpath = args.video.expanduser().resolve()
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            print(f"Cannot open video: {vpath}", file=sys.stderr)
            sys.exit(1)
        v_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not (v_fps > 1e-3):
            v_fps = 30.0
        viz_video_out = out_dir / f"{vpath.stem}_viz.mp4"

    next_frame_to_read = 0
    n_done = 0

    def grab_video_frame(fi: int) -> Any | None:
        """Sequential read when JSONL frame_index is 0,1,2,...; seek only on backward jumps."""
        nonlocal next_frame_to_read
        fi = int(fi)
        if fi < next_frame_to_read:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            next_frame_to_read = fi
        while next_frame_to_read < fi:
            cap.read()
            next_frame_to_read += 1
        ok, img = cap.read()
        if ok and img is not None:
            next_frame_to_read = fi + 1
            return img
        return None

    try:
        with ann.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                people = rec.get("people") or []
                fi = rec.get("frame_index")
                src = rec.get("source")

                if cap is not None and fi is not None:
                    img = grab_video_frame(int(fi))
                    if img is None:
                        print(f"Skip frame {fi}", flush=True)
                        continue
                    out_path = out_dir / f"{int(fi):06d}_viz.jpg"
                else:
                    if not src:
                        continue
                    p = Path(src)
                    if not p.is_file():
                        print(f"Skip missing image: {p.name}", flush=True)
                        continue
                    img = imread_unicode(p)
                    if img is None:
                        print(f"Skip unreadable: {p.name}", flush=True)
                        continue
                    out_path = out_dir / f"{p.stem}_viz.jpg"

                cols = colors_for_n(len(people))
                for i, person in enumerate(people):
                    kps = person.get("keypoints") or []
                    if kps:
                        draw_person(img, kps, cols[i], args.conf)

                if cap is not None and fi is not None and viz_video_out is not None:
                    if viz_writer is None:
                        fd, tmp = tempfile.mkstemp(suffix=".mp4")
                        os.close(fd)
                        viz_video_tmp = Path(tmp)
                        h, w = img.shape[:2]
                        viz_writer = cv2.VideoWriter(
                            str(viz_video_tmp),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            v_fps,
                            (w, h),
                        )
                        if not viz_writer.isOpened():
                            print(
                                "Warning: could not open VideoWriter; skipping mp4.",
                                flush=True,
                            )
                            viz_writer = None
                            try:
                                viz_video_tmp.unlink(missing_ok=True)
                            except OSError:
                                pass
                            viz_video_tmp = None
                    if viz_writer is not None:
                        viz_writer.write(img)

                if not imwrite_unicode(out_path, img):
                    print(f"Failed to write: {out_path.name}", file=sys.stderr)
                else:
                    print(out_path.name, flush=True)
                n_done += 1
                if args.max_frames > 0 and n_done >= args.max_frames:
                    break
    finally:
        if viz_writer is not None:
            viz_writer.release()
        if cap is not None:
            cap.release()
        if viz_video_tmp is not None and viz_video_tmp.exists() and viz_video_out is not None:
            try:
                shutil.move(str(viz_video_tmp), str(viz_video_out))
                print(f"Wrote {viz_video_out.name}", flush=True)
            except OSError as e:
                print(
                    f"Warning: could not save viz video to {viz_video_out}: {e}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
