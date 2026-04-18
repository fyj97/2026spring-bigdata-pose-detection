#!/usr/bin/env python3
"""
Multi-person skeleton annotation with Ultralytics YOLOv8 / YOLO11-Pose (pip install only).

Outputs:
  - annotations.jsonl : one line per image, or per sampled video frame
  - run_meta.json     : run configuration

Video sampling (--video_sample_interval, default 1.0 s) limits how many frames are written.

Install:
  pip install -r requirements-yolo.txt

Example:
  python annotate_yolo_pose.py --source .\\images --output_dir .\\out_yolo --model yolo11n-pose.pt
  python annotate_yolo_pose.py --source .\\video.mp4 --output_dir .\\out_yolo --model yolov8m-pose.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

COCO17_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def pack_people(r: Any) -> list[dict[str, Any]]:
    """Build people list from a single ultralytics.engine.results.Results."""
    people: list[dict[str, Any]] = []
    if r.keypoints is None:
        return people
    xy = r.keypoints.xy.cpu().numpy()
    if xy.size == 0:
        return people

    conf = r.keypoints.conf
    if conf is not None:
        cf = conf.cpu().numpy()
    else:
        cf = np.ones((xy.shape[0], xy.shape[1]), dtype=np.float32)

    boxes = None
    if r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes.xyxy.cpu().numpy()

    n = xy.shape[0]
    for i in range(n):
        kps: list[list[float]] = []
        for j in range(xy.shape[1]):
            kps.append(
                [
                    float(xy[i, j, 0]),
                    float(xy[i, j, 1]),
                    float(cf[i, j]),
                ]
            )
        entry: dict[str, Any] = {
            "person_index": i,
            "keypoints": kps,
            "keypoint_names": list(COCO17_NAMES),
        }
        if boxes is not None and i < len(boxes):
            entry["bbox_xyxy"] = [float(x) for x in boxes[i].tolist()]
        people.append(entry)
    return people


def is_video_path(p: Path) -> bool:
    return p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv"}


def video_stride_for_interval(fps: float, interval_sec: float) -> int:
    """Frames between writes; interval_sec <= 0 means every frame (stride 1)."""
    if interval_sec <= 0:
        return 1
    if fps <= 1e-3 or not np.isfinite(fps):
        fps = 30.0
    return max(1, round(float(fps) * float(interval_sec)))


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLOv8/YOLO11 multi-person pose -> JSONL")
    ap.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Image file, image directory, or video file.",
    )
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument(
        "--model",
        type=str,
        default="yolo11n-pose.pt",
        help="Pose weights (e.g. yolo11n-pose.pt, yolov8m-pose.pt). Auto-downloads if missing.",
    )
    ap.add_argument("--conf", type=float, default=0.15, help="Detection confidence threshold.")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference size (multiple of 32).")
    ap.add_argument("--device", type=str, default="", help="e.g. 0 or cpu. Empty = default.")
    ap.add_argument(
        "--save_viz",
        action="store_true",
        help="Also save overlaid frames to output_dir/viz (slower).",
    )
    ap.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Video only: stop after this many frames (0 = entire video).",
    )
    ap.add_argument(
        "--video_sample_interval",
        type=float,
        default=1.0,
        help="Video only: seconds between annotated frames (default 1). Use 0 to annotate every frame.",
    )
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Missing ultralytics. Run: pip install -r requirements-yolo.txt", file=sys.stderr)
        sys.exit(1)

    src = args.source.expanduser().resolve()
    out_root = args.output_dir.expanduser().resolve()
    if not src.exists():
        print(f"Source not found: {src}", file=sys.stderr)
        sys.exit(1)

    out_root.mkdir(parents=True, exist_ok=True)
    viz_dir = out_root / "viz"
    if args.save_viz:
        viz_dir.mkdir(parents=True, exist_ok=True)
        from cv_path import imwrite as _imwrite_unicode

        def save_plot(path: str, arr: Any) -> None:
            if not _imwrite_unicode(Path(path), arr):
                print(f"Warning: failed to write {path}", flush=True)

    else:
        save_plot = None

    device = args.device if args.device else None

    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"Failed to load model {args.model!r}: {e}", file=sys.stderr)
        print("Try: yolov8n-pose.pt or upgrade: pip install -U ultralytics", file=sys.stderr)
        sys.exit(1)

    def predict_kw() -> dict[str, Any]:
        k: dict[str, Any] = {
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
            "verbose": False,
        }
        if device is not None:
            k["device"] = device
        return k

    merged_path = out_root / "annotations.jsonl"
    meta_path = out_root / "run_meta.json"
    n_written = 0

    video_stride = 1
    is_video = src.is_file() and is_video_path(src)
    if is_video:
        try:
            import cv2
        except ImportError:
            print(
                "Video processing needs OpenCV. Install: pip install opencv-python",
                file=sys.stderr,
            )
            sys.exit(1)

    with merged_path.open("w", encoding="utf-8") as out:
        if is_video:
            cap = cv2.VideoCapture(str(src))
            if not cap.isOpened():
                print(f"Cannot open video: {src}", file=sys.stderr)
                sys.exit(1)
            v_fps = float(cap.get(cv2.CAP_PROP_FPS))
            video_stride = video_stride_for_interval(v_fps, args.video_sample_interval)
            frame_idx = 0
            try:
                while True:
                    if args.max_frames > 0 and frame_idx >= args.max_frames:
                        break
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    if frame_idx % video_stride == 0:
                        results = model.predict(source=frame, **predict_kw())
                        r = results[0]
                        people = pack_people(r)
                        rec = {
                            "format": "ultralytics_pose_coco17",
                            "model": args.model,
                            "source": str(src),
                            "frame_index": frame_idx,
                            "num_people": len(people),
                            "people": people,
                        }
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n_written += 1
                        if args.save_viz and save_plot is not None:
                            plotted = r.plot()
                            save_plot(str(viz_dir / f"{frame_idx:06d}.jpg"), plotted)
                    frame_idx += 1
            finally:
                cap.release()
        else:
            if src.is_file():
                sources = [str(src)]
            else:
                exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                sources = sorted(str(p) for p in src.iterdir() if p.suffix.lower() in exts)
                if not sources:
                    print(f"No images found under: {src}", file=sys.stderr)
                    sys.exit(1)

            results = model.predict(source=sources, **predict_kw())
            for r in results:
                people = pack_people(r)
                p = Path(r.path)
                rec = {
                    "format": "ultralytics_pose_coco17",
                    "model": args.model,
                    "source": str(p),
                    "frame_index": None,
                    "num_people": len(people),
                    "people": people,
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1

                if args.save_viz and save_plot is not None:
                    plotted = r.plot()
                    save_plot(str(viz_dir / f"{p.stem}_viz.jpg"), plotted)

    meta = {
        "source": str(src),
        "max_frames": args.max_frames if is_video else 0,
        "video_sample_interval": args.video_sample_interval if is_video else None,
        "video_frame_stride": video_stride if is_video else None,
        "model": args.model,
        "merged_annotations": str(merged_path),
        "records": n_written,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "device": device or "default",
        "keypoint_schema": "COCO-17",
        "keypoint_names": COCO17_NAMES,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {n_written} records to {merged_path}")
    print(f"Meta: {meta_path}")


if __name__ == "__main__":
    main()
