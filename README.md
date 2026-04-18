# YOLO pose annotation and visualization

Multi-person COCO-17 keypoint detection on images or video with Ultralytics YOLO (YOLOv8 / YOLO11-Pose), exported as JSONL, then drawn as skeleton overlays.

## Setup

```bash
pip install -r requirements-yolo.txt
```

The visualization script needs OpenCV. If `cv2` is missing: `pip install opencv-python`.

Run the commands below from the project root (this folder).

## Important files

| File | Role |
|------|------|
| `annotate_yolo_pose.py` | Runs inference; writes `annotations.jsonl` and `run_meta.json` |
| `visualize_yolo_annotations.py` | Reads the JSONL; draws skeletons on images or video frames |
| `cv_path.py` | Unicode-safe image read/write for visualization (used by the visualize script) |
| `requirements-yolo.txt` | Python dependencies |

**Outputs (under `--output_dir` by default):**

- `annotations.jsonl` — one JSON object per image or per video frame  
- `run_meta.json` — summary of the annotation run  

## 1. Run annotation: `annotate_yolo_pose.py`


**Single video:**

```bash
python annotate_yolo_pose.py --source ./video.mp4 --output_dir ./out_yolo --model yolov8m-pose.pt
```

Useful options: `--conf`, `--device` (e.g. `0` or `cpu`), `--max_frames` (video: stop after N frames), `--save_viz` (also saves overlaid frames under `out_yolo/viz`).

## 2. Run visualization: `visualize_yolo_annotations.py`

```

**Video mode** (records include `frame_index`):

```bash
python visualize_yolo_annotations.py --annotations ./out_yolo/annotations.jsonl --video ./video.mp4 --output_dir ./out_yolo/viz2
```

Options: `--conf` (keypoint confidence threshold), `--max_frames` (only process the first N records; handy for quick tests).

**Typical workflow:** run step **1** to create `out_yolo/annotations.jsonl`, then run step **2** with the matching image or video command.
