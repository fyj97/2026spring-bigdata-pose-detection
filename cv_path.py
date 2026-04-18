"""OpenCV read/write that work with non-ASCII paths on Windows (e.g. OneDrive + 中文目录)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def imread(path: Path | str) -> Any | None:
    p = Path(path)
    try:
        raw = p.read_bytes()
    except OSError:
        return None
    if not raw:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def imwrite(path: Path | str, img: Any) -> bool:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower() or ".jpg"
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
        ext = ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    try:
        with p.open("wb") as f:
            f.write(buf.tobytes())
    except OSError:
        return False
    return True
