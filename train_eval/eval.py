#!/usr/bin/env python3
"""Evaluate an LMSCA-YOLO checkpoint on a Varroa Ultralytics dataset YAML."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_ULTRALYTICS = REPO_ROOT / "ultralytics"


def load_yolo():
    sys.path.insert(0, str(LOCAL_ULTRALYTICS))
    from ultralytics import YOLO

    return YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an LMSCA-YOLO Varroa checkpoint.")
    parser.add_argument("--weights", required=True, help="Checkpoint or model weights.")
    parser.add_argument("--data-yaml", required=True, help="Ultralytics dataset YAML for Varroa.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default="runs/varroa_eval")
    parser.add_argument("--name", default="lmsca_yolo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    YOLO = load_yolo()
    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data_yaml,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
    )
    print(f"mAP50={metrics.box.map50:.6f}")
    print(f"mAP50-95={metrics.box.map:.6f}")


if __name__ == "__main__":
    main()
