#!/usr/bin/env python3
"""Train the LMSCA-YOLO model on a Varroa Ultralytics dataset YAML."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_ULTRALYTICS = REPO_ROOT / "ultralytics"
DEFAULT_MODEL = REPO_ROOT / "models_config/yolov8/lmsca/yolov8_lmsca_all.yaml"


def load_yolo():
    sys.path.insert(0, str(LOCAL_ULTRALYTICS))
    from ultralytics import YOLO

    return YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LMSCA-YOLO on a Varroa dataset.")
    parser.add_argument("--data-yaml", required=True, help="Ultralytics dataset YAML for Varroa.")
    parser.add_argument("--model-yaml", default=str(DEFAULT_MODEL), help="LMSCA model YAML.")
    parser.add_argument("--pretrained", default=None, help="Optional checkpoint for partial weight loading.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint to resume.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--project", default="runs/varroa")
    parser.add_argument("--name", default="lmsca_yolo")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    YOLO = load_yolo()
    if args.resume:
        model = YOLO(args.resume)
        resume = True
    else:
        model = YOLO(args.model_yaml)
        if args.pretrained:
            model.load(args.pretrained)
        resume = False
    model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
        seed=args.seed,
        resume=resume,
    )


if __name__ == "__main__":
    main()
