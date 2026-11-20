#!/usr/bin/env python3
"""Run HDN-YOLO-Lite inference on images or a directory.

Example:
    python predict.py --weights runs/hdn_yolo_lite/exp/weights/best.pt \
                      --source /path/to/images --device 0
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO

from hdn_yolo_lite import install_model_builder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weights", required=True, help="Trained .pt checkpoint.")
    parser.add_argument("--source", required=True, help="Image, directory, or glob.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/hdn_yolo_lite_predict")
    parser.add_argument("--name", default="predict")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    install_model_builder()

    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=True,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
    )


if __name__ == "__main__":
    main()
