#!/usr/bin/env python3
"""Evaluate a trained HDN-YOLO-Lite checkpoint.

Reports mAP@50, mAP@50:95, and per-class AP@50 on a YOLO-format split.
Example:
    python test.py --weights runs/hdn_yolo_lite/exp/weights/best.pt \
                   --data configs/dataset.yaml --split test --device 0
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO

from hdn_yolo_lite import install_model_builder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weights", required=True, help="Trained .pt checkpoint.")
    parser.add_argument("--data", required=True, help="Dataset YAML.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold used for AP computation.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--project", default="runs/hdn_yolo_lite_val")
    parser.add_argument("--name", default="test")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Needed if --weights points to a model YAML; harmless for .pt checkpoints.
    install_model_builder()

    model = YOLO(args.weights)
    model.info()

    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        conf=args.conf,
        iou=args.iou,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        plots=True,
    )

    print("\nEvaluation summary")
    print(f"  split      : {args.split}")
    print(f"  mAP@50     : {metrics.box.map50:.5f}")
    print(f"  mAP@50:95  : {metrics.box.map:.5f}")
    ap50_by_class = {
        int(index): float(value) for index, value in zip(metrics.box.ap_class_index, metrics.box.ap50)
    }
    for index, name in metrics.names.items():
        value = ap50_by_class.get(index)
        value_text = f"{value:.5f}" if value is not None else "n/a"
        print(f"  AP@50[{name}] : {value_text}")


if __name__ == "__main__":
    main()
