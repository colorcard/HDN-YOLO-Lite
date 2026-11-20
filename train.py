#!/usr/bin/env python3
"""Train HDN-YOLO-Lite on a YOLO-format defect dataset.

The defaults reproduce the paper's training configuration:
YOLO11m + LMSCSP-DeepOnly + hard-distribution-targeted augmentation + NWD.

Example:
    python train.py --data configs/dataset.yaml --weights yolo11m.pt --device 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from hdn_yolo_lite import install_model_builder, install_nwd_loss

ROOT = Path(__file__).resolve().parent

# Hard-distribution-targeted training (paper Section 3.3).
# These values are passed unchanged to the Ultralytics trainer.
TARGETED_AUGMENTATION = {
    # Colour perturbation
    "hsv_h": 0.004,
    "hsv_s": 0.18,
    "hsv_v": 0.18,
    # Lightweight geometric perturbation
    "translate": 0.02,
    "scale": 0.07,
    "degrees": 0.2,
    "shear": 0.3,
    "perspective": 0.0005,
    # Low-ratio layout recombination
    "mosaic": 0.08,
    "fliplr": 0.15,
    # Disabled aggressive operations
    "flipud": 0.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "erasing": 0.0,
    "bgr": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--data", required=True, help="Dataset YAML (see configs/dataset.example.yaml).")
    parser.add_argument(
        "--model",
        default=str(ROOT / "configs" / "hdn_yolo_lite_yolo11m.yaml"),
        help="HDN-YOLO-Lite model YAML.",
    )
    parser.add_argument("--weights", default="yolo11m.pt", help="Pretrained YOLO11m checkpoint or empty string.")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="CUDA device id, e.g. 0, or cpu.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/hdn_yolo_lite")
    parser.add_argument("--name", default="hdn_yolo_lite_yolo11m")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nwd-ratio", type=float, default=0.035, help="NWD mixing weight r_s.")
    parser.add_argument("--nwd-constant", type=float, default=12.16, help="NWD normalization constant C.")
    parser.add_argument("--cache", choices=("none", "ram", "disk"), default="none")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing an existing output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    install_model_builder()
    install_nwd_loss(ratio=args.nwd_ratio, constant=args.nwd_constant)

    model = YOLO(args.model)
    model.train(
        data=args.data,
        pretrained=args.weights or None,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        cache=None if args.cache == "none" else args.cache,
        # Training control (paper defaults)
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.001,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        seed=args.seed,
        deterministic=True,
        amp=True,
        cos_lr=False,
        close_mosaic=8,
        # Loss weights and matching
        box=7.5,
        cls=0.5,
        dfl=1.5,
        single_cls=False,
        rect=False,
        fraction=1.0,
        plots=True,
        val=True,
        **TARGETED_AUGMENTATION,
    )


if __name__ == "__main__":
    main()
