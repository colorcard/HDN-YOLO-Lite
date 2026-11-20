# HDN-YOLO-Lite

Implementation of HDN-YOLO-Lite, a YOLO11m-based detector for defect detection
in heavily nitrogen-doped n-type 4H-SiC wafers. The method combines
LMSCSP-DeepOnly, hard-distribution-targeted training, and NWD-based box
regression.

## Structure

```text
.
├── configs/
│   ├── hdn_yolo_lite_yolo11m.yaml   # model definition
│   └── dataset.example.yaml         # dataset template
├── hdn_yolo_lite/
│   ├── modules.py                   # LMSConv, LMSBottleneck, C3k2LMS
│   └── loss.py                      # NWD and CIoU + NWD loss
├── train.py
├── test.py
├── predict.py
├── requirements.txt
└── LICENSE
```

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.11, PyTorch 2.11.0, Ultralytics 8.4.30, CUDA 13.0.

## Dataset

The dataset is not included. Use a standard YOLO detection layout:

```text
<dataset>/
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

```bash
cp configs/dataset.example.yaml configs/dataset.yaml
# edit the dataset path
```

Classes: `0 = BPD`, `1 = TD` (TED + TSD), `2 = PLD`.

## Training

```bash
python train.py --data configs/dataset.yaml --weights yolo11m.pt --device 0
```

Defaults: AdamW, lr `0.001`, 400 epochs, patience 60, batch 4, `640×640`,
seed 0, mosaic closed for the last 8 epochs, NWD `r_s = 0.035` and
`C = 12.16`. Targeted augmentation is defined in
`train.py::TARGETED_AUGMENTATION`.

## Evaluation

```bash
python test.py \
  --weights runs/hdn_yolo_lite/hdn_yolo_lite_yolo11m/weights/best.pt \
  --data configs/dataset.yaml --split test --device 0
```

## Inference

```bash
python predict.py \
  --weights runs/hdn_yolo_lite/hdn_yolo_lite_yolo11m/weights/best.pt \
  --source /path/to/images --device 0
```

## Notes

- LMSCSP-DeepOnly is applied only at backbone layer 8 and neck layer 22.
- NWD is used only during training; inference uses the standard YOLO11 head.
- The code extends Ultralytics at runtime, without modifying `site-packages`.
- Checkpoints contain `C3k2LMS`; import `hdn_yolo_lite` before loading them
  outside the provided scripts.

## License

AGPL-3.0. See `LICENSE`.
