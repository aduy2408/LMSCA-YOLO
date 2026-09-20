# LMSCA-YOLO

Research code for **LMSCA-YOLO: Lightweight Multi-Scale Compressed Attention for Tiny Varroa Mite Detection**.

This is the compact paper implementation. It keeps the local Ultralytics source, the LMSCA model YAML, and small Varroa training and evaluation entry points. It is not a production package and does not include the exploratory reports, unrelated dataset experiments, slide assets, or deployment tooling from the larger working directory.

## Repository layout

```text
ultralytics/                         Local Ultralytics implementation
models_config/yolov8/lmsca/
└── yolov8_lmsca_all.yaml             LMSCA-YOLO architecture
train_eval/
├── train.py                          Varroa training entry point
└── eval.py                           Varroa evaluation entry point
```

## Model

The reference YAML combines:

- `IRDCB` lightweight inverted residual depthwise feature fusion
- `SAGRI` scale-aligned gated residual injection from `P2` into `P3`
- `LDown` for lightweight detail downsampling
- `DeformableHeadConv` before the detection branches
- `KVCompressedAttention` with compressed keys and values
- `P3` and `P4` detection heads, with KVCA ratios `r=4` and `r=2`

The implementation of these blocks lives directly in:

```text
ultralytics/ultralytics/nn/modules/block.py
ultralytics/ultralytics/nn/modules/__init__.py
ultralytics/ultralytics/nn/tasks.py
```

The model graph is:

```text
YOLOv8-style backbone
  -> IRDCB multi-scale pyramid
  -> SAGRI(P2, P3)
  -> deformable P3/P4 head convolutions
  -> KVCA on P3/P4
  -> Detect(P3, P4)
```

## Training

Provide an Ultralytics dataset YAML for the Varroa dataset:

```bash
python train_eval/train.py \\
  --data-yaml /path/to/varroa.yaml \\
  --epochs 100 \\
  --imgsz 640 \\
  --batch 4 \\
  --device 0
```

The LMSCA architecture is the default model configuration. To initialize from a compatible checkpoint:

```bash
python train_eval/train.py \\
  --data-yaml /path/to/varroa.yaml \\
  --pretrained /path/to/checkpoint.pt
```

To resume a run:

```bash
python train_eval/train.py \\
  --data-yaml /path/to/varroa.yaml \\
  --resume /path/to/last.pt
```

## Evaluation

```bash
python train_eval/eval.py \\
  --weights /path/to/best.pt \\
  --data-yaml /path/to/varroa.yaml \\
  --split test \\
  --device 0
```

## Source lineage

The LMSCA blocks were synchronized from the authoritative Hugging Face dataset repository:

```text
TLMHoang/custom_yolo_code_full_ver
```

The surrounding implementation follows the YOLO_FTSC Ultralytics lineage. The public implementation name for the scale-aligned injection block is `SAGRI`.

## Paper context

The manuscript reports LMSCA-YOLO results of `91.27% mAP50`, `35.66% mAP50-95`, `2.23M` parameters, and `9.31 GFLOPs` on the Bee Varroa dataset. Reproduction requires the exact dataset split, checkpoint initialization, image size, training seed, and evaluation settings used for the paper.

## Research use

This repository is intentionally small and experiment-oriented. Dataset files, checkpoints, and run outputs are external inputs and are not included here.
