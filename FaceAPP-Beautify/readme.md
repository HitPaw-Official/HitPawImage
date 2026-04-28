# Face APP Beautify

A face beautify / makeup transfer demo and inference workflow.

## Installation

Install the required dependencies:

```bash
pip install numpy opencv-python pillow tqdm onnxruntime
```

## Model

Download the models from the following link:
[models](https://huggingface.co/HitPawOfficial/HitPawImage/tree/main/FaceAPP-Beautify/model)
[face_models.pth](https://huggingface.co/HitPawOfficial/HitPawImage/tree/main/FaceAPP-Beautify/pth)

Place the downloaded model files (including detector, parsing, and transfer ONNX weights) under the project `model/` directory so that paths such as `model/HP_FACE_DETECTOR.onnx`, `model/faceparsing.onnx`, and `model/appface_transfer_female/` / `model/appface_transfer_male/` resolve correctly.

## Usage

1. Put test images in `inputs/`.
2. Run the main entry script:

```text
appface.py
```

Example:

```bash
python appface.py -t 0 -d 1 -f "[398, 233, 715, 645]"
```

3. Results are written under `outputs/`.

## Main Parameters

- `-t` / `--type`: processing gender preset, `int`, `0` or `1`, default `0`  
  - `0`: female style set (`model/appface_transfer_female/`)  
  - `1`: male style set (`model/appface_transfer_male/`)
- `-d` / `--detail`: sharpness / detail strength, `float`, default `1`  
  - `1`: default model behavior  
  - `1.1`–`1.5`: extra sharpening layered on top
- `-f` / `--face_rect`: optional face bounding box, string parsed as a four-number list `[x1, y1, x2, y2]`  
  - If omitted, all detected faces above the confidence threshold are processed.

## Requirements

Runtime Python packages are listed in the Installation section above (`numpy`, `opencv-python`, `pillow`, `tqdm`, `onnxruntime`).
