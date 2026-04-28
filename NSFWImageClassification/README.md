# NSFW Image Classification with a Fine-Tuned ViT

## Model source

Fine-tuning is based on: [Falconsai/nsfw_image_detection on HuggingFace](https://huggingface.co/Falconsai/nsfw_image_detection).

## Environment setup

Create and configure the environment as follows:

```bash
conda create -n nsfw python=3.10
pip install -r requirements.txt
```

## Usage

### Train

```bash
python gather_img_to_txt.py
CUDA_VISIBLE_DEVICES=0,1 python vit_finetune_nsfw.py
```

### Inference

```bash
python afs_nsfw_img_cls.py
```

Example checkpoint paths (adjust to your machine or deployment):

[models](https://huggingface.co/HitPawOfficial/HitPawImage/tree/main/NSFWImageClassification)

### Demo

```bash
python demo.py
```

### Speed, VRAM, and demo notes

- **Speed & VRAM:** On an RTX 4090, a single image with one main person takes about **0.15 s**, with roughly **1 GB** VRAM.
- **Demo:** see the screenshot below.

![Demo illustration](nsfw-demo.png)
