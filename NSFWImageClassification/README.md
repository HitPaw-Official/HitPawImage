# NSFW Image Classification Model Fine-Tuned from ViT

## Model Source
Fine-tuned model: [HuggingFace](https://huggingface.co/Falconsai/nsfw_image_detection)

## Environment Setup
Set up the environment with the following commands:
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

[models](https://huggingface.co/HitPawOfficial/HitPawImage/tree/main/NSFWImageClassification)

### Demo  
```bash
python demo.py
```

### Inference Memory, Speed, and Demo
Speed and memory: On an RTX 4090, testing one single-person image takes 0.15s with 1GB VRAM usage.  
Demo:  
![demo](nsfw-demo.png)
