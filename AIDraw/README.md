# **AIDraw**: *wonderful text2img project*

## Prerequisites

### Anaconda (Python 3.8)

```bash
# create conda env
conda create --name draw python==3.8

# activate conda env
conda activate draw

# (optional) set pip's Tsinghua mirror
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### Packages

```bash
# Install PyTorch
pip install "path/to/your/file/torch-1.12.1+cu113-cp38-cp38-linux_x86_64.whl"

# Install torchvision
pip install "path/to/your/file/torchvision-0.13.1+cu113-cp38-cp38-linux_x86_64.whl"

# Install necessary packages
pip install -r "path/to/your/project/requirements.txt"

# Install basicsr in develop mode
cd fs/basicsr
python setup.py develop
```

## Model

Download model weights:

- [models](https://huggingface.co/HitPawOfficial/HitPawImage/tree/main/AIDraw)

Put the downloaded model files into `fs/weights`.

## Inference

### 1) Text2img / Img2img

```bash
cd your/project/dir

# text2img
python demo_draw_v1.2_en_t2i_fast.py --prompt "Wide angle, zoomed out, portrait of a beautiful girl with long wavy hair, sheer clothes, painted by ArtGerm, edouard bisson, Roberto Ferri, Ross Tran, Pixar, low angle shot, doe eyes, dynamic pose, rain, blue and green, digital painting, cinematic lighting, trending on artstation, Unreal Engine 5, 8K" --output_dir "data/outputs/your_name" --resolution 3840 2160 --scale 20 --device cuda:0 --hq True --safe_check False

# img2img
python demo_draw_v1.2_en_i2i_fast.py --prompt "Wide angle, zoomed out, portrait of a beautiful girl with long wavy hair, sheer clothes, painted by ArtGerm, edouard bisson, Roberto Ferri, Ross Tran, Pixar, low angle shot, doe eyes, dynamic pose, rain, blue and green, digital painting, cinematic lighting, trending on artstation, Unreal Engine 5, 8K" --init_img "data/init_images/1.jpg" --strength 0.8 --output_dir "data/outputs/your_name" --resolution 1024 1024 --scale 10 --device cuda:0 --hq True --safe_check False
```

### Parameters

- `--prompt`: The text prompt used to generate images.
- `--output_dir`: Output directory to write results to.
- `--resolution`: Output resolution as `(w, h)`. Supported examples:

```text
1024 1024
1920 1080
1080 1920
2048 2048
2560 1536
1536 2560
3840 2160
2160 3840
```

- `--seed`: Random seed for generation.
  - `None`: random seed
  - `12345678`: fixed seed (any integer)
- `--batch_size`: Number of images generated per run. Default: `1`.
- `--scale`: Unconditional guidance scale.
  - `10`: low
  - `15`: mid
  - `20`: high
- `--device`: Device to run on (e.g. `cuda:0`).
- `--from_file`: If specified, load prompts from a `.txt` file. Default: `None`.
- `--hq`: If specified, export HQ images. Default: `True`.
- `--safe_check`: If specified, run safety checker. Default: `False`.
- `--replace_img`: Replacement image used by safety checker. Default: `data/replacement/1.jpeg`.