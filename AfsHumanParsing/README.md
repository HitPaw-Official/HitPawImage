<!-- main documents -->

# AfsHumanParsing: combine m2fp and sapiens models

## Getting Started with AfsHumanParsing

### (1) Installation

Install Python dependencies:
```
conda create -n hp python=3.10
conda activate hp
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install opencv-python tqdm json-tricks ultralytics timm
pip install tb-nightly -i https://mirrors.aliyun.com/pypi/simple
pip install modelscope==1.15.0
pip install -U openmim
mim install mmcv==2.2.0
pip install -r requirements.txt
python setup.py develop
```

### (2) Download model files
[models](https://huggingface.co/HitPawOfficial/HitPawImage/tree/main/AfsHumanParsing) 

## Example run:
```
# For detailed code, refer to `human_parsing.py`
sapiens_ckpt = "./ahp_ckpts/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2"
m2fp_ckpt = "./ahp_ckpts/cv_resnet101_image-multiple-human-parsing"
detect_ckpt = "./ahp_ckpts/yolov8x.pt"
real_esrgan_ckpt = "./ahp_ckpts/RealESRGAN_x4plus.pth"
hp = HumanParsing(sapiens_ckpt=sapiens_ckpt,
                    m2fp_ckpt=m2fp_ckpt,
                    detect_ckpt=detect_ckpt,
                    real_esrgan_ckpt=real_esrgan_ckpt,
                    )

img_path = './test.jpg'
# img_path = './test.jpg'
img = cv2.imread(img_path)
res = hp.run_with_detect(img)
cv2.imwrite('./test_result.png', res)
```

Input is a BGR 3-channel image. Output is a single-channel mask image. Note that the mask is saved in PNG format.

## Mask output legend:

|    Part Name    |  Pixel ID  |
|       ----      |    ----    |
|     Left-arm    |     10     |
|     Skirt       |     20     |
|     Hair        |     30     |
|     Pants       |     40     |
|    Sunglasses   |     50     |
|     Left-leg    |     60     |
|     Torso-skin  |     70     |
|      Face       |     80     |
|   UpperClothes  |     90     |
|    Right-leg    |     100    | 
|     Right-arm   |     110    |
|     Coat        |     120    |
|    Left-shoe    |     130    | 
|    Right-shoe   |     140    |
|      Hat        |     150    |
|      Dress      |     160    |
|     Socks       |     170    | 
|     Scarf       |     180    | 
|    Gloves       |     190    | 
|   Apparel (Decorations) |     200    | 
|  LowerClothing  |     210    |
