import os
import gc
import re
import argparse
import torch
import random
import numpy as np
import cv2
import PIL
from PIL import Image
from tqdm import tqdm
# diffusers==0.18.0
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, AutoencoderKL
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, StableDiffusionControlNetPipeline, StableDiffusionControlNetInpaintPipeline
from diffusers.utils import load_image
from pytorch_lightning import seed_everything
from transformers import pipeline, AutoImageProcessor, UperNetForSemanticSegmentation
from controlnet_aux import MLSDdetector, NormalBaeDetector, OpenposeDetector, HEDdetector, LineartDetector
from fs.face_swap import swap_face
from fs.face_restoration import FaceRestorerCodeFormer


ada_palette = np.asarray([
      [0, 0, 0],
      [120, 120, 120],
      [180, 120, 120],
      [6, 230, 230],
      [80, 50, 50],
      [4, 200, 3],
      [120, 120, 80],
      [140, 140, 140],
      [204, 5, 255],
      [230, 230, 230],
      [4, 250, 7],
      [224, 5, 255],
      [235, 255, 7],
      [150, 5, 61],
      [120, 120, 70],
      [8, 255, 51],
      [255, 6, 82],
      [143, 255, 140],
      [204, 255, 4],
      [255, 51, 7],
      [204, 70, 3],
      [0, 102, 200],
      [61, 230, 250],
      [255, 6, 51],
      [11, 102, 255],
      [255, 7, 71],
      [255, 9, 224],
      [9, 7, 230],
      [220, 220, 220],
      [255, 9, 92],
      [112, 9, 255],
      [8, 255, 214],
      [7, 255, 224],
      [255, 184, 6],
      [10, 255, 71],
      [255, 41, 10],
      [7, 255, 255],
      [224, 255, 8],
      [102, 8, 255],
      [255, 61, 6],
      [255, 194, 7],
      [255, 122, 8],
      [0, 255, 20],
      [255, 8, 41],
      [255, 5, 153],
      [6, 51, 255],
      [235, 12, 255],
      [160, 150, 20],
      [0, 163, 255],
      [140, 140, 140],
      [250, 10, 15],
      [20, 255, 0],
      [31, 255, 0],
      [255, 31, 0],
      [255, 224, 0],
      [153, 255, 0],
      [0, 0, 255],
      [255, 71, 0],
      [0, 235, 255],
      [0, 173, 255],
      [31, 0, 255],
      [11, 200, 200],
      [255, 82, 0],
      [0, 255, 245],
      [0, 61, 255],
      [0, 255, 112],
      [0, 255, 133],
      [255, 0, 0],
      [255, 163, 0],
      [255, 102, 0],
      [194, 255, 0],
      [0, 143, 255],
      [51, 255, 0],
      [0, 82, 255],
      [0, 255, 41],
      [0, 255, 173],
      [10, 0, 255],
      [173, 255, 0],
      [0, 255, 153],
      [255, 92, 0],
      [255, 0, 255],
      [255, 0, 245],
      [255, 0, 102],
      [255, 173, 0],
      [255, 0, 20],
      [255, 184, 184],
      [0, 31, 255],
      [0, 255, 61],
      [0, 71, 255],
      [255, 0, 204],
      [0, 255, 194],
      [0, 255, 82],
      [0, 10, 255],
      [0, 112, 255],
      [51, 0, 255],
      [0, 194, 255],
      [0, 122, 255],
      [0, 255, 163],
      [255, 153, 0],
      [0, 255, 10],
      [255, 112, 0],
      [143, 255, 0],
      [82, 0, 255],
      [163, 255, 0],
      [255, 235, 0],
      [8, 184, 170],
      [133, 0, 255],
      [0, 255, 92],
      [184, 0, 255],
      [255, 0, 31],
      [0, 184, 255],
      [0, 214, 255],
      [255, 0, 112],
      [92, 255, 0],
      [0, 224, 255],
      [112, 224, 255],
      [70, 184, 160],
      [163, 0, 255],
      [153, 0, 255],
      [71, 255, 0],
      [255, 0, 163],
      [255, 204, 0],
      [255, 0, 143],
      [0, 255, 235],
      [133, 255, 0],
      [255, 0, 235],
      [245, 0, 255],
      [255, 0, 122],
      [255, 245, 0],
      [10, 190, 212],
      [214, 255, 0],
      [0, 204, 255],
      [20, 0, 255],
      [255, 255, 0],
      [0, 153, 255],
      [0, 41, 255],
      [0, 255, 204],
      [41, 0, 255],
      [41, 255, 0],
      [173, 0, 255],
      [0, 245, 255],
      [71, 0, 255],
      [122, 0, 255],
      [0, 255, 184],
      [0, 92, 255],
      [184, 255, 0],
      [0, 133, 255],
      [255, 214, 0],
      [25, 194, 194],
      [102, 255, 0],
      [92, 0, 255],
  ])

re_attention = re.compile(r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""", re.X)
re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)


def prompt_parser(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text

    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith('\\'):
            res.append([text[1:], 1.0])
        elif text == '(':
            round_brackets.append(len(res))
        elif text == '[':
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ')' and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == ']' and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re_break, text)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1
    
    prompt = ''
    for p in res:
        if p[1] != 1:
            if p[1] >= 1.4:
                p[1] = 1.4
            prompt += "(%s)%g" %(p[0], p[1])
        else:
            prompt += p[0]
    return prompt


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    # print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.


def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd_path", type=str, default='checkpoint/v2.2/sd', help="path to checkpoint of stable diffusion model")
    parser.add_argument("--control_path", type=str, default='checkpoint/v2.2/controlnet', help="path to checkpoint of controlnet model")
    parser.add_argument("--vae_path", type=str, default='checkpoint/v2.2/vae', help="path to checkpoint of vae model")
    parser.add_argument("--sd_model_id", type=str, default='16_identitycrisis', help="stable diffusion model id")
    parser.add_argument("--control_model_id", type=str, default=None, help="controlnet diffusion model id")
    parser.add_argument("--vae_model_id", type=str, default="1_sd_vae_ft_mse", help="vae model id")
    parser.add_argument("--prompt", type=str, nargs="?", help="the prompt to render",
                        default="lineart, 2d, muscluar, absurdres, 1girl, revealing clothes, pixie wings, ((monster girl, fairy, magical, disney, see-through wings))")
    # 8k portrait of beautiful cyborg with brown hair, intricate, elegant, highly detailed, majestic, digital photography, art by artgerm and ruan jia and greg rutkowski surreal painting gold butterfly filigree, broken glass, (masterpiece, sidelighting, finely detailed beautiful eyes: 1.2), hdr
    parser.add_argument("--negative_prompt", type=str, nargs="?", help="the prompt to render",
                        default='easynegative, loli, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, (low quality, worst quality:1.4), normal quality, jpeg artifacts, signature, watermark, username, blurry, monochrome')
    parser.add_argument("--output_dir", type=str, nargs="?", help="dir to write results to",
                        default="data/outputs/2023_8_22/test_res/16_identitycrisis/1024_768/")
    parser.add_argument("--resolution", type=int, nargs="*", default=[1024,768],  # [1024,1408],[1408,1024],[2048,2816],[2816,2048]
                        help="image (width, height) in pixel space")
    parser.add_argument("--seed", type=int, default=None, help="None->random seed,any int number(12345678)->fixed seed")
    parser.add_argument("--batch_size", type=int, default=4, help="generate batch_size images once time")
    parser.add_argument("--init_img", type=str, default=None)  # "data/init_images/4.jpg"
    parser.add_argument("--mask_img", type=str, default=None, help="only use in inpainting")
    parser.add_argument("--condition_img", type=str, default=None)
    parser.add_argument("--strength", type=float, default=0.5, help="unconditional guidance scale")
    parser.add_argument("--controlnet_scale", type=float, default=1.0, help="control weight")
    parser.add_argument("--scale", type=int, default=7, help="unconditional guidance scale")
    parser.add_argument("--start", type=float, default=0.0, help="controlnet start control")
    parser.add_argument("--end", type=float, default=0.85, help="controlnet end control")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--from_file", type=str, default='', help="if specified, load prompts from this file(format->txt)")
    parser.add_argument("--safety_check", type=bool, default=False, help="if specified, export HQ images")
    parser.add_argument("--replace_img", type=str, default='data/replacement/1.jpeg',
                        help="if specified, replace nsfw with this image")
    parser.add_argument("--face_swap", type=str, default=None, help='if disabled, use None, else use image path')
    parser.add_argument("--face_restoration", type=bool, default=True)
    parser.add_argument("--face_index", type=dict, default={0}, help="face id")
    parser.add_argument("--fidelity_weight", type=float, default=0.8, help="face restore weight,[0,1]")

    opt = parser.parse_args()

    return opt


def preprocessor(image, method, mask=None):
    """
    image:['url', 'local path', 'PIL Image']
    method:["1_canny", "2_depth", "3_mlsd", "4_normalbae", "5_openpose", "6_scribble", "7_seg", "8_inpaint", "9_tile"]
    """
    image = load_image(image)
    image = np.array(image)

    if method == '1_canny':
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        control_image = Image.fromarray(image)
    elif method == '2_depth':
        depth_estimator = pipeline('depth-estimation', 'checkpoint/v2.2/processor/Depth')
        image = Image.fromarray(image)
        image = depth_estimator(image)['depth']
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        control_image = Image.fromarray(image)
    elif method == '3_mlsd':
        processor = MLSDdetector.from_pretrained('checkpoint/v2.2/processor/Annotators')  # lllyasviel/ControlNet
        control_image = processor(image)
    elif method == '4_normalbae':
        processor = NormalBaeDetector.from_pretrained("checkpoint/v2.2/processor/Annotators")
        control_image = processor(image)
    elif method == '5_openpose':
        processor = OpenposeDetector.from_pretrained('checkpoint/v2.2/processor/Annotators')  # lllyasviel/ControlNet
        control_image = processor(image, hand_and_face=True)
    elif method == '6_scribble':
        processor = HEDdetector.from_pretrained('checkpoint/v2.2/processor/Annotators')
        control_image = processor(image, scribble=True)
    elif method == '7_seg':
        image_processor = AutoImageProcessor.from_pretrained("checkpoint/v2.2/processor/upernet")
        image_segmentor = UperNetForSemanticSegmentation.from_pretrained("checkpoint/v2.2/processor/upernet")

        pixel_values = image_processor(image, return_tensors="pt").pixel_values
        with torch.no_grad():
            outputs = image_segmentor(pixel_values)
        image = Image.fromarray(image)
        seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
        for label, color in enumerate(ada_palette):
            color_seg[seg == label, :] = color
        control_image = control_image = Image.fromarray(color_seg.astype(np.uint8))
    elif method == '8_inpaint':
        image = Image.fromarray(image)
        mask = load_image(mask)
        control_image = make_inpaint_condition(image, mask)
    elif method == '9_tile':
        control_image = Image.fromarray(image)
    else:
        raise 'method not in ["1_canny", "2_depth", "3_mlsd", "4_normalbae", "5_openpose", "6_scribble", "7_seg", "8_inpaint", "9_tile"]'

    return control_image


class AIArt:
    def __init__(self, args):
        self.args = args
        self.device = None
        self.sd_model_path = None
        self.control_model_path = None
        self.vae_model_path = None
        self.t2i_pipe = None
        self.i2i_pipe = None
        self.controlnet = None
    
    def get_i2i_pipe(self, control_model_id, safety_check):
        self.delete()
        if safety_check:
            if control_model_id is not None:
                if control_model_id == '8_inpaint':
                    sd_pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(self.sd_model_path, controlnet=self.controlnet, torch_dtype=torch.float16)
                else:
                    sd_pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(self.sd_model_path, controlnet=self.controlnet, torch_dtype=torch.float16)
            else:
                sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(self.sd_model_path, revision="fp16", torch_dtype=torch.float16)
        else:
            if control_model_id is not None:
                if control_model_id == '8_inpaint':
                    sd_pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(self.sd_model_path, controlnet=self.controlnet, safety_checker=None, torch_dtype=torch.float16)
                else:
                    sd_pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(self.sd_model_path, controlnet=self.controlnet, safety_checker=None, torch_dtype=torch.float16)
            else:
                sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(self.sd_model_path, revision="fp16", torch_dtype=torch.float16, safety_checker=None)
        self.i2i_pipe = sd_pipe.to(self.device)
        del sd_pipe
        gc.collect()

    def get_t2i_pipe(self, control_model_id, safety_check):
        self.delete()
        if safety_check:
            if control_model_id is not None:
                sd_pipe = StableDiffusionControlNetPipeline.from_pretrained(self.sd_model_path, controlnet=self.controlnet, torch_dtype=torch.float16)
            else:
                sd_pipe = StableDiffusionPipeline.from_pretrained(self.sd_model_path, revision="fp16", torch_dtype=torch.float16)
        else:
            if control_model_id is not None:
                sd_pipe = StableDiffusionControlNetPipeline.from_pretrained(self.sd_model_path, controlnet=self.controlnet, safety_checker=None, torch_dtype=torch.float16)
            else:
                sd_pipe = StableDiffusionPipeline.from_pretrained(self.sd_model_path, revision="fp16", torch_dtype=torch.float16, safety_checker=None)
        self.t2i_pipe = sd_pipe.to(self.device)
        del sd_pipe
        gc.collect()
    
    def switch_model(self, sd_model_id, control_model_id, vae_model_id, init_img, safety_check, device):
        self.device = device
        self.sd_model_path = os.path.join(self.args.sd_path, sd_model_id)
        if control_model_id is not None:
            self.control_model_path = os.path.join(self.args.control_path, control_model_id)
        if control_model_id is not None:
            self.controlnet = ControlNetModel.from_pretrained(self.control_model_path, torch_dtype=torch.float16)
        if vae_model_id is not None and vae_model_id != '':
            self.vae_model_path = os.path.join(self.args.vae_path, vae_model_id)
            vae = AutoencoderKL.from_pretrained(self.vae_model_path, torch_dtype=torch.float16).to(self.device)
        if init_img:
            self.get_i2i_pipe(control_model_id, safety_check)
            if self.vae_model_path is not None:
                self.i2i_pipe.vae = vae
        else:
            self.get_t2i_pipe(control_model_id, safety_check)
            if self.vae_model_path is not None:
                self.t2i_pipe.vae = vae
    
    def delete(self):
        if self.t2i_pipe is not None:
            del self.t2i_pipe
            self.t2i_pipe = None
            
        if self.i2i_pipe is not None:
            del self.i2i_pipe
            self.i2i_pipe = None
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def load_replacement(x, replace_img):
        try:
            hwc = x.shape
            y = Image.open(replace_img).convert("RGB").resize((hwc[1], hwc[0]))
            y = (np.array(y) / 255.0).astype(x.dtype)
            assert y.shape == x.shape
            return y
        except Exception:
            return x

    @staticmethod
    def calculate_hw(resolution):
        """
        base_resolution:[512, 384], [384, 512], [512, 512], [512, 960], [960, 512], [640, 384], [384, 640], [512, 704], [704, 512]
        resolution:[1024, 768], [768, 1024],
                   [1024, 1024], [2048, 2048],
                   [1920, 1024], [1024, 1920], [3840, 2160], [2160, 3840]
                   [2048, 1536], [1536, 2048], [2560, 1536], [1536, 2560]
                   [1024, 1408], [1408, 1024], [2048, 2816], [2816, 2048]
        """
        out_w, out_h = resolution
        if out_w == out_h:
            if out_w // 4 < 512:
                input_resolution, sr_rate = (512, 512), 2
            else:
                input_resolution, sr_rate = (512, 512), 4
        elif out_w > out_h:
            if max(out_w, out_h) // 2 == 960:
                input_resolution, sr_rate = (960, 512), 2
            elif max(out_w, out_h) // 4 == 960:
                input_resolution, sr_rate = (960, 512), 4
            elif max(out_w, out_h) // 4 == 640:
                input_resolution, sr_rate = (640, 384), 4
            elif max(out_w, out_h) // 2 == 512:
                input_resolution, sr_rate = (512, 384), 2
            elif max(out_w, out_h) // 4 == 512:
                input_resolution, sr_rate = (512, 384), 4
            elif max(out_w, out_h) // 2 == 704:
                input_resolution, sr_rate = (704, 512), 2
            elif max(out_w, out_h) // 4 == 704:
                input_resolution, sr_rate = (704, 512), 4
        else:
            if max(out_w, out_h) // 2 == 960:
                input_resolution, sr_rate = (512, 960), 2
            elif max(out_w, out_h) // 4 == 960:
                input_resolution, sr_rate = (512, 960), 4
            elif max(out_w, out_h) // 4 == 640:
                input_resolution, sr_rate = (384, 640), 4
            elif max(out_w, out_h) // 2 == 512:
                input_resolution, sr_rate = (384, 512), 2
            elif max(out_w, out_h) // 4 == 512:
                input_resolution, sr_rate = (384, 512), 4
            elif max(out_w, out_h) // 2 == 704:
                input_resolution, sr_rate = (512, 704), 2
            elif max(out_w, out_h) // 4 == 704:
                input_resolution, sr_rate = (512, 704), 4

        return input_resolution, sr_rate

    @staticmethod
    def numpy_to_pil(images):
        """Convert a numpy image or a batch of images to a PIL image."""
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    @torch.no_grad()
    def text2img(self,
                 prompt='',
                 resolution=(1024, 1024),
                 guidance_scale=7,
                 seed=None,
                 output_dir='',
                 number='',
                 from_file=None,
                 replace_img='',
                 save_format='jpg',
                 ddim_steps=30,
                 batch_size=1,
                 eta=0.,
                 init_image=None,
                 strength=0.8,
                 negative_prompt=None,
                 condition_image=None,
                 controlnet_scale=1.4,
                 start=0,
                 end=0.85
                 ):
        os.makedirs(output_dir, exist_ok=True)
        if not seed:
            seed = random.randint(9999, 10000000)
        seed_everything(seed)

        if not from_file:
            assert prompt is not None
            data = [prompt] * batch_size
        else:
            with open(from_file, "r") as f:
                text = f.readlines()
                data = [t.strip() for t in text if t != '\n']
        # 0 prompt weight
        from compel import Compel
        if self.i2i_pipe is not None:
            compel_proc = Compel(tokenizer=self.i2i_pipe.tokenizer, text_encoder=self.i2i_pipe.text_encoder, truncate_long_prompts=True)
        else:
            compel_proc = Compel(tokenizer=self.t2i_pipe.tokenizer, text_encoder=self.t2i_pipe.text_encoder, truncate_long_prompts=True)

        # 1. get height/width
        (width, height), sr_rate = self.calculate_hw(resolution)
        images, nsfw = [], []
        has_nsfw_concept = None

        if init_image is not None:
            image = load_img(init_image)
            assert int(image.shape[2]), int(image.shape[3]) != (height, width)
        
        if self.args.control_model_id is not None:
            condition_image = preprocessor(condition_image, self.args.control_model_id, mask=self.args.mask_img)

        # 2. generate images
        for prompt in tqdm(data, desc="data"):
            base_count = len(os.listdir(output_dir)) + 1
            if seed is None:
                cur_generator = None
            else:
                cur_generator = torch.Generator("cuda").manual_seed(seed)
            # n_p = 'windows, canvas frame, ((disfigured)), ((bad art)), ((deformed)),((extra limbs)),((close up)),((b&w)), wierd colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), Photoshop, video game, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad anatomy, ((naked)), ((nude)), ((NSFW)), (((claws)))'
            n_p = 'EasyNegative,bad hand, (worst quality,low quality:1.4),lowres,normal quality,ugly,bad anatomy,cropped, longbody,bad hands, missing fingers,uncoordinated face,unnatural face,uncoordinated eyes,unnatural eyes,uncoordinated body,unnatural body,bad hair,long neck,lots of hands, signature,watermark,username,text,jpeg artifacts,error,artist name,trademark,title,extra digit,fewer digits,duplicate, multiple view,reference sheet,(greyscale,monochrome:1.1),(depth of field,blurry:1.2) '
            if negative_prompt is not None:
                if len(negative_prompt) == 0:
                    negative_prompt = n_p
            else:
                negative_prompt = n_p
            
            # prompt preprocess
            prompt = prompt_parser(prompt)
            negative_prompt = prompt_parser(negative_prompt)
            prompt_embeds = compel_proc(prompt)
            negative_prompt_embeds = compel_proc(negative_prompt)
            
            if init_image is not None:
                if self.args.control_model_id is not None:
                    if self.args.control_model_id == '8_inpaint':
                        init_image = load_image(init_image)
                        mask_img = load_image(self.args.mask_img)
                        res = self.i2i_pipe(
                            prompt=None,
                            negative_prompt=None,
                            generator=cur_generator,
                            image=init_image,
                            mask_image=mask_img,
                            control_image=condition_image,
                            strength=strength,
                            num_inference_steps=ddim_steps,
                            control_guidance_start=start,
                            control_guidance_end=end,
                            prompt_embeds=prompt_embeds,
                            negative_prompt_embeds=negative_prompt_embeds,
                            output_type="np",
                        )
                    else:
                        res = self.i2i_pipe(
                            prompt=None,
                            negative_prompt=None,
                            image=image,
                            control_image=condition_image,
                            width=width,
                            height=height,
                            guidance_scale=guidance_scale,
                            controlnet_conditioning_scale=controlnet_scale,
                            generator=cur_generator,
                            strength=strength,
                            num_inference_steps=ddim_steps,
                            control_guidance_start=start,
                            control_guidance_end=end,
                            prompt_embeds=prompt_embeds,
                            negative_prompt_embeds=negative_prompt_embeds,
                            output_type="np",
                            )
                    x_checked_image, has_nsfw_concept = res.images, res.nsfw_content_detected
                else:
                    x_checked_image, has_nsfw_concept = self.i2i_pipe(
                        prompt=None,
                        image=image,
                        strength=strength,
                        num_inference_steps=ddim_steps,
                        guidance_scale=guidance_scale,
                        negative_prompt=None,
                        num_images_per_prompt=1,
                        eta=eta,
                        generator=cur_generator,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        output_type="np",
                        return_dict=False
                    )
            else:
                if self.args.control_model_id is not None:
                    # condition_image = preprocessor(condition_image, self.args.control_model_id)
                    res = self.t2i_pipe(
                        prompt=None,
                        negative_prompt=None,
                        image=condition_image,
                        width=width,
                        height=height,
                        guidance_scale=guidance_scale,
                        controlnet_conditioning_scale=controlnet_scale,
                        generator=cur_generator,
                        num_inference_steps=ddim_steps,
                        control_guidance_start=start,
                        control_guidance_end=end,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        output_type="np",
                        )
                    x_checked_image, has_nsfw_concept = res.images, res.nsfw_content_detected
                else:
                    x_checked_image, has_nsfw_concept = self.t2i_pipe(
                        prompt=None,
                        height=height,
                        width=width,
                        num_inference_steps=ddim_steps,
                        guidance_scale=guidance_scale,
                        negative_prompt=None,
                        num_images_per_prompt=1,
                        eta=eta,
                        generator=cur_generator,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        output_type="np",
                        return_dict=False
                    )

            # 3. run safety check
            if has_nsfw_concept is not None:
                for i in range(len(has_nsfw_concept)):
                    if has_nsfw_concept[i]:
                        x_checked_image[i] = self.load_replacement(x_checked_image[i], replace_img)

            # 4. save images
            for x_sample in x_checked_image:
                if torch.is_tensor(x_sample):
                    x_sample = 255.0 * x_sample.permute(1, 2, 0).cpu().numpy()
                else:
                    x_sample = 255.0 * x_sample
                img_out = os.path.join(output_dir, "%s_%s.%s" % (base_count, number, save_format))
                Image.fromarray(x_sample.astype(np.uint8)).save(img_out)
                images.append(os.path.abspath(img_out))
                nsfw.append(has_nsfw_concept[0] if has_nsfw_concept else None)

            # change seed for each image
            seed = random.randint(9999, 10000000)

        return images, nsfw, sr_rate

    def draw(self, prompt, resolution, scale, seed, output_dir, number, from_file, batch_size, replace_img,
             init_image, strength, negative_prompt, condition_image, controlnet_scale, start, end):
        ori_images, nsfws, sr_rates = [], [], []
        if from_file:
            batch_size = 1
        ori_image, nsfw, sr_rate = self.text2img(prompt, resolution, scale, seed, output_dir, number, from_file, replace_img,
                                                 batch_size=batch_size, init_image=init_image,strength=strength,
                                                 negative_prompt=negative_prompt,condition_image=condition_image,
                                                 controlnet_scale=controlnet_scale, start=start, end=end)
    
        ori_images += ori_image
        nsfws += nsfw
        sr_rates += str(sr_rate)
        torch.cuda.empty_cache()

        return ori_images, nsfws, sr_rates


def batch_test_t2i():
    opt = parse_args()
    # text_1 = ["data/prompt/test/1_animal.txt", "data/prompt/test/2_landscape.txt",
    #           "data/prompt/7_landscape.txt"]
    text_1 = ["data/prompt/2023_7_10.txt"]
    res = [[1024, 1024], [2048, 2048], [1536, 2560], [2560, 1536], [1920, 1080], [1080, 1920], [3840, 2160],
           [2160, 3840], [2048, 1536], [1536, 2048], [1024, 768], [768, 1024]]
    ai = AIArt('checkpoint/v2.0', opt.device)
    from image_sr import SR
    sr = SR('checkpoint/v1.3', "cuda:0")
    for text in text_1:
        for model_id in os.listdir('checkpoint/v2.0'):
            ai.switch_model(model_id, init_img=opt.init_img)
            out = opt.output_dir + text.split('/')[-1].split('.')[0] + '/' + model_id + '/'
            for i in range(1):
                for r in res:
                    out_dir = out + '%s_%s_%s' % (r[0], r[1], i + 1)
                    # ori_images, nsfw, sr_rates = ai.draw(opt.prompt, r, opt.scale, opt.seed, out_dir, '1',
                    #                                      text, opt.batch_size, opt.replace_img)
                    ori_images, nsfw, sr_rates = ai.draw(opt.prompt, r, opt.scale, opt.seed, out_dir, '1',
                                                        text, opt.batch_size, opt.replace_img, opt.init_img, opt.strength)
                    _, hq_images = sr.run(ori_images, int(sr_rates[0]))
                    print(ori_images)
                    print(nsfw)
                    print(sr_rates)


def batch_test_i2i():
    opt = parse_args()
    text_1 = ["data/prompt/2023_7_10.txt"]  # 
    res_1 = [1024, 1024], [2048, 2048]
    res_2 = [1536, 2560]
    res_3 = [2560, 1536]
    res_4 = [1920, 1080], [3840, 2160]
    res_5 = [1080, 1920], [2160, 3840]
    res_6 = [768, 1024], [1536, 2048]
    res_7 = [1024, 768], [2048, 1536]

    
    from image_sr import SR
    sr = SR('checkpoint/v1.3', "cuda:1")

    images = [os.path.join("data/init_images", i) for i in os.listdir("data/init_images")]
    images.sort()
    for text in text_1:
        for i in range(1):
            for model_id in os.listdir('checkpoint/v2.0'):
                ai = AIArt('checkpoint/v2.0', opt.device)
                ai.switch_model(model_id, init_img=True)
                for index, res in enumerate([res_1, res_2, res_3, res_4, res_5, res_6, res_7]):
                    init_image = images[index]
                    out = opt.output_dir + text.split('/')[-1].split('.')[0] + '/' + model_id + '/'
                    if type(res) == list:
                        out_dir = out + '%s_%s_%s' % (res[0], res[1], i + 1)
                        ori_images, nsfw, sr_rates = ai.draw(opt.prompt, r, opt.scale, opt.seed, out_dir, '1',
                                                        text, opt.batch_size, opt.replace_img, init_image, opt.strength)
                        _, hq_images = sr.run(ori_images, int(sr_rates[0]))
                    else:
                        for r in res:
                            out_dir = out + '%s_%s_%s' % (r[0], r[1], i + 1)
                            ori_images, nsfw, sr_rates = ai.draw(opt.prompt, r, opt.scale, opt.seed, out_dir, '1',
                                                        text, opt.batch_size, opt.replace_img, init_image, opt.strength)
                            _, hq_images = sr.run(ori_images, int(sr_rates[0]))


def main():
    opt = parse_args()
    if opt.face_restoration:
        face = FaceRestorerCodeFormer()
    ai = AIArt(opt)
    ai.switch_model(opt.sd_model_id, opt.control_model_id, opt.vae_model_id, opt.init_img, opt.safety_check, opt.device)

    outputs = []
    # 1. genarate images
    ori_images, nsfw, sr_rates = ai.draw(opt.prompt, opt.resolution, opt.scale, opt.seed, opt.output_dir, '1',
                                         opt.from_file, opt.batch_size, opt.replace_img, opt.init_img, opt.strength,
                                         opt.negative_prompt, opt.condition_img, opt.controlnet_scale, opt.start, opt.end)
    
    # 2. face swap
    if opt.face_swap is not None and os.path.exists(opt.face_swap):
        for img in ori_images:
            res = swap_face(opt.face_swap, img, model='checkpoint/v2.2/roop/inswapper_128.onnx', faces_index=opt.face_index)
            outputs.append(res)
    
    if len(outputs) == 0:
        outputs = ori_images
    
    # 3. face restoration
    if opt.face_restoration:
        face.send_model_to(opt.device)
        for img in outputs:
            image = np.array(Image.open(img))
            restored_img = face.restore(image, w=opt.fidelity_weight)
            Image.fromarray(restored_img).save(img)
            # outputs.append(img)
        face.send_model_to("cpu")
        torch.cuda.empty_cache()


    print(outputs)
    print(nsfw)
    print(sr_rates)


if __name__ == "__main__":
    main()
    # prompt_parser('white background,1girl,fluttering petals, (depth of field, blurry, blurry background, bokeh:1.2)，')