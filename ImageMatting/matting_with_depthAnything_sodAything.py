import cv2
import os, time
import torch
import numpy as np
from PIL import Image, ImageFile
from torchvision import transforms
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm
from depth_anything.dpt import DepthAnything
from sod_anything.dpt import DPT_DINOv2
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from network_onnx import build_model_onnx
from network import build_model
from ultralytics import YOLO

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
use_cuda = True if DEVICE=='cuda' else False
kernel = np.ones((5,5), np.uint8)


def post_processing(img, a):
    a_dilate = cv2.dilate((a>0).astype(np.uint8), kernel) * 1.0
    img = (img * np.expand_dims(a_dilate, -1)).astype(np.uint8)
    img = Image.fromarray(img)
    a = Image.fromarray(a.astype(np.uint8)).convert('L')
    img.putalpha(a)
    return np.array(img)

def convert_to_color(image_path):
    """
    将灰度图转换为彩色图

    Parameters:
        image_path (str): 图片文件路径

    Returns:
        numpy.ndarray: 转换后的彩色图像
    """
    # img = cv2.imread(image_path)
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)
    if img is None:
        print("error image path: ", image_path)
        raise ValueError("Read image error")

    if len(img.shape) == 2 or img.shape[2] == 1:
        # 将灰度图转换为彩色图
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img

def is_solid_color(image, tolerance=20):
    """
    判断图片是否为纯色或接近纯色。

    :param image: numpy image, color
    :param tolerance: 颜色差异的容忍度，默认为20
    :return: 如果图片是纯色或接近纯色，返回True；否则返回False
    """ 
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 计算所有像素的平均颜色值
    avg_color = np.mean(gray_img)
    
    # 计算每个像素与平均颜色的差异
    diff = np.abs(gray_img - avg_color)
    # 判断所有像素的颜色差异是否都在容忍度范围内
    if np.all(diff <= tolerance):
        return True
    else:
        return False

def composite(fg, bg, a):
    fg = np.array(fg, np.float32)
    alpha = np.expand_dims(a, axis=2)
    im = alpha * fg + (1 - alpha) * bg
    im = im.astype(np.uint8)
    return im



class MattingDetect():
    def __init__(self, ckpt=None, sod_ckpt=None, portrait_ckpt=None, detect_ckpt=None, depth_ckpt=None,
                    max_size=1024,
                    ref_size=512,
                    conf=0.85,
                    seed=3):
        self.seed_torch(seed)
        self.ckpt = ckpt
        self.sod_ckpt = sod_ckpt
        self.portrait_ckpt = portrait_ckpt
        self.detect_ckpt = detect_ckpt
        self.depth_ckpt = depth_ckpt
        self.max_size = max_size
        self.ref_size = ref_size
        self.conf = conf
        self._init_session()
        self.transform = Compose([
                # Resize(
                #     width=518,
                #     height=518,
                #     resize_target=False,
                #     keep_aspect_ratio=True,
                #     ensure_multiple_of=14,
                #     resize_method='lower_bound',
                #     image_interpolation_method=cv2.INTER_LINEAR), # cv2.INTER_CUBIC
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ])
        

    def seed_torch(self, seed):
        # random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False        

    def load_matting_model(self, ckpt_path, p3m_net_arch):
        # create a model
        matting_model = build_model_onnx(p3m_net_arch, pretrained=False)
        matting_model = torch.nn.DataParallel(matting_model)
        # load ckpt
        ckpt = torch.load(ckpt_path)
        matting_model.load_state_dict(ckpt['state_dict'], strict=True)
        if torch.cuda.device_count() > 0:
            matting_model = matting_model.cuda()
        matting_model.eval()
        return matting_model
    
    def load_portrait_matting_model(self, ckpt_path):
        # create a model
        matting_model = build_model('vitae', pretrained=False)
        matting_model = torch.nn.DataParallel(matting_model)
        # load ckpt
        ckpt = torch.load(ckpt_path)
        matting_model.load_state_dict(ckpt['state_dict'], strict=True)
        if torch.cuda.device_count() > 0:
            matting_model = matting_model.cuda()
        matting_model.eval()
        return matting_model

    def load_depth_model(self):
        depth_anything = DepthAnything.from_pretrained(self.depth_ckpt).to(DEVICE)
        depth_anything.eval()
        return depth_anything
    
    def load_sod_anything_model(self):
        model = DPT_DINOv2(encoder='vits', in_chans=4, features=64, out_channels=[48, 96, 192, 384]).to("cuda")
        ckpt = torch.load(self.sod_ckpt)
        model.load_state_dict({k[7:]:v for k, v in ckpt['state_dict'].items()}, strict=True)
        model.eval()
        return model

    def _init_session(self):
        self.detect_predictor = YOLO(self.detect_ckpt, task='detect')
        self.depth_anything_model = self.load_depth_model()
        self.matting_sod_model = self.load_sod_anything_model()
        self.matting_model = self.load_matting_model(self.ckpt, 'vitae_4channel')
        self.matting_portrait_model = self.load_portrait_matting_model(self.portrait_ckpt)

    def get_model_input_size(self, W, H):
        if max(H, W) < self.ref_size or min(H, W) > self.ref_size:
            if H>=W:
                resize_w = self.ref_size
                resize_h = int(self.ref_size*H/W)
            else:
                resize_h = self.ref_size
                resize_w = int(self.ref_size*W/H)
        else:
            resize_h, resize_w = H, W

        if min(resize_h, resize_w) < 32:
            if resize_w >= resize_h:
                resize_w = int(resize_w / resize_h * 32)
                resize_h = 32
            else:
                resize_h = int(resize_h / resize_w * 32)
                resize_w = 32

        if max(resize_h, resize_w) > self.max_size:
            if resize_h >= resize_w:
                new_h = self.max_size
                new_w = int(self.max_size*resize_w/resize_h)
            else:
                new_w = self.max_size
                new_h = int(self.max_size*resize_h/resize_w)  
            new_h = new_h - (new_h % 32)
            new_w = new_w - (new_w % 32) 
        else:                 
            new_h = resize_h - (resize_h % 32)
            new_w = resize_w - (resize_w % 32)
        return new_w, new_h      
        
    def depth_anything_run(self, raw_image):
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        image = self.transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            depth = self.depth_anything_model(image)
        # depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = depth[None][0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        depth = depth.cpu().numpy()
        return depth
    
    def sod_anything_run(self, img, depth):
        img = img / 255.
        scale_img = np.concatenate([img, depth[:, :, np.newaxis]], axis=2)
        if use_cuda:
            tensor_img = torch.from_numpy(scale_img.astype(np.float32)[:, :, :]).permute(2, 0, 1).cuda()
        else:
            tensor_img = torch.from_numpy(scale_img.astype(np.float32)[:, :, :]).permute(2, 0, 1)
        tensor_img = tensor_img.unsqueeze(0)
        with torch.no_grad():
            matte = self.matting_sod_model(tensor_img)
        matte = matte.data.cpu().numpy()[0,0,:,:]
        matte = (matte * 255).astype(np.uint8)
        return matte

    def matting(self, im, mask):
        model = self.matting_model
        H, W, _ = im.shape
        new_w, new_h = self.get_model_input_size(W, H)
        scale_img = Image.fromarray(im)
        scale_img= scale_img.resize((new_w, new_h), Image.ANTIALIAS)
        scale_img = np.array(scale_img)

        scale_mask = Image.fromarray(mask)
        scale_mask = scale_mask.resize((new_w, new_h), Image.ANTIALIAS)
        scale_mask = np.array(scale_mask)[:, :, np.newaxis]
        scale_img = np.concatenate([scale_img, scale_mask], axis=2)
        if use_cuda:
            tensor_img = torch.from_numpy(scale_img.astype(np.float32)[:, :, :]).permute(2, 0, 1).cuda()
        else:
            tensor_img = torch.from_numpy(scale_img.astype(np.float32)[:, :, :]).permute(2, 0, 1)
        input_t = tensor_img / 255.
        input_t = input_t.unsqueeze(0)
        with torch.no_grad():
            _, _, matte = model(input_t)[:3]
        matte = matte.data.cpu().numpy()[0,0,:,:]
        matte = matte * matte
        matte = (matte * 255).astype(np.uint8)
        matte = Image.fromarray(matte)
        matte = matte.resize((W, H), Image.ANTIALIAS)
        matte = np.array(matte) 
        return matte
    
    def matting_portrait(self, im):
        H, W, _ = im.shape
        new_w, new_h = self.get_model_input_size(W, H)
        scale_img = Image.fromarray(im)
        scale_img= scale_img.resize((new_w, new_h), Image.ANTIALIAS)
        scale_img = np.array(scale_img)
        if use_cuda:
            tensor_img = torch.from_numpy(scale_img.astype(np.float32)[:, :, :]).permute(2, 0, 1).cuda()
        else:
            tensor_img = torch.from_numpy(scale_img.astype(np.float32)[:, :, :]).permute(2, 0, 1)
        input_t = tensor_img / 255.0
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        input_t = normalize(input_t)
        input_t = input_t.unsqueeze(0)
        with torch.no_grad():
            _, _, matte = self.matting_portrait_model(input_t)[:3]
        matte = matte.data.cpu().numpy()[0,0,:,:]
        matte = matte * matte
        matte = (matte * 255).astype(np.uint8)
        matte = Image.fromarray(matte)
        matte = matte.resize((W, H), Image.ANTIALIAS)
        matte = np.array(matte)
        return matte
        
    def get_mask_box(self, mask, rate=0.1):
        h, w = mask.shape[:2]
        mask = (mask>5).astype(np.int32)
        if np.sum(mask) == 0:
            return 0, 0, w, h  
        y_coords, x_coords = np.nonzero(mask)      
        x_min = x_coords.min()  
        x_max = x_coords.max()  
        y_min = y_coords.min()  
        y_max = y_coords.max()
        enlarge_x, enlarge_h = int(rate*(x_max-x_min)), int(rate*(y_max-y_min)) 
        x_min = max(0, x_min-enlarge_x)
        x_max = min(w, x_max+enlarge_x)
        y_min = max(0, y_min-enlarge_h)
        y_max = min(h, y_max+enlarge_h)
        if x_max-x_min <= 32 or y_max-y_min <= 32:
            return 0, 0, w, h
        return x_min, y_min, x_max, y_max
    
    def detect_portrait(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # input is BGR, detection input is RGB
        results = self.detect_predictor(img, conf=self.conf, classes=[0])
        boxes = results[0].boxes
        return True if len(boxes) > 0 else False
    
    def run(self, img):
        H, W = img.shape[:2]
        is_solid = is_solid_color(img)
        if is_solid:
            matte_out = (255*np.ones((H, W))).astype(np.uint8)
            return matte_out
        img_518 = cv2.resize(img, (518, 518))
        # depth anything
        depth_out = self.depth_anything_run(img_518)
        # sod 
        mask_out = self.sod_anything_run(img_518, depth_out)
        mask_out = cv2.resize(mask_out, (W, H)) 
        x1, y1, x2, y2 = self.get_mask_box(mask_out)
        # matting
        matte_out = np.zeros_like(mask_out)
        img_cut = img[y1:y2, x1:x2, :]
        mask_cut = mask_out[y1:y2, x1:x2]

        if self.detect_portrait(img_cut.copy()):
            matte_cut = self.matting_portrait(img_cut)
        else:
            matte_cut = self.matting(img_cut, mask_cut)

        matte_out[y1:y2, x1:x2] = matte_cut

        return matte_out


if __name__ == '__main__':
    sod_ckpt = "ckpts/SodAnything_vits_24w_removeRelu_best_ckpt.pth"
    ckpt = "ckpts/Fourchannel_maskChannel_24w_normalize255_onnx_epoch10.pth"
    portrait_ckpt = "ckpts/portrate_selflabelFFinetune_fullAndCut_dp_size512_ImageANTIALIAS_adduserimage_best_ckpt.pth"
    detect_ckpt = "ckpts/yolov8m.pt"
    depth_ckpt = "ckpts/models--LiheYoung--depth_anything_vits14"
    md = MattingDetect(ckpt=ckpt, sod_ckpt=sod_ckpt, portrait_ckpt=portrait_ckpt, detect_ckpt=detect_ckpt, depth_ckpt=depth_ckpt)
    _ = md.run(convert_to_color('./images/first_run.jpg'))  #为了能够避免加载模型导致第一次运行抠图模型时间过长，在创建实例后先运行一次 
    img_path = './images/test.jpg'
    img = convert_to_color(img_path) 
    mask_out = md.run(img)  # mask_out为alpha通道
    out_img = post_processing(img, mask_out)
    cv2.imwrite('./images/test_result.png', out_img)

